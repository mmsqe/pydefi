#!/usr/bin/env python3
"""Gas-diff benchmark for the evm-smith Venom balance-slot peephole on
pydefi's MiniEVMContext (execution-specs EVM).

Deploys the Venom-compiled Snekmate ERC-20 twice, etches the peephole-patched
runtime onto the second, runs an identical operation sequence on each, and
reports per-operation gas: original (keccak) vs patched (NOT-slot), the delta,
and the delta as a percentage of the operation.

Each operation is measured in the **warm** (EIP-2929 hot-path) regime: the
contract account and the storage slots it touches are pre-loaded into the
message access list (discovered via a rolled-back dry run), so the figure
strips the cold-access floor and reflects what a contract that's already been
touched in the block actually pays. The delta is regime-independent (cold,
warm, or intra-tx "dirty" all save the same keccac), so it matches Foundry;
the warm denominator just makes the percentage meaningful.

Run:  /Users/mavis/Documents/crypto/pydefi/.venv/bin/python \
          scripts/bench_venom_erc20.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PYDEFI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYDEFI_ROOT))

from ethereum.forks.cancun.state import (  # noqa: E402
    TransientStorage,
    begin_transaction,
    rollback_transaction,
)
from ethereum.forks.cancun.vm import Message  # noqa: E402
from ethereum.forks.cancun.vm.interpreter import process_message  # noqa: E402
from ethereum_types.bytes import Bytes20  # noqa: E402
from ethereum_types.numeric import U256, Uint  # noqa: E402

from pydefi.venom import balance_patch as patch_venom  # noqa: E402
from pydefi.venom.erc20_abi import (  # noqa: E402
    APPROVE,
    BALANCE_OF,
    BURN,
    MINT,
    TRANSFER,
    TRANSFER_FROM,
    addr20,
    arg_addr,
    word,
)
from tests.conftest import MiniEVMContext  # noqa: E402

addr, low = arg_addr, addr20  # local names used below

ARTIFACT = patch_venom.DEFAULT_ARTIFACT

ONE = 10**18
MAX = (1 << 256) - 1
GAS = 5_000_000


def warm_gas(ctx, to: bytes, calldata: bytes) -> int:
    """Gas of executing `calldata` against `to` with the account + every
    storage slot it touches pre-warmed (EIP-2929). The touched slots are
    discovered by a dry run on a snapshot that is then rolled back, so the
    measured run mutates state exactly once (like a normal call)."""
    ts = TransientStorage()
    begin_transaction(ctx._state, ts)
    evm = ctx._apply_computation(to, ctx.deployer, calldata, 0, GAS)
    warm_addrs = set(evm.accessed_addresses)
    warm_keys = set(evm.accessed_storage_keys)
    rollback_transaction(ctx._state, ts)

    msg = Message(
        block_env=ctx._block_env,
        tx_env=ctx._make_tx_env(GAS),
        caller=Bytes20(ctx.deployer),
        target=Bytes20(to),
        current_target=Bytes20(to),
        gas=Uint(GAS),
        value=U256(0),
        data=calldata,
        code_address=Bytes20(to),
        code=ctx._get_code_at(to),
        depth=Uint(0),
        should_transfer_value=False,
        is_static=False,
        accessed_addresses=warm_addrs,
        accessed_storage_keys=warm_keys,
        parent_evm=None,
    )
    evm2 = process_message(msg)
    assert evm2.error is None, f"reverted: {calldata[:4].hex()}"
    return GAS - int(evm2.gas_left)


def measure_all(ctx, contract: bytes, tag: int) -> dict[str, int]:
    """Run an identical op sequence on `contract`, returning {op: warm gas}.
    `tag` keeps the fresh addresses distinct between the two contracts."""
    dep = ctx.deployer

    def call(data: bytes):
        r = ctx.call(contract, data)
        assert not r.is_error, f"setup reverted: {data[:4].hex()}"
        return r

    counter = [0]

    def fresh() -> bytes:
        counter[0] += 1
        return low((tag << 24) | counter[0])

    g: dict[str, int] = {}

    a = fresh()
    call(MINT + addr(a) + word(ONE))                       # make slot nonzero
    g["mint (warm)"] = warm_gas(ctx, contract, MINT + addr(a) + word(ONE))

    call(MINT + addr(dep) + word(100 * ONE))
    g["burn"] = warm_gas(ctx, contract, BURN + word(ONE))

    c = fresh()
    call(TRANSFER + addr(c) + word(ONE))                   # warm recipient slot
    g["transfer (warm)"] = warm_gas(ctx, contract, TRANSFER + addr(c) + word(ONE))

    call(APPROVE + addr(dep) + word(MAX))                  # deployer approves itself
    d = fresh()
    call(TRANSFER_FROM + addr(dep) + addr(d) + word(ONE))
    g["transferFrom"] = warm_gas(ctx, contract, TRANSFER_FROM + addr(dep) + addr(d) + word(ONE))

    e = fresh()
    call(MINT + addr(e) + word(ONE))
    g["balanceOf"] = warm_gas(ctx, contract, BALANCE_OF + addr(e))

    f = fresh()
    call(APPROVE + addr(f) + word(ONE))
    g["approve (control)"] = warm_gas(ctx, contract, APPROVE + addr(f) + word(2))
    return g


def main() -> int:
    if not ARTIFACT.exists():
        print(f"missing vendored artifact {ARTIFACT}")
        return 1

    creation = patch_venom.creation_from_artifact(ARTIFACT)
    runtime = patch_venom.runtime_from_artifact(ARTIFACT)
    patched = patch_venom.patch(runtime)

    ctx = MiniEVMContext()
    orig = ctx.deploy(creation)
    opt = ctx.deploy(creation)
    ctx.set_code(opt, patched)

    gO = measure_all(ctx, orig, tag=0xA0)
    gP = measure_all(ctx, opt, tag=0xB0)

    assert (int.from_bytes(ctx.call(orig, BALANCE_OF + addr(ctx.deployer)).output, "big")
            == int.from_bytes(ctx.call(opt, BALANCE_OF + addr(ctx.deployer)).output, "big")), "parity"

    print("Venom ERC-20 gas benchmark — warm / hot-path "
          "(execution-specs EVM via MiniEVMContext)")
    print(f"runtime {len(runtime)} B, {patch_venom.count_sites(runtime)} balance sites "
          f"patched, length preserved\n")
    print(f"{'operation':<20}{'keccak':>9}{'NOT-slot':>10}{'Δ gas':>8}{'Δ %':>9}")
    print("-" * 56)
    ops = ["mint (warm)", "burn", "transfer (warm)", "transferFrom",
           "balanceOf", "approve (control)"]
    tot_o = tot_p = 0
    for op in ops:
        o, p = gO[op], gP[op]
        d = p - o
        pct = (d / o * 100) if o else 0.0
        if op != "approve (control)":
            tot_o += o
            tot_p += p
        print(f"{op:<20}{o:>9}{p:>10}{d:>+8}{pct:>+8.2f}%")
    print("-" * 56)
    dt = tot_p - tot_o
    print(f"{'workload total*':<20}{tot_o:>9}{tot_p:>10}{dt:>+8}{dt / tot_o * 100:>+8.2f}%")
    print("\n* sum of all ops except the approve control (no balance access).")
    print("  Warm regime: account + touched slots pre-warmed (EIP-2929); the\n"
          "  SSTORE-reset cost the optimization can't remove stays in the\n"
          "  denominator. Δ is regime-independent and matches Foundry; a pure\n"
          "  read (balanceOf) is dominated by slot derivation, hence the big %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
