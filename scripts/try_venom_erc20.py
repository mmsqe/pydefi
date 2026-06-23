#!/usr/bin/env python3
"""Try the evm-smith Venom balance-slot peephole on pydefi's MiniEVMContext
(the execution-specs EVM — the same semantics EVMYulLean formalizes).

Deploys the Venom-compiled Snekmate ERC-20 twice, etches the peephole-patched
runtime onto the second instance, then exercises mint / transfer / balanceOf
to confirm behaviour parity and measure the gas delta — plus a NOT(addr)
no-aliasing spot check against Vyper's named slots.

Run:  /Users/mavis/Documents/crypto/pydefi/.venv/bin/python \
          scripts/try_venom_erc20.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PYDEFI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYDEFI_ROOT))

from pydefi.venom import balance_patch as patch_venom  # noqa: E402
from pydefi.venom.erc20_abi import (  # noqa: E402
    BALANCE_OF,
    MINT,
    OWNER,
    TOTAL_SUPPLY,
    TRANSFER,
    addr20,
    arg_addr,
    word,
)
from tests.conftest import MiniEVMContext  # noqa: E402

addr, low_addr = arg_addr, addr20  # local names used below

ARTIFACT = patch_venom.DEFAULT_ARTIFACT

ONE = 10**18


def main() -> int:
    if not ARTIFACT.exists():
        print(f"missing vendored artifact {ARTIFACT}")
        return 1

    creation = patch_venom.creation_from_artifact(ARTIFACT)
    runtime = patch_venom.runtime_from_artifact(ARTIFACT)
    patched = patch_venom.patch(runtime)
    print(f"venom runtime {len(runtime)} B, balance sites "
          f"{patch_venom.count_sites(runtime)} -> {patch_venom.count_sites(patched)} "
          f"after patch (length preserved: {len(runtime) == len(patched)})\n")

    ctx = MiniEVMContext()

    # Deploy both via the creation code so the constructor sets immutables /
    # ownable state / is_minter[deployer]; then etch the patched runtime onto
    # the second instance (mirrors the Foundry vm.etch flow).
    orig = ctx.deploy(creation)
    opt = ctx.deploy(creation)
    ctx.set_code(opt, patched)

    def call(to: bytes, data: bytes):
        r = ctx.call(to, data)
        assert not r.is_error, f"call reverted: {data[:4].hex()}"
        return r

    def bal(to: bytes, who: bytes) -> int:
        return int.from_bytes(call(to, BALANCE_OF + addr(who)).output, "big")

    A = low_addr(0xAA)
    B = low_addr(0xBB)

    # ---- mint (warm): pre-mint to warm the slot, then measure ----
    for c in (orig, opt):
        call(c, MINT + addr(A) + word(ONE))
    gO = call(orig, MINT + addr(A) + word(ONE)).gas_used
    gP = call(opt, MINT + addr(A) + word(ONE)).gas_used
    assert bal(orig, A) == bal(opt, A) == 2 * ONE, "mint parity"
    print(f"mint(warm)     orig {gO:>6}  opt {gP:>6}  Δ {gP - gO:+d}")

    # ---- transfer (warm): fund deployer, warm B, then measure ----
    for c in (orig, opt):
        call(c, MINT + addr(ctx.deployer) + word(100 * ONE))
        call(c, MINT + addr(B) + word(ONE))
    gO = call(orig, TRANSFER + addr(B) + word(ONE)).gas_used
    gP = call(opt, TRANSFER + addr(B) + word(ONE)).gas_used
    assert bal(orig, B) == bal(opt, B), "transfer recipient parity"
    assert bal(orig, ctx.deployer) == bal(opt, ctx.deployer), "transfer sender parity"
    print(f"transfer(warm) orig {gO:>6}  opt {gP:>6}  Δ {gP - gO:+d}")

    # ---- balanceOf (warm) ----
    gO = call(orig, BALANCE_OF + addr(A)).gas_used
    gP = call(opt, BALANCE_OF + addr(A)).gas_used
    print(f"balanceOf(warm)orig {gO:>6}  opt {gP:>6}  Δ {gP - gO:+d}")

    # ---- NOT(addr) no-aliasing safety on the patched contract ----
    # Mint to the low addresses that collide with Vyper's named slots
    # (owner@1, balanceOf-base@2, totalSupply@4). A *raw* address-as-slot
    # scheme would corrupt those; NOT(addr) lands every balance above 2^160.
    owner_before = call(opt, OWNER).output
    ts_before = int.from_bytes(call(opt, TOTAL_SUPPLY).output, "big")
    for slot in (0x01, 0x02, 0x04):
        call(opt, MINT + addr(low_addr(slot)) + word(ONE))
    owner_after = call(opt, OWNER).output
    ts_after = int.from_bytes(call(opt, TOTAL_SUPPLY).output, "big")
    assert owner_after == owner_before, "owner corrupted!"
    assert ts_after == ts_before + 3 * ONE, "totalSupply corrupted!"
    for slot in (0x01, 0x02, 0x04):
        assert bal(opt, low_addr(slot)) == ONE, f"balance[{slot}] wrong"
    print("\nno-aliasing: minted to colliders 0x01/0x02/0x04 — owner & "
          "totalSupply intact, balances correct ✓")
    print("all behaviour parity assertions passed ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
