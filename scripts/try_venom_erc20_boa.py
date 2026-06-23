#!/usr/bin/env python3
"""Try the evm-smith Venom balance-slot peephole on **titanoboa** (boa).

The titanoboa alternative to scripts/try_venom_erc20.py (which uses pydefi's
MiniEVMContext / execution-specs). boa runs the *pre-compiled* Foundry Venom
bytecode directly — no Vyper compilation — so boa's bundled compiler version
is irrelevant. Deploys the original twice, etches the peephole-patched runtime
onto the second, then exercises mint / transfer / balanceOf to confirm
behaviour parity and the gas delta, plus a NOT(addr) no-aliasing spot check.

Setup (one-time, in a venv):
    pip install titanoboa
Run:
    python scripts/try_venom_erc20_boa.py
"""

from __future__ import annotations

import sys

import boa  # noqa: E402

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

low = addr20  # local name used below

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

    # Deploy both via the creation code (constructor sets ownable / is_minter),
    # then etch the patched runtime onto the second instance.
    orig, _ = boa.env.deploy_code(bytecode=creation)
    opt, _ = boa.env.deploy_code(bytecode=creation)
    boa.env.set_code(opt, patched)

    eoa = bytes(boa.env.eoa.canonical_address)

    def call(to, data):
        return boa.env.raw_call(to, data=data)  # raises on revert

    def bal(to, who: bytes) -> int:
        return int.from_bytes(call(to, BALANCE_OF + arg_addr(who)).output, "big")

    A = low(0xAA)
    B = low(0xBB)

    # mint (warm): pre-mint, then measure
    for c in (orig, opt):
        call(c, MINT + arg_addr(A) + word(ONE))
    gO = call(orig, MINT + arg_addr(A) + word(ONE)).get_gas_used()
    gP = call(opt, MINT + arg_addr(A) + word(ONE)).get_gas_used()
    assert bal(orig, A) == bal(opt, A) == 2 * ONE, "mint parity"
    print(f"mint(warm)     orig {gO:>6}  opt {gP:>6}  Δ {gP - gO:+d}")

    # transfer (warm)
    for c in (orig, opt):
        call(c, MINT + arg_addr(eoa) + word(100 * ONE))
        call(c, MINT + arg_addr(B) + word(ONE))
    gO = call(orig, TRANSFER + arg_addr(B) + word(ONE)).get_gas_used()
    gP = call(opt, TRANSFER + arg_addr(B) + word(ONE)).get_gas_used()
    assert bal(orig, B) == bal(opt, B), "transfer recipient parity"
    assert bal(orig, eoa) == bal(opt, eoa), "transfer sender parity"
    print(f"transfer(warm) orig {gO:>6}  opt {gP:>6}  Δ {gP - gO:+d}")

    # balanceOf (warm)
    gO = call(orig, BALANCE_OF + arg_addr(A)).get_gas_used()
    gP = call(opt, BALANCE_OF + arg_addr(A)).get_gas_used()
    print(f"balanceOf(warm)orig {gO:>6}  opt {gP:>6}  Δ {gP - gO:+d}")

    # NOT(addr) no-aliasing on the patched contract: mint to the colliding
    # low addresses (owner@1, balanceOf-base@2, totalSupply@4) and verify
    # named state survives.
    owner_before = call(opt, OWNER).output
    ts_before = int.from_bytes(call(opt, TOTAL_SUPPLY).output, "big")
    for slot in (0x01, 0x02, 0x04):
        call(opt, MINT + arg_addr(low(slot)) + word(ONE))
    assert call(opt, OWNER).output == owner_before, "owner corrupted!"
    assert int.from_bytes(call(opt, TOTAL_SUPPLY).output, "big") == ts_before + 3 * ONE
    for slot in (0x01, 0x02, 0x04):
        assert bal(opt, low(slot)) == ONE
    print("\nno-aliasing: minted to colliders 0x01/0x02/0x04 — owner & "
          "totalSupply intact ✓")
    print("all behaviour parity assertions passed ✓  (titanoboa / boa)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
