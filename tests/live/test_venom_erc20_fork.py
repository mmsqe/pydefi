"""Anvil-fork-of-Sepolia proof of the evm-smith Venom balance-slot peephole.

Forks Sepolia with Anvil, then deploys the vendored Venom-compiled Snekmate
ERC-20 (:mod:`pydefi.venom`) twice — once unpatched, once through
``patch_creation`` init code (NOT-slot runtime) — and exercises
mint / transfer / balanceOf to assert:

* the patched deploy returns a runtime with **zero** balance keccak sites
  (the original has :func:`expected_runtime_sites`),
* behaviour parity (identical balances after identical ops),
* a positive ``gasUsed`` delta (the NOT-slot variant is cheaper), and
* NOT(addr) no-aliasing — minting to the Vyper named-slot colliders
  (owner@1, balanceOf-base@2, totalSupply@4) corrupts neither.

This is the local-fork twin of :mod:`tests.live.test_venom_erc20_sepolia_execution`
(which broadcasts the *same* ``patch_creation`` deploy path for real). It runs
on Anvil's prefunded unlocked dev account, so it needs no key and spends no ETH.

Gated to the ``fork`` marker (skipped unless ``-m fork``); requires ``anvil`` on
PATH and an RPC to fork (``SEPOLIA_RPC_URL``; defaults to a public endpoint).
Run::

    pytest tests/live/test_venom_erc20_fork.py -m fork -s
"""

from __future__ import annotations

import pytest

from tests.live.conftest import SEPOLIA_RPC_URL, _anvil_node
from tests.live.venom_erc20_common import (
    ALIAS_PROBES,
    ONE,
    VenomErc20,
    creation_pair,
    expected_runtime_sites,
    low_addr,
)


@pytest.fixture
async def sepolia_fork_w3():
    """Function-scoped Anvil fork of Sepolia (forks ``SEPOLIA_RPC_URL``)."""
    async with _anvil_node(["--fork-url", SEPOLIA_RPC_URL]) as w3:
        yield w3


@pytest.mark.fork
class TestVenomErc20Fork:
    async def test_balance_peephole_parity_and_gas(self, sepolia_fork_w3):
        w3 = sepolia_fork_w3
        deployer = (await w3.eth.accounts)[0]  # prefunded, unlocked

        orig_creation, patched_creation = creation_pair()
        orig = await VenomErc20.deploy(w3, deployer, None, orig_creation)
        opt = await VenomErc20.deploy(w3, deployer, None, patched_creation)

        # The patched creation deploys to a runtime with no balance keccak left.
        assert await orig.balance_sites() == expected_runtime_sites()
        assert await opt.balance_sites() == 0

        A = low_addr(0xAA)
        B = low_addr(0xBB)

        # mint (warm): pre-mint so the slot is already nonzero, then measure.
        await orig.mint(A, ONE)
        await opt.mint(A, ONE)
        g_mint_o = await orig.mint(A, ONE)
        g_mint_p = await opt.mint(A, ONE)
        assert await orig.balance_of(A) == await opt.balance_of(A) == 2 * ONE, "mint parity"
        assert g_mint_p < g_mint_o, f"patched mint not cheaper: {g_mint_p} !< {g_mint_o}"

        # transfer (warm): fund the deployer, warm the recipient, then measure.
        await orig.mint(deployer, 100 * ONE)
        await opt.mint(deployer, 100 * ONE)
        await orig.transfer(B, ONE)
        await opt.transfer(B, ONE)
        g_xfer_o = await orig.transfer(B, ONE)
        g_xfer_p = await opt.transfer(B, ONE)
        assert await orig.balance_of(B) == await opt.balance_of(B), "transfer recipient parity"
        assert await orig.balance_of(deployer) == await opt.balance_of(deployer), "transfer sender parity"
        assert g_xfer_p < g_xfer_o, f"patched transfer not cheaper: {g_xfer_p} !< {g_xfer_o}"

        # NOT(addr) no-aliasing: minting to the named-slot colliders must leave
        # owner / totalSupply intact on the patched contract.
        owner_before = await opt.owner()
        ts_before = await opt.total_supply()
        for slot in ALIAS_PROBES:
            await opt.mint(low_addr(slot), ONE)
        assert await opt.owner() == owner_before, "owner corrupted by NOT-slot aliasing"
        assert await opt.total_supply() == ts_before + len(ALIAS_PROBES) * ONE, "totalSupply corrupted"
        for slot in ALIAS_PROBES:
            assert await opt.balance_of(low_addr(slot)) == ONE, f"balance[{slot}] wrong"

        print(
            "\nVenom balance-slot peephole — Anvil fork of Sepolia"
            f"\n  deploy gas  orig {orig.deploy_gas:>8}  opt {opt.deploy_gas:>8}"
            f"\n  mint(warm)  orig {g_mint_o:>8}  opt {g_mint_p:>8}  Δ {g_mint_p - g_mint_o:+d}"
            f"\n  transfer    orig {g_xfer_o:>8}  opt {g_xfer_p:>8}  Δ {g_xfer_p - g_xfer_o:+d}"
            "\n  parity ✓   no-aliasing ✓"
        )
