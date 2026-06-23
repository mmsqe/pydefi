"""Broadcast the evm-smith Venom balance-slot peephole on the Sepolia testnet.

The real-chain twin of :mod:`tests.live.test_venom_erc20_fork`: instead of an
Anvil fork it signs and broadcasts the *same* ``patch_creation`` deploy path to
Sepolia, the end-to-end execution proof a fork/etch cannot give — real signing,
mempool, gas, and on-chain runtime. ``etch``/``set_code`` is impossible on a
public chain, so the optimized runtime is reached only by deploying init code
whose embedded runtime is peephole-patched (length-preserving), with the
constructor still running so the signer is the owner/minter.

Kept deliberately lean (4 broadcasts: 2 deploys + 1 mint each) — the
keccak→NOT delta is regime-independent, so a single mint pair already shows the
saving. It asserts:

* the patched deploy's on-chain runtime has **zero** balance keccak sites,
* mint parity (identical balances), and
* a positive ``gasUsed`` delta (NOT-slot mint is cheaper on-chain).

Self-funding is unnecessary — the signer mints its own token — but it needs
Sepolia ETH for gas. Gated to the ``testnet`` marker (skipped unless
``-m testnet``), so a bare ``pytest`` never broadcasts. Run::

    SEPOLIA_RPC_URL=... SEPOLIA_PRIVATE_KEY=<funded with Sepolia ETH> \\
        pytest tests/live/test_venom_erc20_sepolia_execution.py -m testnet -s
"""

from __future__ import annotations

import os

import pytest
from eth_account import Account
from web3 import AsyncWeb3, Web3

from tests.live.sepolia_helpers import require_env
from tests.live.venom_erc20_common import (
    ONE,
    VenomErc20,
    creation_pair,
    expected_runtime_sites,
    low_addr,
)

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL", "").strip()
SEPOLIA_PRIVATE_KEY = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
SEPOLIA_CHAIN_ID = 11155111
#: ~0.02 ETH: two ~6.9 KB-initcode deploys (~1.5M gas each) + two mints.
MIN_GAS_ETH = 2 * 10**16


async def _connect() -> AsyncWeb3:
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY")
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(SEPOLIA_RPC_URL))
    if await w3.eth.chain_id != SEPOLIA_CHAIN_ID:
        pytest.skip("SEPOLIA_RPC_URL is not pointed at Sepolia")
    return w3


@pytest.mark.testnet
class TestVenomErc20SepoliaExecution:
    """Real Sepolia broadcast of the original vs NOT-slot ERC-20."""

    async def test_balance_peephole_broadcast(self):
        w3 = await _connect()
        account = Account.from_key(SEPOLIA_PRIVATE_KEY)
        sender = Web3.to_checksum_address(account.address)
        if await w3.eth.get_balance(sender) < MIN_GAS_ETH:
            pytest.skip("signer underfunded with Sepolia ETH — top up via a faucet")

        orig_creation, patched_creation = creation_pair()
        orig = await VenomErc20.deploy(w3, sender, SEPOLIA_PRIVATE_KEY, orig_creation)
        opt = await VenomErc20.deploy(w3, sender, SEPOLIA_PRIVATE_KEY, patched_creation)

        # The patched init code deployed a NOT-slot runtime on the real chain.
        assert await orig.balance_sites() == expected_runtime_sites()
        assert await opt.balance_sites() == 0

        # One mint each to the same fresh holder; compare the on-chain receipts.
        holder = low_addr(0xA11CE)
        g_o = await orig.mint(holder, ONE)
        g_p = await opt.mint(holder, ONE)
        assert await orig.balance_of(holder) == await opt.balance_of(holder) == ONE, "mint parity"
        assert g_p < g_o, f"patched mint not cheaper on-chain: {g_p} !< {g_o}"

        print(
            "\nVenom balance-slot peephole — Sepolia broadcast"
            f"\n  orig {orig.address}   opt {opt.address}"
            f"\n  deploy gas  orig {orig.deploy_gas:>8}  opt {opt.deploy_gas:>8}"
            f"\n  mint gasUsed orig {g_o:>8}  opt {g_p:>8}  Δ {g_p - g_o:+d}"
        )
