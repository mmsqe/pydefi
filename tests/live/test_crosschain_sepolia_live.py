"""Live Sepolia test for the cross-chain destination builders.

The cherry-picked ``test_sepolia_cctp_execution_live.py`` exercises the raw CCTP
``depositForBurnWithHook`` / ``CCTPComposer.receiveAndExecute`` flow with a
hand-built DeFiVM program.  This test instead drives the ``pydefi.crosschain``
builders against **live Aave V3 on Sepolia**: it resolves the real Aave Pool via
:func:`~pydefi.crosschain.compose.build_supply_template`, then compiles the
destination compose program with
:func:`~pydefi.crosschain.compose.build_compose_deposit_program` and checks the
real supply calldata is embedded.

Read-only (no broadcast, no keys) — needs only ``SEPOLIA_RPC_URL`` and skips
otherwise.  The source burn isn't covered here; see
``test_crosschain_cctp_sepolia_live.py`` for the full sign-and-broadcast path
(``build_deposit_route`` with a ``dst_domain`` override for testnet lanes).

Run with::

    pytest -m live tests/live/test_crosschain_sepolia_live.py
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest
from web3 import AsyncWeb3

from pydefi.crosschain.compose import build_compose_deposit_program, build_supply_template
from pydefi.deployments import get_token
from pydefi.types import ZERO_ADDRESS, Address, ChainId, TokenAmount
from pydefi.vm.yield_deposit import AMOUNT_SENTINEL
from pydefi.yields.router import YieldMarket
from tests.live.sepolia_helpers import require_env

_USER = Address("0x" + "AA" * 20)
_COMPOSER = Address("0x" + "C0" * 20)


def _aave_usdc_market(token) -> YieldMarket:
    return YieldMarket(
        protocol="aave_v3",
        chain_id=ChainId.SEPOLIA,
        token=token,
        supply_apy=Decimal("0"),
        utilization=Decimal("0"),
        available_liquidity=TokenAmount(token=token, amount=0),
        market_id=f"aave_v3:{ChainId.SEPOLIA}:USDC",
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_compose_deposit_program_against_live_aave_v3_sepolia():
    require_env("SEPOLIA_RPC_URL")
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(os.environ["SEPOLIA_RPC_URL"].strip()))
    usdc = get_token("USDC", ChainId.SEPOLIA)

    # Resolves the live Aave V3 Sepolia Pool via from_chain, then encodes the
    # supply call with the runtime-amount sentinel.
    supply = await build_supply_template(_aave_usdc_market(usdc), w3, _USER)
    assert supply.spender != ZERO_ADDRESS  # real on-chain Aave Pool address
    assert supply.token == usdc.address
    assert bytes(usdc.address) in supply.template  # the supplied asset
    assert AMOUNT_SENTINEL.to_bytes(32, "big") in supply.template  # patch site

    # Compile the destination program the composer would run on settlement.
    program = build_compose_deposit_program(
        supply_template=supply.template,
        protocol=supply.spender,
        target_token=supply.token,
        composer=_COMPOSER,
    )
    assert isinstance(program, bytes) and len(program) > 0
    assert supply.template in program  # supply call embedded, amount patched at runtime
