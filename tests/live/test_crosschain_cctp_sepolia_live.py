"""Sign-and-broadcast testnet test for the Cross-Chain Router Builder.

Parallels ``test_sepolia_cctp_execution_live.py`` but drives the
``pydefi.crosschain`` route builder end to end instead of a hand-built program:

1. Acquire USDC on Sepolia with a real swap (reusing that module's helper).
2. ``build_deposit_route`` compiles the destination ``supply -> Aave V3``
   program and the source ``approve`` + ``depositForBurnWithHook`` legs.
3. Sign and broadcast the source legs on Sepolia.
4. Relay ``CCTPComposer.receiveAndExecute`` on Arbitrum Sepolia (poll Circle's
   sandbox attestation), then assert the sender's aUSDC balance grew — i.e. the
   bridged USDC was supplied into Aave on the destination, crediting the user.

``testnet``-marked and gated on the same env vars as the sibling test:
``SEPOLIA_RPC_URL``, ``SEPOLIA_PRIVATE_KEY``, ``ARBITRUM_SEPOLIA_RPC_URL``,
``ARBITRUM_SEPOLIA_CCTP_COMPOSER`` (deployed with a valid interpreter).

Run with::

    pytest -m testnet tests/live/test_crosschain_cctp_sepolia_live.py
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest
from eth_account import Account
from eth_contract import Contract
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3

from pydefi.bridge.cctp import CCTP
from pydefi.crosschain.route import build_deposit_route
from pydefi.lending.aave_v3 import AaveV3
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.utils import ERC20_ABI, send_signed
from pydefi.yields.router import YieldMarket
from tests.live.sepolia_helpers import USDC_SEP, USDC_SEPOLIA, connect, require_env
from tests.live.test_sepolia_cctp_execution_live import (
    ARBITRUM_SEPOLIA_CCTP_COMPOSER,
    ARBITRUM_SEPOLIA_RPC_URL,
    ARBITRUM_SEPOLIA_USDC_ADDRESS,
    CCTP_DOMAIN_ARBITRUM,
    CCTP_IRIS_SANDBOX_API,
    TOKEN_MESSENGER_SEPOLIA,
    _derive_cctp_message_from_burn_tx,
    _relay_cctp_compose_receive_and_execute,
    _swap_and_quote_usdc_from_eth,
)

_ARBITRUM_SEPOLIA = 421614
_MAX_FEE = 20_000  # USDC sub-units; amount must exceed this to cover the CCTP fee
_FAST_FINALITY = 500
_SWAP_ETH_IN = 2 * 10**15


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_crosschain_deposit_route_broadcast() -> None:
    require_env(
        "SEPOLIA_RPC_URL",
        "SEPOLIA_PRIVATE_KEY",
        "ARBITRUM_SEPOLIA_RPC_URL",
        "ARBITRUM_SEPOLIA_CCTP_COMPOSER",
    )
    private_key = os.environ["SEPOLIA_PRIVATE_KEY"].strip()
    w3 = await connect(os.environ["SEPOLIA_RPC_URL"].strip())
    sender = Web3.to_checksum_address(Account.from_key(private_key).address)
    dst_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(ARBITRUM_SEPOLIA_RPC_URL))

    # The destination deposit target: Aave V3 USDC on Arbitrum Sepolia.  Resolve
    # the aToken up front so we can skip cleanly if the reserve isn't listed.
    usdc_arb = Token(
        chain_id=_ARBITRUM_SEPOLIA,
        address=Address(HexBytes(ARBITRUM_SEPOLIA_USDC_ADDRESS)),
        symbol="USDC",
        decimals=6,
    )
    try:
        aave = await AaveV3.from_chain(dst_w3, _ARBITRUM_SEPOLIA)
        reserve = await aave.get_reserve_data(usdc_arb)
    except Exception as exc:  # noqa: BLE001 — testnet reserve may be absent
        pytest.skip(f"no Aave V3 USDC reserve on Arbitrum Sepolia: {exc}")
    a_usdc = Contract.from_abi(ERC20_ABI, to=Web3.to_checksum_address(reserve.a_token_address))
    pre_a_balance = int(await a_usdc.fns.balanceOf(sender).call(dst_w3))

    # 1. Acquire USDC on Sepolia via a real swap.
    usdc = Contract.from_abi(ERC20_ABI, to=USDC_SEPOLIA)
    block = await w3.eth.get_block("latest")
    deadline = int(block["timestamp"]) + 20 * 60
    if int(await w3.eth.get_balance(sender)) <= _SWAP_ETH_IN:
        pytest.fail("insufficient Sepolia ETH for swap+gas; top up the faucet")
    before = int(await usdc.fns.balanceOf(sender).call(w3))
    await _swap_and_quote_usdc_from_eth(
        w3=w3, private_key=private_key, sender=sender, swap_eth_in=_SWAP_ETH_IN, deadline=deadline, variant="v2"
    )
    received = int(await usdc.fns.balanceOf(sender).call(w3)) - before
    if received <= _MAX_FEE:
        pytest.skip(f"swap produced {received} USDC (<= max_fee {_MAX_FEE}); top up or raise swap_eth_in")

    # 2. Build the cross-chain deposit route with the crosschain builder.
    composer = Address(HexBytes(Web3.to_checksum_address(ARBITRUM_SEPOLIA_CCTP_COMPOSER)))
    cctp_lane = CCTP(
        w3=w3,
        src_chain_id=ChainId.SEPOLIA,
        dst_chain_id=_ARBITRUM_SEPOLIA,
        token_messenger_address=Address(HexBytes(TOKEN_MESSENGER_SEPOLIA)),
        src_usdc_address=Address(HexBytes(USDC_SEPOLIA)),
        api_base_url=CCTP_IRIS_SANDBOX_API,
    )
    market = YieldMarket(
        protocol="aave_v3",
        chain_id=_ARBITRUM_SEPOLIA,
        token=usdc_arb,
        supply_apy=Decimal("0"),
        utilization=Decimal("0"),
        available_liquidity=TokenAmount(token=usdc_arb, amount=0),
        market_id=f"aave_v3:{_ARBITRUM_SEPOLIA}:USDC",
    )
    route = await build_deposit_route(
        bridge=cctp_lane,
        w3_dst=dst_w3,
        amount_in=TokenAmount(USDC_SEP, received),
        composer=composer,
        user=Address(HexBytes(sender)),
        dest_market=market,
        max_fee=_MAX_FEE,
        min_finality_threshold=_FAST_FINALITY,
        dst_domain=CCTP_DOMAIN_ARBITRUM,
    )
    assert [leg.kind for leg in route.steps] == ["approve", "bridge_compose"]

    # 3. Sign + broadcast the source legs (approve, then burn) on Sepolia.
    nonce = int(await w3.eth.get_transaction_count(sender))
    burn_tx_hash = ""
    for leg in route.build_transactions():
        tx_hash, status = await send_signed(
            w3,
            private_key=private_key,
            sender=sender,
            to=Web3.to_checksum_address(leg["to"]),
            data=bytes.fromhex(leg["data"].removeprefix("0x")),
            nonce=nonce,
        )
        assert status == 1, f"{leg['to']} leg failed: {tx_hash}"
        nonce += 1
        burn_tx_hash = tx_hash  # the final leg is the depositForBurnWithHook

    # 4. Relay receiveAndExecute on Arbitrum Sepolia; the composer runs the
    #    compiled supply program, crediting the sender in Aave.
    _, relay_status = await _relay_cctp_compose_receive_and_execute(
        destination_rpc_url=ARBITRUM_SEPOLIA_RPC_URL,
        private_key=private_key,
        sender=sender,
        burn_tx_hash=burn_tx_hash,
        composer_address=ARBITRUM_SEPOLIA_CCTP_COMPOSER,
        message_hash=await _derive_cctp_message_from_burn_tx(w3, burn_tx_hash),
    )
    assert relay_status == 1, "composer receiveAndExecute reverted"

    post_a_balance = int(await a_usdc.fns.balanceOf(sender).call(dst_w3))
    assert post_a_balance > pre_a_balance, (
        f"aUSDC did not grow: {pre_a_balance} -> {post_a_balance}; the destination supply did not credit the user"
    )
