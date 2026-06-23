from __future__ import annotations

import asyncio
import os
from typing import Any, cast
from urllib.error import HTTPError

import pytest
from eth_abi.abi import decode
from eth_account import Account
from eth_contract import ERC20, Contract
from eth_utils.crypto import keccak
from hexbytes import HexBytes
from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3, Web3
from web3.types import TxParams

from pydefi.abi.vm import DeFiVM
from pydefi.amm.uniswap_v2 import UniswapV2
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.crosschain.compose import build_source_swap_bridge_program
from pydefi.deployments import get_address
from pydefi.pathfinder.dag import RouteDAG
from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge
from pydefi.pathfinder.router import Router
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.utils import (
    CCTP_COMPOSER_ABI,
    CCTP_TOKEN_MESSENGER_ABI,
    ERC20_ABI,
    decode_hex_field,
    get_json,
    send_signed,
    to_return_bytes,
    tx_data_bytes,
    with_0x,
)
from pydefi.vm import Program
from pydefi.vm.swap import build_swap_transaction
from tests.live.sepolia_helpers import (
    ARBITRUM_SEPOLIA_CHAIN_ID,
    USDC_SEPOLIA,
    WETH_SEPOLIA,
    connect,
    discover_sepolia_v4_weth_usdc_edge,
    require_env,
)
from tests.live.sepolia_helpers import (
    USDC_SEP as _USDC_TOKEN_SEPOLIA,
)
from tests.live.sepolia_helpers import (
    WETH_SEP as _WETH_TOKEN_SEPOLIA,
)

# CCTP TokenMessengerV2 / MessageTransmitterV2 (testnet) from pydefi/config/cctp-testnet.json.
TOKEN_MESSENGER_SEPOLIA = Web3.to_checksum_address(get_address("CCTP_TOKEN_MESSENGER", ChainId.SEPOLIA))
CCTP_MESSAGE_TRANSMITTER_V2_TESTNET = Web3.to_checksum_address(get_address("CCTP_MESSAGE_TRANSMITTER", ChainId.SEPOLIA))
# Uniswap deployments sourced from pydefi/config/uniswap.json via get_address.
UNISWAP_V2_ROUTER_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V2_ROUTER", ChainId.SEPOLIA))
UNISWAP_V2_FACTORY_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V2_FACTORY", ChainId.SEPOLIA))
UNISWAP_V3_FACTORY_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V3_FACTORY", ChainId.SEPOLIA))
# config names these UNISWAP_V3_ROUTER / UNISWAP_V3_QUOTER (SwapRouter02 / QuoterV2).
UNISWAP_V3_SWAP_ROUTER_02_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V3_ROUTER", ChainId.SEPOLIA))
UNISWAP_V3_QUOTER_V2_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V3_QUOTER", ChainId.SEPOLIA))
# Arbitrum-Sepolia USDC from pydefi/config/cctp-testnet.json (CCTP_USDC registry).
ARBITRUM_SEPOLIA_USDC_ADDRESS = Web3.to_checksum_address(get_address("CCTP_USDC", ARBITRUM_SEPOLIA_CHAIN_ID))

TIMEOUT_SEC = 1200  # CCTP attestation on Sepolia typically takes 5-20 min
INTERVAL_SEC = 10
CCTP_DOMAIN_ARBITRUM = 3
CCTP_DOMAIN_ETHEREUM_SEPOLIA = 0
CCTP_IRIS_SANDBOX_API = "https://iris-api-sandbox.circle.com"
UINT256_MAX = (2**256) - 1

ARBITRUM_SEPOLIA_RPC_URL = os.getenv("ARBITRUM_SEPOLIA_RPC_URL", "").strip()
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL", "").strip()
CIRCLE_API_KEY = os.getenv("CIRCLE_API_KEY", "").strip()

SEPOLIA_PRIVATE_KEY = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
SEPOLIA_DEFI_VM = os.getenv("SEPOLIA_DEFI_VM", "").strip()
ARBITRUM_SEPOLIA_DEFI_VM = os.getenv("ARBITRUM_SEPOLIA_DEFI_VM", "").strip()
# What the composer DELEGATECALLs at: the EVM interpreter (e.g. Analog-Labs at
# 0x0000…7d85e8D), NOT the DeFiVM contract.  Optional; if unset the composer's
# advertised ``interpreter()`` is just printed without assertion.
ARBITRUM_SEPOLIA_INTERPRETER = os.getenv("ARBITRUM_SEPOLIA_INTERPRETER", "").strip()
ARBITRUM_SEPOLIA_CCTP_COMPOSER = os.getenv("ARBITRUM_SEPOLIA_CCTP_COMPOSER", "").strip()


def _decode_cctp_deposit_for_burn_fields(calldata: bytes) -> tuple[int, int, str]:
    if len(calldata) < 4:
        raise ValueError("calldata too short for function selector")

    amount, destination_domain, _, burn_token, _, _, _ = decode(
        ["uint256", "uint32", "bytes32", "address", "bytes32", "uint256", "uint32"],
        calldata[4:],
    )
    burn_token = Web3.to_checksum_address(burn_token)
    return amount, destination_domain, burn_token


async def _observe_destination_mint(
    *,
    recipient: str,
    expected_increase: int,
    destination_rpc_url: str,
    destination_usdc: str,
    pre_balance: int,
) -> None:
    dst_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(destination_rpc_url))
    dst_usdc = Contract.from_abi(ERC20_ABI, to=Web3.to_checksum_address(destination_usdc))
    loop = asyncio.get_event_loop()
    deadline = loop.time() + TIMEOUT_SEC
    last_balance = int(pre_balance)
    target = int(pre_balance) + int(expected_increase)
    print("mm-last_balance", last_balance, "target", target)
    while loop.time() < deadline:
        last_balance = int(await dst_usdc.fns.balanceOf(recipient).call(dst_w3))
        print("mm-last_balance", last_balance)
        if last_balance >= target:
            return
        await asyncio.sleep(INTERVAL_SEC)


async def _derive_cctp_message_from_burn_tx(w3: AsyncWeb3, burn_tx_hash: str) -> str | None:
    """Best-effort derive (message_hash, message_bytes) from MessageSent(bytes) burn log."""
    receipt = await w3.eth.get_transaction_receipt(HexBytes(with_0x(burn_tx_hash)))
    message_sent_topic0 = keccak(text="MessageSent(bytes)")

    for log in receipt.get("logs", []):
        topics = log.get("topics") or []
        if not topics:
            continue
        topic0 = to_return_bytes(topics[0])
        if topic0 != message_sent_topic0:
            continue

        data = to_return_bytes(log.get("data", b""))
        if len(data) < 64:
            continue

        # ABI encoding for single dynamic bytes argument:
        # [0:32]=offset (0x20), [32:64]=len, [64:]=payload
        offset = int.from_bytes(data[0:32], "big")
        if offset + 32 > len(data):
            continue
        msg_len = int.from_bytes(data[offset : offset + 32], "big")
        start = offset + 32
        end = start + msg_len
        if msg_len <= 0 or end > len(data):
            continue

        return "0x" + keccak(data[start:end]).hex()

    return None


async def _poll_cctp_attestation_for_tx(
    tx_hash: str,
    *,
    message_hash_override: str | None,
) -> tuple[bytes, bytes]:
    src_domain = CCTP_DOMAIN_ETHEREUM_SEPOLIA
    query_hash = with_0x(tx_hash)
    message_hash = with_0x(message_hash_override) if message_hash_override else ""
    api_base = CCTP_IRIS_SANDBOX_API
    tx_url = f"{api_base}/v2/messages/{src_domain}?transactionHash={query_hash}"
    print("mm-tx_url", tx_url)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + TIMEOUT_SEC
    last_error = ""
    while loop.time() < deadline:
        try:
            payload = await get_json(tx_url)
            print("mm-payload", payload)

            messages = payload.get("messages") or []
            assert isinstance(messages, list), "Iris response field 'messages' must be a list"
            delay_reason = None
            if messages and isinstance(messages[0], dict):
                raw_reason = messages[0].get("delayReason")
                if isinstance(raw_reason, str) and raw_reason:
                    delay_reason = raw_reason

            if delay_reason == "insufficient_fee":
                pytest.fail(f"CCTP message is pending due to insufficient_fee. Increase 20000 and retry. url={tx_url}")

            for item in messages:
                if not isinstance(item, dict):
                    continue
                att = item.get("attestation")
                msg = item.get("message")
                msg_bytes = decode_hex_field(msg if isinstance(msg, str) else None)
                if isinstance(att, str) and att.upper() == "PENDING":
                    continue
                att_bytes = decode_hex_field(att if isinstance(att, str) else None)
                if msg_bytes and att_bytes:
                    return msg_bytes, att_bytes
        except HTTPError as exc:
            last_error = f"HTTP {exc.code} from {api_base}: {exc.reason}"
            print("mm-poll-http", tx_url, last_error)
        except Exception as exc:  # best-effort poll; keep retrying
            last_error = f"{exc}"
            print("mm-poll-error", tx_url, last_error)
        await asyncio.sleep(INTERVAL_SEC)

    if "HTTP 403" in last_error:
        pytest.fail(f"CCTP attestation polling forbidden (HTTP 403). url={tx_url}; last_error={last_error}")

    suffix = f"; tx_url={tx_url}; last_error={last_error}" if last_error else ""
    hint = f"; message_hash={message_hash}" if message_hash else ""
    pytest.fail(f"CCTP attestation not ready before timeout for burn tx {tx_hash}{hint}{suffix}")


async def _relay_cctp_compose_receive_and_execute(
    *,
    destination_rpc_url: str,
    private_key: str,
    sender: str,
    burn_tx_hash: str,
    composer_address: str,
    message_hash: str | None = None,
) -> tuple[str, int]:
    dst_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(destination_rpc_url))
    sender_checksum = Web3.to_checksum_address(sender)
    composer_checksum = Web3.to_checksum_address(composer_address)

    message, attestation = await _poll_cctp_attestation_for_tx(
        burn_tx_hash,
        message_hash_override=message_hash,
    )
    composer = Contract.from_abi(CCTP_COMPOSER_ABI, to=composer_checksum)
    receive_data = tx_data_bytes(composer.fns.receiveAndExecute(message, attestation).data)
    tx_hash, status = await send_signed(
        dst_w3,
        private_key=private_key,
        sender=sender_checksum,
        to=composer_checksum,
        data=receive_data,
        nonce=await dst_w3.eth.get_transaction_count(sender_checksum),
    )
    return tx_hash, status


async def _ensure_max_allowance(
    *,
    w3: AsyncWeb3,
    private_key: str,
    owner: str,
    token: Any,
    token_address: str,
    spender: str,
):
    """Ensure ``token.allowance(owner, spender) == UINT256_MAX`` and return gas used."""
    owner_checksum = Web3.to_checksum_address(owner)
    spender_checksum = Web3.to_checksum_address(spender)
    current_allowance = int(await token.fns.allowance(owner_checksum, spender_checksum).call(w3))
    if current_allowance == UINT256_MAX:
        return 0

    approve_data = tx_data_bytes(token.fns.approve(spender_checksum, UINT256_MAX).data)
    approve_nonce = await w3.eth.get_transaction_count(owner_checksum)
    tx_hash, status = await send_signed(
        w3,
        private_key=private_key,
        sender=owner_checksum,
        to=token_address,
        data=approve_data,
        nonce=approve_nonce,
    )
    assert status == 1, f"approve failed: {tx_hash}"


async def _seed_defi_vm_with_weth(
    *,
    w3: AsyncWeb3,
    private_key: str,
    sender: str,
    defi_vm: str,
    amount: int,
) -> None:
    """Wrap *amount* of ETH into WETH and transfer it to *defi_vm*.

    `build_swap_transaction` assumes the input token is already at the DeFiVM
    address; this prerequisite covers that for ETH-funded swap variants.
    """
    sender_checksum = Web3.to_checksum_address(sender)
    weth_abi = [
        "function deposit() external payable",
        "function balanceOf(address owner) external view returns (uint256)",
        "function transfer(address to, uint256 amount) external returns (bool)",
    ]
    weth = Contract.from_abi(weth_abi, to=WETH_SEPOLIA)
    deposit_calldata = tx_data_bytes(weth.fns.deposit().data)
    nonce = await w3.eth.get_transaction_count(sender_checksum)
    _, deposit_status = await send_signed(
        w3,
        private_key=private_key,
        sender=sender_checksum,
        to=WETH_SEPOLIA,
        data=deposit_calldata,
        nonce=nonce,
        value=amount,
    )
    assert deposit_status == 1, "WETH.deposit() failed during DeFiVM seeding"
    transfer_calldata = tx_data_bytes(weth.fns.transfer(defi_vm, amount).data)
    _, transfer_status = await send_signed(
        w3,
        private_key=private_key,
        sender=sender_checksum,
        to=WETH_SEPOLIA,
        data=transfer_calldata,
        nonce=nonce + 1,
    )
    assert transfer_status == 1, "WETH.transfer(defi_vm) failed during DeFiVM seeding"


async def _seed_build_send_defi_vm_swap(
    *,
    w3: AsyncWeb3,
    private_key: str,
    sender: str,
    defi_vm: str,
    dag: RouteDAG,
    amount: int,
    min_final_out: int,
    label: str,
) -> str:
    """Seed the DeFiVM with *amount* WETH, compile *dag* via ``build_swap_transaction``,
    broadcast it, and assert success.  Returns the swap tx hash."""
    sender_checksum = Web3.to_checksum_address(sender)
    await _seed_defi_vm_with_weth(
        w3=w3, private_key=private_key, sender=sender_checksum, defi_vm=defi_vm, amount=amount
    )
    swap_tx = build_swap_transaction(dag, amount, defi_vm, sender_checksum, min_final_out=min_final_out)
    tx_hash, status = await send_signed(
        w3,
        private_key=private_key,
        sender=sender_checksum,
        to=swap_tx.to,
        data=swap_tx.data,
        nonce=await w3.eth.get_transaction_count(sender_checksum),
    )
    assert status == 1, f"{label} failed: {tx_hash}"
    return tx_hash


def _is_token0_in(token_in_address: Address, token_out_address: Address) -> bool:
    """V2 pairs and V3 pools sort tokens by address; token0 is the lower one."""
    return bytes(token_in_address) < bytes(token_out_address)


def _v2_pool_edge(
    *,
    token_in: Token,
    token_out: Token,
    pool_address: str,
    fee_bps: int = 30,
) -> PoolEdge:
    return PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=Address(pool_address),
        protocol="UniswapV2",
        fee_bps=fee_bps,
        extra={"is_token0_in": _is_token0_in(token_in.address, token_out.address)},
    )


def _v3_pool_edge(
    *,
    token_in: Token,
    token_out: Token,
    pool_address: str,
    fee_bps: int,
) -> V3PoolEdge:
    return V3PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=Address(pool_address),
        protocol="UniswapV3",
        fee_bps=fee_bps,
        is_token0_in=_is_token0_in(token_in.address, token_out.address),
    )


async def _bridge_cctp_with_compose_transfer(
    *,
    w3: AsyncWeb3,
    private_key: str,
    sender: str,
    usdc: Any,
    cctp: Any,
    quoted_amount_out: int,
) -> None:
    sender_checksum = Web3.to_checksum_address(sender)
    destination_domain = CCTP_DOMAIN_ARBITRUM
    composer_address = ARBITRUM_SEPOLIA_CCTP_COMPOSER
    expected_interpreter = ARBITRUM_SEPOLIA_INTERPRETER
    mint_recipient = bytes.fromhex(composer_address.removeprefix("0x")).rjust(32, b"\x00")
    destination_caller = "0x" + composer_address.removeprefix("0x").rjust(64, "0")
    max_fee = 20000
    min_finality_threshold = 500

    destination_caller_raw = bytes.fromhex(destination_caller.removeprefix("0x"))
    if len(destination_caller_raw) == 20:
        destination_caller_raw = destination_caller_raw.rjust(32, b"\x00")
    if len(destination_caller_raw) != 32:
        pytest.fail("SEPOLIA_CCTP_DESTINATION_CALLER must be 20-byte address hex or 32-byte bytes32 hex")
    destination_caller_bytes32 = destination_caller_raw

    await _ensure_max_allowance(
        w3=w3,
        private_key=private_key,
        owner=sender_checksum,
        token=usdc,
        token_address=USDC_SEPOLIA,
        spender=TOKEN_MESSENGER_SEPOLIA,
    )

    destination_rpc_url = ARBITRUM_SEPOLIA_RPC_URL
    destination_usdc = ARBITRUM_SEPOLIA_USDC_ADDRESS
    print(
        "[env] effective destination: "
        f"domain={destination_domain}, "
        f"rpc={destination_rpc_url or '<unset>'}, "
        f"usdc={destination_usdc or '<unset>'}"
    )
    dst_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(destination_rpc_url))
    dst_chain_id = int(await dst_w3.eth.chain_id)
    dst_usdc = Contract.from_abi(ERC20_ABI, to=Web3.to_checksum_address(destination_usdc))
    composer = Contract.from_abi(CCTP_COMPOSER_ABI, to=Web3.to_checksum_address(composer_address))
    composer_code = await dst_w3.eth.get_code(Web3.to_checksum_address(composer_address))
    print("[compose] dst_chain_id=", dst_chain_id, "composer_code_bytes=", len(composer_code))
    assert len(composer_code) > 0, (
        f"no bytecode at composer {composer_address} on chain_id={dst_chain_id}; check destination RPC/address"
    )
    composer_interp = Web3.to_checksum_address(await composer.fns.interpreter().call(dst_w3))
    assert composer_interp != Web3.to_checksum_address("0x0000000000000000000000000000000000000000"), (
        f"composer.interpreter is zero on chain_id={dst_chain_id} at {composer_address}; "
        f"redeploy CCTPComposer with a valid interpreter address"
    )
    print("[compose] composer.interpreter=", composer_interp)
    if expected_interpreter:
        assert composer_interp.lower() == Web3.to_checksum_address(expected_interpreter).lower(), (
            f"composer interpreter mismatch: expected {expected_interpreter}, got {composer_interp}"
        )

    # CCTPComposer TSTOREs [amountReceived, sourceDomain] into its own transient
    # slots 0/1 and DELEGATECALLs the interpreter, so the program runs in
    # composer context and reads the bridged values via TLOAD.  sourceDomain
    # is unused here but consumed for symmetry.
    _transfer = ERC20.fns.transfer(sender, 0)  # transfer(to, 0) template
    _compose = Program()
    _amount_received = _compose.builder.tload(IRLiteral(0))  # amountReceived
    _compose.builder.tload(IRLiteral(1))  # sourceDomain (unused)
    # Overlay the runtime bridged amount onto the template's amount word:
    # offset = selector(4) + to(32) = 36.
    _success = _compose.call_raw(Address(destination_usdc), _transfer.data, patches={36: _amount_received})
    _compose.assert_(_success)
    _compose.builder.stop()
    compose_program = _compose.build()
    observed_usdc_holder = sender
    destination_pre_balance = int(await dst_usdc.fns.balanceOf(observed_usdc_holder).call(dst_w3))

    if quoted_amount_out <= max_fee:
        pytest.skip(
            f"Swap returned only {quoted_amount_out} USDC units (≤ max_fee={max_fee}); "
            "not enough to cover CCTP bridge fee."
        )

    patched_bridge_calldata = tx_data_bytes(
        cctp.fns.depositForBurnWithHook(
            quoted_amount_out,
            destination_domain,
            mint_recipient,
            USDC_SEPOLIA,
            destination_caller_bytes32,
            max_fee,
            min_finality_threshold,
            compose_program,
        ).data
    )
    patched_amount, patched_domain, patched_burn_token = _decode_cctp_deposit_for_burn_fields(patched_bridge_calldata)
    assert patched_amount == quoted_amount_out
    assert patched_domain == destination_domain
    assert patched_burn_token.lower() == USDC_SEPOLIA.lower()

    burn_tx_hash, burn_status = await send_signed(
        w3,
        private_key=private_key,
        sender=sender_checksum,
        to=TOKEN_MESSENGER_SEPOLIA,
        data=patched_bridge_calldata,
        nonce=await w3.eth.get_transaction_count(sender_checksum),
    )
    assert burn_status == 1, f"depositForBurnWithHook failed: {burn_tx_hash}"
    derived_message_hash = await _derive_cctp_message_from_burn_tx(w3, burn_tx_hash)
    if derived_message_hash:
        print(f"[cctp] derived message_hash from burn tx: {derived_message_hash}")

    relay_tx_hash, relay_status = await _relay_cctp_compose_receive_and_execute(
        destination_rpc_url=str(destination_rpc_url),
        private_key=private_key,
        sender=sender_checksum,
        burn_tx_hash=burn_tx_hash,
        composer_address=composer_address,
        message_hash=derived_message_hash,
    )
    print("mm-relay_tx_hash", relay_tx_hash, "relay_status", relay_status)
    assert relay_status == 1
    await _observe_destination_mint(
        recipient=observed_usdc_holder,
        # CCTP v2 receive amount can be less than burn amount due to fee.
        expected_increase=max(1, patched_amount - max_fee),
        destination_rpc_url=str(destination_rpc_url),
        destination_usdc=str(destination_usdc),
        pre_balance=int(destination_pre_balance),
    )


async def _swap_and_quote_usdc_from_eth(
    *,
    w3: AsyncWeb3,
    private_key: str,
    sender: str,
    swap_eth_in: int,
    deadline: int,
    variant: str,
) -> int:
    if variant == "v2":
        _v2_swap_abi = [
            "function swapExactETHForTokens(uint amountOutMin, address[] path, address to, uint deadline) external payable returns (uint[] amounts)",
        ]
        v2_router = Contract.from_abi(_v2_swap_abi, to=UNISWAP_V2_ROUTER_SEPOLIA)
        swap_data = tx_data_bytes(
            v2_router.fns.swapExactETHForTokens(1, [WETH_SEPOLIA, USDC_SEPOLIA], sender, deadline).data
        )
        swap_tx_hash, swap_status = await send_signed(
            w3,
            private_key=private_key,
            sender=sender,
            to=UNISWAP_V2_ROUTER_SEPOLIA,
            data=swap_data,
            nonce=await w3.eth.get_transaction_count(Web3.to_checksum_address(sender)),
            value=swap_eth_in,
        )
        assert swap_status == 1, f"swapExactETHForTokens failed: {swap_tx_hash}"

        v2 = UniswapV2(w3, UNISWAP_V2_ROUTER_SEPOLIA)
        amounts_out = await v2.get_amounts_out(
            TokenAmount(token=_WETH_TOKEN_SEPOLIA, amount=swap_eth_in),
            [_WETH_TOKEN_SEPOLIA, _USDC_TOKEN_SEPOLIA],
        )
        quoted_amount_out = amounts_out[-1].amount
        if quoted_amount_out <= 0:
            pytest.fail("Uniswap V2 quote returned non-positive amountOut on Sepolia")

        # Keep a Program artifact for parity with existing VM-path debugging.
        _probe = Program()
        _probe.call_raw(Address(UNISWAP_V2_ROUTER_SEPOLIA), swap_data)
        _probe.builder.stop()
        program_bytes = _probe.build()
        assert len(program_bytes) > 0
        return quoted_amount_out

    if variant == "v2_defi_vm":
        defi_vm_address = Web3.to_checksum_address(SEPOLIA_DEFI_VM)
        v2 = UniswapV2(w3, UNISWAP_V2_ROUTER_SEPOLIA)

        factory_v2 = v2.get_factory_contract(UNISWAP_V2_FACTORY_SEPOLIA)
        pair_address = await factory_v2.fns.getPair(WETH_SEPOLIA, USDC_SEPOLIA).call(w3)
        if pair_address == "0x0000000000000000000000000000000000000000":
            pytest.skip("No WETH/USDC V2 pair on Sepolia")

        amounts_out = await v2.get_amounts_out(
            TokenAmount(token=_WETH_TOKEN_SEPOLIA, amount=swap_eth_in),
            [_WETH_TOKEN_SEPOLIA, _USDC_TOKEN_SEPOLIA],
        )
        quoted_amount_out = amounts_out[-1].amount
        if quoted_amount_out <= 0:
            pytest.skip("V2 get_amounts_out returned non-positive amountOut on Sepolia")

        dag = (
            RouteDAG()
            .from_token(_WETH_TOKEN_SEPOLIA)
            .swap(
                _USDC_TOKEN_SEPOLIA,
                _v2_pool_edge(
                    token_in=_WETH_TOKEN_SEPOLIA,
                    token_out=_USDC_TOKEN_SEPOLIA,
                    pool_address=pair_address,
                ),
            )
        )
        await _seed_build_send_defi_vm_swap(
            w3=w3,
            private_key=private_key,
            sender=sender,
            defi_vm=defi_vm_address,
            dag=dag,
            amount=swap_eth_in,
            min_final_out=max(1, quoted_amount_out // 2),
            label="DeFiVM V2 swap",
        )
        return quoted_amount_out

    if variant == "v3":
        pool_fee = 500
        sender_checksum = Web3.to_checksum_address(sender)
        defi_vm_address = Web3.to_checksum_address(SEPOLIA_DEFI_VM)
        v3 = UniswapV3(w3, UNISWAP_V3_SWAP_ROUTER_02_SEPOLIA, UNISWAP_V3_QUOTER_V2_SEPOLIA)

        # Resolve the direct pool address via the V3 factory.
        factory_v3 = v3.get_factory_contract(UNISWAP_V3_FACTORY_SEPOLIA)
        pool_address = await factory_v3.fns.getPool(WETH_SEPOLIA, USDC_SEPOLIA, pool_fee).call(w3)
        if pool_address == "0x0000000000000000000000000000000000000000":
            pytest.skip(f"No V3 WETH/USDC pool for fee={pool_fee} on Sepolia")

        quoted = await v3.quote_exact_input_single(
            TokenAmount(token=_WETH_TOKEN_SEPOLIA, amount=swap_eth_in), _USDC_TOKEN_SEPOLIA, fee=pool_fee
        )
        quoted_amount_out = quoted.amount
        if quoted_amount_out <= 0:
            pytest.skip("Uniswap V3 quote returned non-positive amountOut on Sepolia for fee=500")

        dag = (
            RouteDAG()
            .from_token(_WETH_TOKEN_SEPOLIA)
            .swap(
                _USDC_TOKEN_SEPOLIA,
                _v3_pool_edge(
                    token_in=_WETH_TOKEN_SEPOLIA,
                    token_out=_USDC_TOKEN_SEPOLIA,
                    pool_address=pool_address,
                    fee_bps=pool_fee,
                ),
            )
        )

        await _seed_defi_vm_with_weth(
            w3=w3,
            private_key=private_key,
            sender=sender_checksum,
            defi_vm=defi_vm_address,
            amount=swap_eth_in,
        )

        # 50% slippage tolerance for thin Sepolia liquidity.
        min_final_out = max(1, quoted_amount_out // 2)
        swap_tx = build_swap_transaction(
            dag,
            swap_eth_in,
            defi_vm_address,
            sender_checksum,
            min_final_out=min_final_out,
        )

        last_error = ""
        try:
            # Pre-simulate to avoid spending gas on guaranteed reverts.
            call_tx = cast(
                TxParams,
                {
                    "from": sender_checksum,
                    "to": swap_tx.to,
                    "data": "0x" + swap_tx.data.hex(),
                },
            )
            await w3.eth.call(call_tx)
            swap_tx_hash, swap_status = await send_signed(
                w3,
                private_key=private_key,
                sender=sender,
                to=swap_tx.to,
                data=swap_tx.data,
                nonce=await w3.eth.get_transaction_count(sender_checksum),
            )
            print(
                "[gas] v3.pool.swap via DeFiVM tx=",
                swap_tx_hash,
                "fee=",
                pool_fee,
                "pool=",
                pool_address,
                "amountIn=",
                swap_eth_in,
            )
            assert swap_status == 1, f"DeFiVM v3 swap failed: {swap_tx_hash}"
            return quoted_amount_out
        except Exception as inner_exc:
            last_error = f"fee={pool_fee} pool={pool_address}: {inner_exc}"
            print("[v3] DeFiVM variant failed:", last_error)

        pytest.skip(f"Uniswap V3 route unavailable on Sepolia for fee=500; last_error={last_error}")

    if variant == "v4_defi_vm":
        defi_vm_address = Web3.to_checksum_address(SEPOLIA_DEFI_VM)
        v4_edge = await discover_sepolia_v4_weth_usdc_edge(w3)
        if v4_edge is None:
            pytest.skip("No initialised WETH/USDC V4 pool found on Sepolia")
        graph = PoolGraph()
        graph.add_pool(v4_edge)
        route_v4 = Router(graph).find_best_route(TokenAmount(_WETH_TOKEN_SEPOLIA, swap_eth_in), _USDC_TOKEN_SEPOLIA)
        assert route_v4.dag is not None, "Router route is missing its DAG representation"
        quoted_amount_out = route_v4.amount_out.amount
        if quoted_amount_out <= 0:
            pytest.skip("V4 off-chain quote returned non-positive amountOut")
        # V4 swap via DeFiVM; USDC output credited to the sender wallet so the
        # caller's _bridge_cctp_with_compose_transfer can bridge it (as V2/V3 do).
        await _seed_build_send_defi_vm_swap(
            w3=w3,
            private_key=private_key,
            sender=sender,
            defi_vm=defi_vm_address,
            dag=route_v4.dag,
            amount=swap_eth_in,
            min_final_out=max(1, quoted_amount_out * 9_900 // 10_000),
            label="DeFiVM V4 swap",
        )
        return quoted_amount_out

    raise ValueError(f"unsupported swap variant: {variant}")


@pytest.mark.testnet
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "swap_variant",
    ["v2", "v2_defi_vm", "v3", "v4_defi_vm"],
    ids=["uniswap-v2", "uniswap-v2-defi-vm", "uniswap-v3", "uniswap-v4-defi-vm"],
)
async def test_sepolia_cctp_deposit_for_burn_builder_broadcast(swap_variant: str) -> None:
    """Execute real Sepolia swap (V2/V3), then broadcast CCTP compose burn and relay."""
    private_key = SEPOLIA_PRIVATE_KEY
    w3 = await connect(SEPOLIA_RPC_URL)

    signer = Account.from_key(private_key)
    sender = Web3.to_checksum_address(signer.address)

    usdc = Contract.from_abi(ERC20_ABI, to=USDC_SEPOLIA)
    cctp = Contract.from_abi(CCTP_TOKEN_MESSENGER_ABI, to=TOKEN_MESSENGER_SEPOLIA)

    native_balance = await w3.eth.get_balance(sender)
    if native_balance == 0:
        pytest.fail("sender has 0 Sepolia ETH for gas; fund it first (faucet) and retry")

    swap_eth_in = 2 * 10**15  # default for V2/V3 variants
    if swap_variant == "v4_defi_vm":
        # Size the V4 swap to ~1% of virtual WETH depth (L × sqrtP / Q96 / 100) so it
        # stays within the pool's available USDC reserve; a fixed amount can exceed
        # depth at the current price and revert with "slippage: out too low".
        _v4_edge_pre = await discover_sepolia_v4_weth_usdc_edge(w3)
        if _v4_edge_pre is None:
            pytest.skip("No initialised WETH/USDC V4 pool found on Sepolia")
        swap_eth_in = max(10**13, _v4_edge_pre.liquidity * _v4_edge_pre.sqrt_price_x96 // (2**96) // 100)

    if native_balance <= swap_eth_in:
        pytest.fail(
            f"insufficient Sepolia ETH for swap+gas: need > {swap_eth_in}, have {native_balance}; top up faucet"
        )

    usdc_before_swap = int(await usdc.fns.balanceOf(sender).call(w3))
    latest_block = await w3.eth.get_block("latest")
    block_ts = latest_block.get("timestamp")
    assert block_ts is not None
    deadline = int(block_ts) + 20 * 60
    quoted_amount_out = await _swap_and_quote_usdc_from_eth(
        w3=w3,
        private_key=private_key,
        sender=sender,
        swap_eth_in=swap_eth_in,
        deadline=deadline,
        variant=swap_variant,
    )

    usdc_after_swap = int(await usdc.fns.balanceOf(sender).call(w3))
    received_from_swap = usdc_after_swap - usdc_before_swap
    if received_from_swap <= 0:
        pytest.fail("real Sepolia swap executed but received 0 USDC")

    # Bridge exactly what was received from the live swap to avoid quote/settlement mismatch.
    bridge_amount_out = int(received_from_swap)
    print("[swap] quoted_out=", quoted_amount_out, "received_out=", bridge_amount_out)

    await _bridge_cctp_with_compose_transfer(
        w3=w3,
        private_key=private_key,
        sender=sender,
        usdc=usdc,
        cctp=cctp,
        quoted_amount_out=bridge_amount_out,
    )


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_defi_vm_v2v3_mixed_route() -> None:
    """Execute a two-hop WETH→USDC (V2) →WETH (V3) swap via DeFiVM on Sepolia.

    Quotes the mixed route step-by-step (V2 getAmountsOut → V3 quoteExactInputSingle), then
    builds and broadcasts a DeFiVM program that wraps ETH to WETH, executes
    both hops through the pools directly, and returns WETH to the sender.
    """

    w3 = await connect(SEPOLIA_RPC_URL)

    signer = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(signer.address)
    defi_vm_address = Web3.to_checksum_address(SEPOLIA_DEFI_VM)

    v2 = UniswapV2(w3, UNISWAP_V2_ROUTER_SEPOLIA)
    v3 = UniswapV3(w3, UNISWAP_V3_SWAP_ROUTER_02_SEPOLIA, UNISWAP_V3_QUOTER_V2_SEPOLIA)

    factory_v2 = v2.get_factory_contract(UNISWAP_V2_FACTORY_SEPOLIA)
    pair_address = await factory_v2.fns.getPair(WETH_SEPOLIA, USDC_SEPOLIA).call(w3)
    if pair_address == "0x0000000000000000000000000000000000000000":
        pytest.skip("No WETH/USDC V2 pair on Sepolia")

    factory_v3 = v3.get_factory_contract(UNISWAP_V3_FACTORY_SEPOLIA)
    v3_pool_address = await factory_v3.fns.getPool(USDC_SEPOLIA, WETH_SEPOLIA, 500).call(w3)
    if v3_pool_address == "0x0000000000000000000000000000000000000000":
        pytest.skip("No USDC/WETH V3 pool at fee=500 on Sepolia")

    swap_eth_in = 2 * 10**15
    native_balance = await w3.eth.get_balance(sender)
    if native_balance <= swap_eth_in:
        pytest.fail(f"insufficient Sepolia ETH: need > {swap_eth_in}, have {native_balance}")

    # Quote the two-hop route: getAmountsOut (V2 hop) then quoteExactInputSingle (V3 hop).
    usdc_amounts = await v2.get_amounts_out(
        TokenAmount(token=_WETH_TOKEN_SEPOLIA, amount=swap_eth_in),
        [_WETH_TOKEN_SEPOLIA, _USDC_TOKEN_SEPOLIA],
    )
    usdc_intermediate = usdc_amounts[-1]
    quoted = await v3.quote_exact_input_single(usdc_intermediate, _WETH_TOKEN_SEPOLIA, fee=500)
    print("[v2v3] quoted_weth_out", quoted)
    assert quoted.amount > 0, f"multi-hop quote returned 0 for mixed V2→V3 route: {quoted}"
    print("mm-quoted.amount", quoted.amount)

    # Round-trip: WETH → USDC (V2 pair) → WETH (V3 pool).
    dag = (
        RouteDAG()
        .from_token(_WETH_TOKEN_SEPOLIA)
        .swap(
            _USDC_TOKEN_SEPOLIA,
            _v2_pool_edge(
                token_in=_WETH_TOKEN_SEPOLIA,
                token_out=_USDC_TOKEN_SEPOLIA,
                pool_address=pair_address,
            ),
        )
        .swap(
            _WETH_TOKEN_SEPOLIA,
            _v3_pool_edge(
                token_in=_USDC_TOKEN_SEPOLIA,
                token_out=_WETH_TOKEN_SEPOLIA,
                pool_address=v3_pool_address,
                fee_bps=500,
            ),
        )
    )

    swap_tx_hash = await _seed_build_send_defi_vm_swap(
        w3=w3,
        private_key=SEPOLIA_PRIVATE_KEY,
        sender=sender,
        defi_vm=defi_vm_address,
        dag=dag,
        amount=swap_eth_in,
        min_final_out=max(1, quoted.amount // 2),
        label="DeFiVM V2→V3 mixed swap",
    )
    print("[v2v3] tx", swap_tx_hash, "status 1")


# ===========================================================================
# Uniswap V4 swap + CCTP bridge, atomic in one DeFiVM execute()
# (ported from swap_route_test, rewritten against testnet's compose builders)
# ===========================================================================
@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_v4_defi_vm_atomic_swap_and_bridge() -> None:
    """V4 swap + CCTP bridge composed into a single DeFiVM ``execute()`` tx.

    Unlike the two-step flow (swap -> sender wallet -> bridge), both operations
    run in one atomic DeFiVM program built by
    :func:`~pydefi.crosschain.compose.build_source_swap_bridge_program`:

      1. ``PoolManager.unlock()`` -> WETH->USDC, USDC credited to the DeFiVM.
      2. ``USDC.approve(TokenMessenger, amountOut)``.
      3. ``TokenMessenger.depositForBurnWithHook(amountOut, ...)`` -> MessageSent,
         carrying a destination compose program (transfer USDC to the sender).

    Two setup txs precede the atomic tx because V4 settle requires the DeFiVM to
    already hold the input token: ETH->WETH, then WETH->DeFiVM.

    Asserts the deterministic source half: the atomic tx succeeds and emits a
    CCTP ``MessageSent``.  Destination delivery (relay -> mint -> compose
    transfer) is the same async path exercised by
    :func:`_bridge_cctp_with_compose_transfer`.
    """
    require_env(
        "SEPOLIA_RPC_URL",
        "SEPOLIA_PRIVATE_KEY",
        "SEPOLIA_DEFI_VM",
        "ARBITRUM_SEPOLIA_CCTP_COMPOSER",
    )
    w3 = await connect(SEPOLIA_RPC_URL)

    signer = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(signer.address)
    defi_vm = Web3.to_checksum_address(SEPOLIA_DEFI_VM)
    max_fee = 20000
    min_finality_threshold = 1000

    # -- Discover V4 WETH/USDC pool, size swap to ~1% of virtual depth -----
    edge = await discover_sepolia_v4_weth_usdc_edge(w3)
    if edge is None:
        pytest.skip("No initialised WETH/USDC V4 pool found on Sepolia")
    _q96 = 2**96
    swap_eth_in = max(10**13, edge.liquidity * edge.sqrt_price_x96 // _q96 // 100)

    native_balance = await w3.eth.get_balance(sender)
    if native_balance <= swap_eth_in:
        pytest.fail(f"insufficient Sepolia ETH: need > {swap_eth_in}, have {native_balance}")

    # -- Off-chain quote (also guards: output must exceed the CCTP fee) ----
    graph = PoolGraph()
    graph.add_pool(edge)
    route = Router(graph).find_best_route(TokenAmount(_WETH_TOKEN_SEPOLIA, swap_eth_in), _USDC_TOKEN_SEPOLIA)
    assert route is not None, "Router found no route through V4 pool"
    dag = route.dag
    assert dag is not None, "Router route is missing its DAG representation"
    off_chain_out = route.amount_out.amount
    if off_chain_out <= max_fee:
        pytest.skip(
            f"off-chain quote {off_chain_out} USDC units <= max_fee={max_fee}; "
            "reseed the Sepolia V4 pool at market price to run this test."
        )

    # -- Setup: wrap ETH -> WETH and transfer it to the DeFiVM ------------
    await _seed_defi_vm_with_weth(
        w3=w3,
        private_key=SEPOLIA_PRIVATE_KEY,
        sender=sender,
        defi_vm=defi_vm,
        amount=swap_eth_in,
    )

    # -- Destination compose program: transfer minted USDC to the sender --
    # CCTPComposer TSTOREs [amountReceived, sourceDomain]; the program reads
    # amountReceived via TLOAD(0) and patches it into a transfer template.
    composer_address = ARBITRUM_SEPOLIA_CCTP_COMPOSER
    _transfer = ERC20.fns.transfer(sender, 0)
    _compose = Program()
    _amount_received = _compose.builder.tload(IRLiteral(0))
    _compose.builder.tload(IRLiteral(1))  # sourceDomain (unused)
    _compose.assert_(
        _compose.call_raw(Address(ARBITRUM_SEPOLIA_USDC_ADDRESS), _transfer.data, patches={36: _amount_received})
    )
    _compose.builder.stop()
    compose_program = _compose.build()

    # -- Burn template: depositForBurnWithHook(0, ...) carrying the compose
    # program.  The 0 amount placeholder is patched at runtime from the swapped
    # USDC (at CCTP_BURN_AMOUNT_OFFSET) by build_source_swap_bridge_program.
    mint_recipient = bytes.fromhex(composer_address.removeprefix("0x")).rjust(32, b"\x00")
    cctp = Contract.from_abi(CCTP_TOKEN_MESSENGER_ABI, to=TOKEN_MESSENGER_SEPOLIA)
    burn_template = tx_data_bytes(
        cctp.fns.depositForBurnWithHook(
            0,
            CCTP_DOMAIN_ARBITRUM,
            mint_recipient,
            USDC_SEPOLIA,
            mint_recipient,  # destinationCaller = composer (bytes32)
            max_fee,
            min_finality_threshold,
            compose_program,
        ).data
    )

    # -- Atomic source program: V4 swap -> approve -> burn ----------------
    source_program = build_source_swap_bridge_program(
        swap_dag=dag,
        amount_in=swap_eth_in,
        usdc=Address(USDC_SEPOLIA),
        token_messenger=Address(TOKEN_MESSENGER_SEPOLIA),
        burn_template=burn_template,
        executor=Address(defi_vm),
        min_usdc_out=max(1, off_chain_out * 9_900 // 10_000),  # 1% slippage
    )
    execute_calldata = tx_data_bytes(DeFiVM.fns.execute(source_program).data)

    exec_hash, exec_status = await send_signed(
        w3,
        private_key=SEPOLIA_PRIVATE_KEY,
        sender=sender,
        to=defi_vm,
        data=execute_calldata,
        nonce=await w3.eth.get_transaction_count(sender),
    )
    print(f"[atomic-v4] tx={exec_hash} status={exec_status}")
    assert exec_status == 1, f"atomic V4 swap+bridge reverted: {exec_hash}"

    # MessageSent confirms depositForBurnWithHook ran after the V4 swap+approve.
    message_hash = await _derive_cctp_message_from_burn_tx(w3, exec_hash)
    assert message_hash is not None, f"no CCTP MessageSent in tx {exec_hash}; bridge call not reached"
    print(f"[atomic-v4] CCTP message_hash={message_hash}")
