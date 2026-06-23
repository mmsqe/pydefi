"""Sign-and-broadcast testnet test for the LayerZero OFT compose path.

Parallels ``test_crosschain_cctp_sepolia_live.py`` but for an OFT bridge.  Unlike
CCTP — whose attestation we poll and relay by hand — LayerZero delivers the
compose via its own DVN/executor, so this test verifies the deterministic half:
``build_compose_route`` over :class:`~pydefi.bridge.LayerZeroOFT` produces a real
OFT ``send`` carrying the DeFiVM program as ``composeMsg`` plus ``extraOptions``
funding the destination ``lzReceive`` + ``lzCompose`` gas, which the OFT contract
and LayerZero endpoint accept and dispatch on the source chain.  Destination
execution (``lzCompose`` -> DeFiVM) is delivered asynchronously by LayerZero —
follow the printed LayerZero Scan link to confirm it landed.

Gated on:
``SEPOLIA_RPC_URL``, ``SEPOLIA_PRIVATE_KEY``, ``SEPOLIA_OFT_ADDRESS`` (a native
OFT with a peer set to the destination; when the key holds none the test
self-funds via the OFT's own open ``mint`` — the Sepolia demo OFT is its own
permissionless faucet — and only skips if that mint is unavailable),
``OFT_DST_CHAIN_ID`` (destination testnet EVM chain id, must be in ``_LZ_EID``),
``OFT_COMPOSER_ADDRESS`` (the :class:`OFTComposer` / recipient on the
destination).  Optional ``OFT_SEND_AMOUNT`` (raw units) caps the amount sent.

Run with::

    pytest -m testnet tests/live/test_crosschain_oft_sepolia_live.py
"""

from __future__ import annotations

import os

import pytest
from eth_account import Account
from eth_contract import Contract
from hexbytes import HexBytes
from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3, Web3

from pydefi.bridge.layerzero_oft import LayerZeroOFT
from pydefi.crosschain.route import build_compose_route
from pydefi.types import Address, Token, TokenAmount
from pydefi.utils import send_signed, tx_data_bytes
from pydefi.vm.context import Program
from tests.live.sepolia_helpers import connect, require_env

_SEPOLIA = 11155111
_OFT_ABI = [
    "function balanceOf(address owner) external view returns (uint256)",
    "function decimals() external view returns (uint8)",
    # The Sepolia demo OFT is its own permissionless faucet (open mint).
    "function mint(address to, uint256 amount) external",
]
#: Self-fund this many whole tokens when the key holds no OFT (see _faucet_mint).
_FAUCET_MINT_TOKENS = 100


def _amount_probe_program() -> bytes:
    """A small, real DeFiVM program: read the delivered amount from transient
    slot 0 (where :class:`OFTComposer` stages ``amountLD``) and stash it in slot
    2.  Committed as the ``composeMsg``; the source ``send`` only carries it, so a
    trivial-but-distinctive program keeps the test on the dispatch half while
    still letting us assert the program rides in the calldata."""
    prog = Program()
    prog.builder.tstore(IRLiteral(2), prog.builder.tload(IRLiteral(0)))
    prog.builder.stop()
    return prog.build()


async def _faucet_mint(w3: AsyncWeb3, private_key: str, sender: str, oft_address: str, amount: int) -> int:
    """Self-fund *sender* with *amount* raw units via the OFT's own ``mint(to, amount)``.

    The Sepolia demo OFT is its own permissionless faucet, so the key can mint to
    itself when it holds none — no separate faucet contract needed.  Returns the
    post-mint balance, or ``0`` if the OFT has no public mint (the call reverts);
    the caller then skips, preserving the behaviour for real, non-mintable OFTs.
    """
    erc20 = Contract.from_abi(_OFT_ABI, to=oft_address)
    try:
        nonce = int(await w3.eth.get_transaction_count(Web3.to_checksum_address(sender), "pending"))
        _tx_hash, status = await send_signed(
            w3,
            private_key=private_key,
            sender=sender,
            to=oft_address,
            data=tx_data_bytes(erc20.fns.mint(sender, amount).data),
            nonce=nonce,
        )
    except Exception:  # noqa: BLE001 — any mint failure falls back to skip
        return 0
    if status != 1:
        return 0
    return int(await erc20.fns.balanceOf(sender).call(w3))


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_oft_compose_send_broadcast() -> None:
    require_env(
        "SEPOLIA_RPC_URL",
        "SEPOLIA_PRIVATE_KEY",
        "SEPOLIA_OFT_ADDRESS",
        "OFT_DST_CHAIN_ID",
        "OFT_COMPOSER_ADDRESS",
    )
    private_key = os.environ["SEPOLIA_PRIVATE_KEY"].strip()
    w3 = await connect(os.environ["SEPOLIA_RPC_URL"].strip())
    sender = Web3.to_checksum_address(Account.from_key(private_key).address)

    oft_address = Web3.to_checksum_address(os.environ["SEPOLIA_OFT_ADDRESS"].strip())
    dst_chain_id = int(os.environ["OFT_DST_CHAIN_ID"].strip())
    composer = Address(HexBytes(Web3.to_checksum_address(os.environ["OFT_COMPOSER_ADDRESS"].strip())))

    # The OFT token (this class's unified-address model: the bridged token *is*
    # the OFT contract).  decimals is metadata only — the send uses raw units.
    erc20 = Contract.from_abi(_OFT_ABI, to=oft_address)
    decimals = int(await erc20.fns.decimals().call(w3))
    balance = int(await erc20.fns.balanceOf(sender).call(w3))
    if balance == 0:
        # Self-fund via the OFT's own faucet rather than skipping; only skip if the
        # OFT exposes no public mint (then there is no permissionless way to fund).
        balance = await _faucet_mint(w3, private_key, sender, oft_address, _FAUCET_MINT_TOKENS * 10**decimals)
    if balance == 0:
        pytest.skip(f"sender holds no OFT at {oft_address} and it has no public mint; fund it first")
    requested = int(os.environ.get("OFT_SEND_AMOUNT", "0").strip() or 0)
    amount = min(requested, balance) if requested else balance

    oft_token = Token(chain_id=_SEPOLIA, address=Address(HexBytes(oft_address)), symbol="OFT", decimals=decimals)
    lane = LayerZeroOFT(w3=w3, src_chain_id=_SEPOLIA, dst_chain_id=dst_chain_id, oft_address=oft_address)

    program = _amount_probe_program()
    route = await build_compose_route(
        bridge=lane,
        amount_in=TokenAmount(oft_token, amount),
        composer=composer,
        dest_program=program,
        refund_address=Address(HexBytes(sender)),
    )
    # A native OFT burns from the sender, so the only leg is the compose send.
    assert [leg.kind for leg in route.steps] == ["bridge_compose"]
    send = route.steps[0].tx
    assert program.hex() in send["data"].lower()  # the program rides as composeMsg

    native_fee = int(send["value"])
    if int(await w3.eth.get_balance(sender)) <= native_fee:
        pytest.fail(f"insufficient Sepolia ETH for the LZ fee {native_fee} + gas; top up the faucet")

    tx_hash, status = await send_signed(
        w3,
        private_key=private_key,
        sender=sender,
        to=Web3.to_checksum_address(send["to"]),
        data=bytes.fromhex(send["data"].removeprefix("0x")),
        nonce=int(await w3.eth.get_transaction_count(sender)),
        value=native_fee,
    )
    assert status == 1, f"OFT compose send reverted: {tx_hash}"

    # The send emits the OFT ``OFTSent`` and the endpoint ``PacketSent``; non-empty
    # logs confirm the cross-chain compose message was dispatched.  Destination
    # execution is delivered asynchronously — track it on LayerZero Scan.
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    assert len(receipt["logs"]) > 0, "no logs emitted; the LayerZero packet was not dispatched"
    tx_ref = tx_hash if tx_hash.startswith("0x") else f"0x{tx_hash}"
    print(f"LayerZero Scan: https://testnet.layerzeroscan.com/tx/{tx_ref}")
