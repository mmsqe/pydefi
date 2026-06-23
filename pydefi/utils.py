import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from eth_account import Account
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3


def load_local_env(*, override: bool = True) -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[7:].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ or not os.environ.get(key, "").strip()):
            os.environ[key] = value


async def get_json(url: str) -> dict[str, Any]:
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )

    for headers in (
        {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": user_agent,
        },
        {"User-Agent": user_agent},
    ):
        try:

            def load() -> dict[str, Any]:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=30) as resp:
                    return cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))

            return await asyncio.to_thread(load)
        except HTTPError as exc:
            if exc.code != 403 or headers == {"User-Agent": user_agent}:
                raise

    raise RuntimeError("unreachable")


def tx_data_bytes(raw: Any) -> bytes:
    return to_return_bytes(raw.get("data") if isinstance(raw, dict) else raw)


def to_return_bytes(raw: Any) -> bytes:
    return bytes(HexBytes(raw))


def _strip_0x(s: str) -> str:
    return s.removeprefix("0x").removeprefix("0X")


def with_0x(s: str) -> str:
    return f"0x{_strip_0x(s)}"


def decode_hex_field(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return bytes(HexBytes(with_0x(value)))
    except ValueError:
        return None


async def send_signed(
    w3: AsyncWeb3,
    *,
    private_key: str,
    sender: str,
    to: str,
    data: bytes,
    nonce: int,
    value: int = 0,
) -> tuple[str, int]:
    gas_price = await w3.eth.gas_price
    estimate = await w3.eth.estimate_gas(
        {
            "from": sender,
            "to": Web3.to_checksum_address(to),
            "data": "0x" + data.hex(),
            "value": value,
        }
    )
    tx = {
        "to": Web3.to_checksum_address(to),
        "value": value,
        "data": "0x" + data.hex(),
        "nonce": nonce,
        "chainId": await w3.eth.chain_id,
        "gas": int(estimate * 12 // 10),
        "gasPrice": int(gas_price * 12 // 10),
    }
    signed = Account.sign_transaction(tx, private_key)
    tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
    # Sepolia/testnet inclusion latency can exceed the web3 default (120s)
    # when mempool is backed up or base fee jumps; budget 3 minutes.
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    return tx_hash.hex(), int(receipt["status"])


ERC20_ABI = [
    "function balanceOf(address owner) external view returns (uint256)",
    "function allowance(address owner, address spender) external view returns (uint256)",
    "function approve(address spender, uint256 amount) external returns (bool)",
    "function transfer(address to, uint256 amount) external returns (bool)",
]

CCTP_TOKEN_MESSENGER_ABI = [
    "function depositForBurn(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller, uint256 maxFee, uint32 minFinalityThreshold) external returns (uint64 nonce)",
    "function depositForBurnWithHook(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller, uint256 maxFee, uint32 minFinalityThreshold, bytes hookData) external returns (uint64 nonce)",
]

CCTP_COMPOSER_ABI = [
    "function receiveAndExecute(bytes message, bytes attestation) external payable",
    "function interpreter() external view returns (address)",
]
