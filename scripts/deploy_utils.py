"""Shared CREATE2 deployment utilities for Sepolia/Arbitrum deploy scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import solcx
from web3 import Web3
from web3.types import TxParams

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFI_VM_SOL = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# Analog-Labs raw-bytecode EVM interpreter, live at the same address on mainnet,
# Sepolia and Arbitrum-Sepolia.  It is a DeFiVM constructor arg, so baking it in
# (together with via_ir=False below) makes the CREATE2 DeFiVM address match the
# mainnet deployment 0x96074E76d6eDF2B0BaEd3dd91fE2B9D0ee6EA06b on every chain.
DEFAULT_INTERPRETER = Web3.to_checksum_address("0x0000000000001e3F4F615cd5e20c681Cf7d85e8D")
# The interpreter the already-deployed Arbitrum-Sepolia CCTPComposer was built
# against; kept distinct so re-running the composer deploy reuses that address
# instead of creating a second composer.
DEFAULT_COMPOSER_INTERPRETER = Web3.to_checksum_address("0x64fE558B0F9a5dC18D4A36c85Ba99c3f222F7bde")
DEFAULT_CREATE2_DEPLOYER = Web3.to_checksum_address("0x4e59b44847b379578588920cA78FbF26c0B4956C")


def parse_salt_hex(value: str) -> bytes:
    """Parse a 32-byte CREATE2 salt from a 0x-prefixed hex string."""
    raw = bytes.fromhex(value.removeprefix("0x"))
    if len(raw) != 32:
        raise ValueError(f"CREATE2 salt must be exactly 32 bytes, got {len(raw)}")
    return raw


def compute_create2_address(factory: str, salt: bytes, init_code: bytes) -> str:
    """Return the deterministic address for a CREATE2 deployment."""
    h = Web3.keccak(b"\xff" + bytes.fromhex(factory.removeprefix("0x")) + salt + Web3.keccak(init_code))
    return Web3.to_checksum_address("0x" + h[-20:].hex())


def ensure_solc(version: str = "0.8.24") -> None:
    """Install *version* of solc if not already available."""
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def compile_contract(path: Path, contract_name: str, *, via_ir: bool = False) -> dict[str, Any]:
    """Compile *contract_name* from *path* and return the ``{"abi": …, "bin": …}`` dict."""
    ensure_solc("0.8.24")
    extra: dict[str, Any] = {}
    if via_ir:
        extra["via_ir"] = True
    compiled = solcx.compile_files(
        [str(path)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
        evm_version="cancun",  # required for tload/tstore (EIP-1153)
        **extra,
    )
    key = next(k for k in compiled if k.endswith(f":{contract_name}"))
    return compiled[key]


def deploy_contract_create2(
    w3: Web3,
    *,
    abi: list[dict[str, Any]],
    bytecode: str,
    deployer: Any,
    chain_id: int,
    constructor_args: tuple[Any, ...],
    factory: str,
    salt: bytes,
) -> str:
    """Deploy *bytecode* via the CREATE2 *factory* and return the deployed address.

    Idempotent: if bytecode is already deployed at the predicted address, that
    address is returned immediately without sending a transaction.
    """
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    init_code_hex = contract.constructor(*constructor_args).build_transaction({"from": deployer.address})["data"]
    init_code = bytes.fromhex(init_code_hex.removeprefix("0x"))
    predicted = compute_create2_address(factory, salt, init_code)

    if len(w3.eth.get_code(Web3.to_checksum_address(predicted))) > 1:
        return predicted

    call_data = salt + init_code

    # Pre-flight: simulate via eth_call.  Arachnid's deterministic deployer
    # (and most CREATE2 factories) returns the deployed address as the last
    # 32 bytes (or 20 bytes) of returndata.  All-zero returndata means
    # CREATE2 returned 0 — i.e. the constructor reverted, the address was
    # already taken, or the factory at this address doesn't actually do
    # CREATE2.  Surface that distinction here instead of after a sent tx.
    try:
        simulated = w3.eth.call(
            cast(
                TxParams,
                {
                    "from": deployer.address,
                    "to": Web3.to_checksum_address(factory),
                    "data": "0x" + call_data.hex(),
                    "value": 0,
                },
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"CREATE2 simulation reverted on factory {factory}: {exc}.  "
            f"Check the factory has the standard Arachnid deployer bytecode "
            f"(0x4e59b44847b379578588920cA78FbF26c0B4956C) or set CREATE2_DEPLOYER "
            f"to a working factory."
        ) from exc

    sim_hex = simulated.hex() if simulated else ""
    sim_addr = ("0x" + sim_hex[-40:]) if len(sim_hex) >= 40 else ""
    if not sim_addr or int(sim_addr, 16) == 0:
        raise RuntimeError(
            f"CREATE2 factory {factory} returned zero address from simulation.  "
            f"Likely causes: (1) constructor reverted (check args: {constructor_args}); "
            f"(2) address {predicted} already has bytecode from an earlier deploy; "
            f"(3) factory at this address is not actually a CREATE2 deployer "
            f"(returndata={sim_hex!r})."
        )
    if Web3.to_checksum_address(sim_addr) != Web3.to_checksum_address(predicted):
        raise RuntimeError(
            f"CREATE2 address mismatch: predicted {predicted}, factory simulation returned "
            f"{Web3.to_checksum_address(sim_addr)}.  Either the factory uses a non-standard "
            f"address-derivation rule, or compute_create2_address has the wrong factory address."
        )

    gas_price = w3.eth.gas_price

    def build_tx(nonce: int) -> dict[str, Any]:
        tx: dict[str, Any] = {
            "from": deployer.address,
            "to": Web3.to_checksum_address(factory),
            "data": "0x" + call_data.hex(),
            "nonce": nonce,
            "chainId": chain_id,
            "gasPrice": int(gas_price * 12 // 10),
            "value": 0,
        }
        tx["gas"] = int(w3.eth.estimate_gas(cast(TxParams, tx)) * 12 // 10)
        return tx

    nonce = w3.eth.get_transaction_count(deployer.address, "pending")
    tx = build_tx(nonce)
    signed = deployer.sign_transaction(tx)
    try:
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except Exception as exc:
        if "nonce too low" not in str(exc).lower():
            raise
        nonce = w3.eth.get_transaction_count(deployer.address, "pending")
        tx = build_tx(nonce)
        signed = deployer.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if int(receipt["status"]) != 1:
        raise RuntimeError(f"CREATE2 deployment failed: {tx_hash.hex()}")

    # Arbitrum (and some other L2s) can lag between receipt visibility and
    # post-state code visibility on the same RPC.  Poll a few times before
    # declaring the deploy a silent no-op.
    import time

    for _ in range(10):
        if len(w3.eth.get_code(Web3.to_checksum_address(predicted))) > 1:
            return predicted
        time.sleep(0.5)

    raise RuntimeError(
        f"CREATE2 deployment tx {tx_hash.hex()} succeeded "
        f"(block={receipt.get('blockNumber')}, gasUsed={receipt.get('gasUsed')}) "
        f"but no code at predicted address {predicted} after 5s.  Simulation returned "
        f"{sim_addr} so the prediction was correct; the live deploy silently no-op'd.  "
        f"Either the factory at {factory} on this chain has non-standard bytecode "
        f"(eth_call simulates but the real CREATE2 path differs), or the constructor "
        f"reverted only in real-execution mode (e.g. an opcode disallowed on this L2).  "
        f"Cross-check the factory's runtime bytecode against the Arachnid deployer "
        f"reference (https://github.com/Arachnid/deterministic-deployment-proxy)."
    )
