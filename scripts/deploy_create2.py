"""Deploy DeFiVM (and optionally CCTPComposer) via CREATE2.

Usage::

    python3 scripts/deploy_create2.py sepolia
    python3 scripts/deploy_create2.py arbitrum-sepolia

Per-chain config lives in :data:`CHAINS`.  All other knobs are env-var
overrides documented in the chain block below.
"""

from __future__ import annotations

import argparse
import os

from deploy_utils import (
    DEFAULT_COMPOSER_INTERPRETER,
    DEFAULT_CREATE2_DEPLOYER,
    DEFAULT_INTERPRETER,
    REPO_ROOT,
    compile_contract,
    deploy_contract_create2,
    parse_salt_hex,
)
from eth_account import Account
from web3 import HTTPProvider, Web3

from pydefi.utils import load_local_env

load_local_env()

DEFI_VM_SOL = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"
CCTP_COMPOSER_SOL = REPO_ROOT / "pydefi" / "bridge" / "CCTPComposer.sol"

CREATE2_SALT_VM = "0x1111111111111111111111111111111111111111111111111111111111111111"
CREATE2_SALT_COMPOSER = "0x2222222222222222222222222222222222222222222222222222222222222222"

# Per-chain config.  ``composer`` is None when the chain doesn't deploy a
# CCTPComposer; otherwise it carries the testnet-specific defaults that env
# vars (``CCTP_MESSAGE_TRANSMITTER``, ``<CHAIN>_USDC``) can override.
CHAINS: dict[str, dict] = {
    "sepolia": {
        "chain_id": 11155111,
        "rpc_env": "SEPOLIA_RPC_URL",
        "composer": None,
    },
    "arbitrum-sepolia": {
        "chain_id": 421614,
        "rpc_env": "ARBITRUM_SEPOLIA_RPC_URL",
        "composer": {
            "usdc": "0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d",
            "usdc_env": "ARBITRUM_SEPOLIA_USDC",
            "message_transmitter": "0xE737e5cEBEEBa77EFE34D4aa090756590b1CE275",
        },
    },
}


def main(chain: str) -> None:
    cfg = CHAINS[chain]

    rpc_url = os.getenv(cfg["rpc_env"], "").strip()
    private_key = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
    if not rpc_url:
        raise SystemExit(f"{cfg['rpc_env']} is not set")
    if not private_key:
        raise SystemExit("SEPOLIA_PRIVATE_KEY is not set")

    w3 = Web3(HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise SystemExit(f"Failed to connect to {cfg['rpc_env']}")

    chain_id = int(w3.eth.chain_id)
    if chain_id != cfg["chain_id"]:
        raise SystemExit(f"Expected chain_id={cfg['chain_id']}, got {chain_id}")

    deployer = Account.from_key(private_key)
    create2_factory = Web3.to_checksum_address(os.getenv("CREATE2_DEPLOYER", DEFAULT_CREATE2_DEPLOYER))
    interpreter = Web3.to_checksum_address(os.getenv("DEFI_VM_INTERPRETER", DEFAULT_INTERPRETER))

    if len(w3.eth.get_code(create2_factory)) <= 1:
        raise SystemExit(
            f"No code at CREATE2 deployer {create2_factory}. Set CREATE2_DEPLOYER to a valid factory on this chain."
        )
    if len(w3.eth.get_code(interpreter)) <= 1:
        raise SystemExit(
            "No interpreter code at DEFI_VM_INTERPRETER/default address. "
            "Deploy interpreter first and set DEFI_VM_INTERPRETER."
        )

    # Mainnet recipe: via_ir=False + the Analog interpreter (above) so the
    # CREATE2 address matches the mainnet DeFiVM on every chain.
    vm_compiled = compile_contract(DEFI_VM_SOL, "DeFiVM", via_ir=False)
    vm_salt = parse_salt_hex(os.getenv("CREATE2_SALT_VM", CREATE2_SALT_VM))
    vm_address = deploy_contract_create2(
        w3,
        abi=vm_compiled["abi"],
        bytecode=vm_compiled["bin"],
        deployer=deployer,
        chain_id=chain_id,
        constructor_args=(interpreter,),
        factory=create2_factory,
        salt=vm_salt,
    )

    composer_address = None
    if cfg["composer"] is not None:
        c = cfg["composer"]
        owner = Web3.to_checksum_address(os.getenv("CCTP_COMPOSER_OWNER", deployer.address))
        message_transmitter = Web3.to_checksum_address(os.getenv("CCTP_MESSAGE_TRANSMITTER", c["message_transmitter"]))
        usdc = Web3.to_checksum_address(os.getenv(c["usdc_env"], c["usdc"]))
        composer_compiled = compile_contract(CCTP_COMPOSER_SOL, "CCTPComposer")
        composer_salt = parse_salt_hex(os.getenv("CREATE2_SALT_COMPOSER", CREATE2_SALT_COMPOSER))
        # Pass the EVM *interpreter* (raw-bytecode interpreter), NOT the DeFiVM
        # contract.  CCTPComposer.receiveAndExecute DELEGATECALLs the interpreter
        # directly with the program as calldata; if vm_address is passed here the
        # program bytes hit DeFiVM's selector dispatch and fall into
        # DEXCallbackRouter's fallback ("unknown callback selector").
        #
        # The composer keeps its own interpreter (the one the already-deployed
        # Arbitrum-Sepolia composer was built against), decoupled from the DeFiVM
        # interpreter above, so this stays idempotent — re-running reuses the
        # existing composer rather than minting a second one at a new address.
        composer_interpreter = Web3.to_checksum_address(
            os.getenv("CCTP_COMPOSER_INTERPRETER", DEFAULT_COMPOSER_INTERPRETER)
        )
        composer_address = deploy_contract_create2(
            w3,
            abi=composer_compiled["abi"],
            bytecode=composer_compiled["bin"],
            deployer=deployer,
            chain_id=chain_id,
            constructor_args=(message_transmitter, usdc, composer_interpreter, owner),
            factory=create2_factory,
            salt=composer_salt,
        )

    print(f"Deployment complete ({chain} CREATE2)")
    print(f"DeFiVM: {vm_address}")
    if composer_address is not None:
        print(f"CCTPComposer: {composer_address}")
    print(f"CREATE2 deployer: {create2_factory}")
    print(f"CREATE2 vm salt: 0x{vm_salt.hex()}")
    if composer_address is not None:
        print(f"CREATE2 composer salt: 0x{composer_salt.hex()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chain",
        choices=sorted(CHAINS.keys()),
        nargs="?",
        help="Chain to deploy to.  Omit to deploy to every chain in CHAINS.",
    )
    args = parser.parse_args()
    targets = [args.chain] if args.chain else sorted(CHAINS.keys())
    for c in targets:
        main(c)
