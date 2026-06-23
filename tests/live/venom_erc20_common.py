"""Shared helpers for the Venom balance-slot peephole ERC-20 testnet tests.

Both the anvil-fork-of-Sepolia test (:mod:`tests.live.test_venom_erc20_fork`)
and the real Sepolia broadcast test
(:mod:`tests.live.test_venom_erc20_sepolia_execution`) deploy the evm-smith
Venom-compiled Snekmate ERC-20 (vendored at :mod:`pydefi.venom`) in two
variants — the original keccak-slot runtime and the
:func:`pydefi.venom.balance_patch.patch_creation`-optimized NOT-slot runtime —
then exercise mint / transfer / balanceOf to confirm behaviour parity and the
``gasUsed`` delta.

Unlike the local scripts (``scripts/try_venom_erc20*.py``) which ``etch`` the
patched runtime — a state override only possible in a simulated EVM — these
deploy through ``patch_creation`` init code, so the *same* deploy path works on
an anvil fork and on a real chain. The constructor runs in both cases, so the
deployer is the owner/minter and ``mint`` succeeds.

:class:`VenomErc20` abstracts over the two send modes: an anvil unlocked dev
account (``private_key=None`` → ``eth_sendTransaction``) and a real signer
(``private_key`` set → locally-signed raw tx).
"""

from __future__ import annotations

from dataclasses import dataclass

from eth_account import Account
from web3 import AsyncWeb3, Web3

from pydefi.venom import balance_patch
from pydefi.venom.erc20_abi import (  # selectors/encoders, re-exported for the live tests
    BALANCE_OF,
    MINT,
    OWNER,
    TOTAL_SUPPLY,
    TRANSFER,
    addr20,
    word,
)

low_addr = addr20  # re-exported alias used by the live tests

ONE = 10**18

#: Vyper named-slot collision probes — owner@1, balanceOf-base@2, totalSupply@4.
#: A raw address-as-slot scheme would corrupt these; NOT(addr) lands every
#: balance above 2**160, so they must survive a mint to these low addresses.
ALIAS_PROBES = (0x01, 0x02, 0x04)


def _addr_bytes(a: str | bytes) -> bytes:
    if isinstance(a, str):
        a = bytes.fromhex(a[2:] if a.startswith("0x") else a)
    return a[-20:].rjust(20, b"\x00")


def abi_addr(a: str | bytes) -> bytes:
    """20-byte address (hex str or bytes) -> 32-byte left-padded ABI word."""
    return b"\x00" * 12 + _addr_bytes(a)


def creation_pair() -> tuple[bytes, bytes]:
    """(original_creation, patched_creation) from the vendored artifact.

    ``patched_creation`` deploys to a runtime with every balance keccak site
    rewritten to ``NOT(addr)``; the peephole is length-preserving so the
    constructor's ``CODECOPY``/``RETURN`` offsets stay valid.
    """
    creation = balance_patch.creation_from_artifact()
    runtime = balance_patch.runtime_from_artifact()
    return creation, balance_patch.patch_creation(creation, runtime)


def expected_runtime_sites() -> int:
    """Balance keccak sites in the unpatched runtime (the original deploys to
    this many; the patched deploys to 0)."""
    return balance_patch.count_sites(balance_patch.runtime_from_artifact())


async def _send(
    w3: AsyncWeb3,
    *,
    sender: str,
    private_key: str | None,
    to: str | None,
    data: bytes,
    value: int = 0,
) -> dict:
    """Build, send (signed or unlocked), and await the receipt of one tx.

    ``to=None`` is a contract creation. With ``private_key`` the tx is signed
    locally (real chain); without it the node signs for an unlocked account
    (anvil dev account).
    """
    tx: dict = {"from": Web3.to_checksum_address(sender), "value": value, "data": "0x" + data.hex()}
    if to is not None:
        tx["to"] = Web3.to_checksum_address(to)

    if private_key is None:
        tx_hash = await w3.eth.send_transaction(tx)
    else:
        estimate = await w3.eth.estimate_gas(tx)
        signed_tx = {
            "value": value,
            "data": "0x" + data.hex(),
            "nonce": await w3.eth.get_transaction_count(tx["from"], "pending"),
            "chainId": await w3.eth.chain_id,
            "gas": int(estimate * 12 // 10),
            "gasPrice": int(await w3.eth.gas_price * 12 // 10),
        }
        if to is not None:
            signed_tx["to"] = tx["to"]  # omitted entirely for creation
        signed = Account.sign_transaction(signed_tx, private_key)
        tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)

    # Testnet inclusion latency can exceed web3's 120s default under load.
    return await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=0.1)


async def eth_view(w3: AsyncWeb3, to: str, data: bytes) -> bytes:
    """``eth_call`` a view selector and return raw return bytes (no gas)."""
    return bytes(await w3.eth.call({"to": Web3.to_checksum_address(to), "data": "0x" + data.hex()}))


@dataclass
class VenomErc20:
    """A deployed instance, bound to its (w3, sender, signing mode)."""

    w3: AsyncWeb3
    address: str          # checksummed
    sender: str
    private_key: str | None
    deploy_gas: int

    @classmethod
    async def deploy(
        cls, w3: AsyncWeb3, sender: str, private_key: str | None, creation: bytes
    ) -> "VenomErc20":
        receipt = await _send(w3, sender=sender, private_key=private_key, to=None, data=creation)
        assert int(receipt["status"]) == 1, "deploy reverted"
        return cls(
            w3=w3,
            address=Web3.to_checksum_address(receipt["contractAddress"]),
            sender=sender,
            private_key=private_key,
            deploy_gas=int(receipt["gasUsed"]),
        )

    async def _tx(self, data: bytes) -> int:
        receipt = await _send(
            self.w3, sender=self.sender, private_key=self.private_key, to=self.address, data=data
        )
        assert int(receipt["status"]) == 1, f"{data[:4].hex()} reverted"
        return int(receipt["gasUsed"])

    async def mint(self, to: str | bytes, amount: int) -> int:
        return await self._tx(MINT + abi_addr(to) + word(amount))

    async def transfer(self, to: str | bytes, amount: int) -> int:
        return await self._tx(TRANSFER + abi_addr(to) + word(amount))

    async def balance_of(self, who: str | bytes) -> int:
        return int.from_bytes(await eth_view(self.w3, self.address, BALANCE_OF + abi_addr(who)), "big")

    async def total_supply(self) -> int:
        return int.from_bytes(await eth_view(self.w3, self.address, TOTAL_SUPPLY), "big")

    async def owner(self) -> bytes:
        return await eth_view(self.w3, self.address, OWNER)

    async def code(self) -> bytes:
        return bytes(await self.w3.eth.get_code(self.address))

    async def balance_sites(self) -> int:
        """Remaining balance keccak sites in the *deployed* runtime."""
        return balance_patch.count_sites(await self.code())
