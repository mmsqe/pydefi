"""Shared ERC-20 ABI selectors and encoders for the Venom peephole tooling.

Centralises the 4-byte function selectors and the address/word encoders that
were otherwise duplicated across the demos, the gas benchmark, and the
differential / parity tests.
"""

from __future__ import annotations

# 4-byte function selectors (keccak256(signature)[:4]).
MINT = bytes.fromhex("40c10f19")  # mint(address,uint256)
BURN = bytes.fromhex("42966c68")  # burn(uint256)
TRANSFER = bytes.fromhex("a9059cbb")  # transfer(address,uint256)
TRANSFER_FROM = bytes.fromhex("23b872dd")  # transferFrom(address,address,uint256)
APPROVE = bytes.fromhex("095ea7b3")  # approve(address,uint256)
BALANCE_OF = bytes.fromhex("70a08231")  # balanceOf(address)
ALLOWANCE = bytes.fromhex("dd62ed3e")  # allowance(address,address)
TOTAL_SUPPLY = bytes.fromhex("18160ddd")  # totalSupply()
OWNER = bytes.fromhex("8da5cb5b")  # owner()


def word(x: int) -> bytes:
    """32-byte big-endian encoding of a uint256."""
    return int(x).to_bytes(32, "big")


def selector(sig: str) -> bytes:
    """4-byte function selector for an ABI signature, e.g. ``transfer(address,uint256)``."""
    from vyper.utils import method_id

    return method_id(sig)


def addr20(x: int | bytes) -> bytes:
    """20-byte address, from an int or an already-20-byte value."""
    if isinstance(x, (bytes, bytearray)):
        b = bytes(x)
        if len(b) != 20:
            raise ValueError(f"expected a 20-byte address, got {len(b)} bytes")
        return b
    return int(x).to_bytes(20, "big")


def arg_addr(x: int | bytes) -> bytes:
    """32-byte left-padded ABI encoding of an address argument."""
    return b"\x00" * 12 + addr20(x)


def addr_str(x: int | bytes) -> str:
    """`0x`-prefixed hex address string (for titanoboa contract calls)."""
    return "0x" + addr20(x).hex()
