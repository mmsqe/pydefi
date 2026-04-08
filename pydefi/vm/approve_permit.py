"""
Input primitives for ApproveProxy and Permit2 flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from eth_contract import ABIStruct, Contract


class ApproveProxyDeposit(ABIStruct):
    """Struct for ``ApproveProxy.execute`` deposits."""

    token: Annotated[str, "address"]
    amount: Annotated[int, "uint256"]


class Permit2PermitDetails(ABIStruct):
    """Struct for Permit2 permit details."""

    token: Annotated[str, "address"]
    amount: Annotated[int, "uint160"]
    expiration: Annotated[int, "uint48"]
    nonce: Annotated[int, "uint48"]


class Permit2PermitSingle(ABIStruct):
    """Struct for Permit2 PermitSingle."""

    details: Permit2PermitDetails
    spender: Annotated[str, "address"]
    sigDeadline: Annotated[int, "uint256"]


class Permit2AllowanceTransferDetail(ABIStruct):
    """Struct for Permit2 batched transfer details."""

    from_addr: Annotated[str, "address"]
    to: Annotated[str, "address"]
    amount: Annotated[int, "uint160"]
    token: Annotated[str, "address"]


@dataclass(frozen=True)
class Permit2PermitRequest:
    """Input bundle for Permit2 ``permit`` call."""

    owner: str
    permit_single: Permit2PermitSingle
    signature: bytes | str


_APPROVE_PROXY_ABI = ApproveProxyDeposit.human_readable_abi() + [
    "function execute(bytes program, ApproveProxyDeposit[] deposits)",
]
_APPROVE_PROXY_TEMPLATE = Contract.from_abi(_APPROVE_PROXY_ABI)

_PERMIT2_ABI = (
    Permit2PermitSingle.human_readable_abi()
    + Permit2AllowanceTransferDetail.human_readable_abi()
    + [
        "function permit(address owner, Permit2PermitSingle permitSingle, bytes signature)",
        "function transferFrom(address from, address to, uint160 amount, address token)",
        "function transferFrom(Permit2AllowanceTransferDetail[] transferDetails)",
    ]
)
_PERMIT2_TEMPLATE = Contract.from_abi(_PERMIT2_ABI)


def _normalise_permit2_signature(signature: bytes | str) -> bytes:
    """Normalise a Permit2 signature to raw bytes."""
    if isinstance(signature, bytes):
        return signature
    if isinstance(signature, str):
        return bytes.fromhex(signature.removeprefix("0x"))
    raise TypeError(f"signature must be bytes or hex str, got {type(signature).__name__!r}")


def build_approve_proxy_execute_calldata(
    vm_program: bytes,
    deposits: list[ApproveProxyDeposit],
) -> bytes:
    """Build calldata for ``ApproveProxy.execute(program, deposits)``."""
    merged: dict[str, int] = {}
    for token, amount in deposits:
        merged[token] = merged.get(token, 0) + amount
    compact = [ApproveProxyDeposit(token=token, amount=amount) for token, amount in merged.items() if amount > 0]
    return _APPROVE_PROXY_TEMPLATE.fns.execute(vm_program, compact).data


def build_permit2_permit_calldata(
    owner: str,
    permit_single: Permit2PermitSingle,
    signature: bytes | str,
) -> bytes:
    """Build calldata for Permit2 ``permit(owner, permitSingle, signature)``."""
    sig_bytes = _normalise_permit2_signature(signature)
    return _PERMIT2_TEMPLATE.fns.permit(owner, permit_single, sig_bytes).data


def build_permit2_transfer_from_calldata(
    from_addr: str,
    to_addr: str,
    amount: int,
    token: str,
) -> bytes:
    """Build calldata for Permit2 ``transferFrom(from, to, amount, token)``."""
    return _PERMIT2_TEMPLATE.fns.transferFrom(from_addr, to_addr, amount, token).data
