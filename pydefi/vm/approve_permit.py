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
    "function execute(bytes program, ApproveProxyDeposit[] deposits) payable",
    "function vm() view returns (address)",
]
ApproveProxy = Contract.from_abi(_APPROVE_PROXY_ABI)

_PERMIT2_ABI = (
    Permit2PermitSingle.human_readable_abi()
    + Permit2AllowanceTransferDetail.human_readable_abi()
    + [
        "function approve(address token, address spender, uint160 amount, uint48 expiration)",
        "function allowance(address user, address token, address spender) view returns (uint160 amount, uint48 expiration, uint48 nonce)",
        "function permit(address owner, Permit2PermitSingle permitSingle, bytes signature)",
        "function transferFrom(address from, address to, uint160 amount, address token)",
        "function transferFrom(Permit2AllowanceTransferDetail[] transferDetails)",
    ]
)
Permit2 = Contract.from_abi(_PERMIT2_ABI)
