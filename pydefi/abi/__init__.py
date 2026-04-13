"""
pydefi.abi — centralised ABI definitions for all supported DeFi protocols.

Import Contract objects and ABIStruct classes from the sub-modules::

    from pydefi.abi.amm import UNISWAP_V2_ROUTER, UNISWAP_V3_ROUTER
    from pydefi.abi.bridge import CCTP_TOKEN_MESSENGER_V2, LAYERZERO_OFT
    from pydefi.abi.approve_permit import APPROVE_PROXY, PERMIT2
"""

from pydefi.abi.amm import (
    CURVE_POOL,
    CURVE_REGISTRY,
    UNISWAP_V2_FACTORY,
    UNISWAP_V2_PAIR,
    UNISWAP_V2_ROUTER,
    UNISWAP_V3_FACTORY,
    UNISWAP_V3_POOL,
    UNISWAP_V3_QUOTER_V2,
    UNISWAP_V3_ROUTER,
    ExactInputParams,
    ExactInputSingleParams,
    ExactOutputSingleParams,
    QuoteExactInputSingleParams,
    QuoteExactOutputSingleParams,
)
from pydefi.abi.approve_permit import (
    APPROVE_PROXY,
    PERMIT2,
    ApproveProxyDeposit,
    Permit2PermitDetails,
    Permit2PermitRequest,
    Permit2PermitSingle,
    Permit2PermitTransferFrom,
    Permit2PermitTransferFromRequest,
    Permit2SignatureTransferDetails,
    Permit2TokenPermissions,
)
from pydefi.abi.bridge import (
    ACROSS_SPOKE_POOL,
    CCTP_TOKEN_MESSENGER_V2,
    GASZIP,
    LAYERZERO_OFT,
    MAYAN_FORWARDER,
    MAYAN_SWIFT_V2,
    STARGATE_FACTORY,
    STARGATE_POOL,
    STARGATE_ROUTER,
    MayanSwiftOrderParams,
    MessagingFee,
    OFTSendParam,
)

__all__ = [
    # approve_permit
    "APPROVE_PROXY",
    "ApproveProxyDeposit",
    "PERMIT2",
    "Permit2PermitDetails",
    "Permit2PermitRequest",
    "Permit2PermitSingle",
    "Permit2PermitTransferFrom",
    "Permit2PermitTransferFromRequest",
    "Permit2SignatureTransferDetails",
    "Permit2TokenPermissions",
    # amm
    "CURVE_POOL",
    "CURVE_REGISTRY",
    "UNISWAP_V2_FACTORY",
    "UNISWAP_V2_PAIR",
    "UNISWAP_V2_ROUTER",
    "UNISWAP_V3_FACTORY",
    "UNISWAP_V3_POOL",
    "UNISWAP_V3_QUOTER_V2",
    "UNISWAP_V3_ROUTER",
    "ExactInputParams",
    "ExactInputSingleParams",
    "ExactOutputSingleParams",
    "QuoteExactInputSingleParams",
    "QuoteExactOutputSingleParams",
    # bridge
    "ACROSS_SPOKE_POOL",
    "CCTP_TOKEN_MESSENGER_V2",
    "GASZIP",
    "LAYERZERO_OFT",
    "MAYAN_FORWARDER",
    "MAYAN_SWIFT_V2",
    "STARGATE_FACTORY",
    "STARGATE_POOL",
    "STARGATE_ROUTER",
    "MayanSwiftOrderParams",
    "MessagingFee",
    "OFTSendParam",
]
