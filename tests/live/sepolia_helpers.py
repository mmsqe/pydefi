"""Shared Sepolia constants and helpers for live integration tests."""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import pytest
from eth_abi import encode as abi_encode
from eth_utils.crypto import keccak
from web3 import AsyncWeb3, Web3

from pydefi.abi.amm import UNISWAP_V4_STATE_VIEW
from pydefi.deployments import get_address, get_token
from pydefi.pathfinder.graph import V4PoolEdge
from pydefi.types import Address, ChainId

_SEP = ChainId.SEPOLIA
SEPOLIA_CHAIN_ID = int(_SEP)  # 11155111
ARBITRUM_SEPOLIA_CHAIN_ID = 421614

# ---------------------------------------------------------------------------
# Sepolia token addresses (checksum form for Web3 calls)
# ---------------------------------------------------------------------------

WETH_SEPOLIA = Web3.to_checksum_address(get_address("WETH", _SEP))
USDC_SEPOLIA = Web3.to_checksum_address(get_address("USDC", _SEP))
UNI_SEPOLIA = Web3.to_checksum_address(get_address("UNI", _SEP))

# ---------------------------------------------------------------------------
# Sepolia token objects
# ---------------------------------------------------------------------------

WETH_SEP = get_token("WETH", _SEP)
USDC_SEP = get_token("USDC", _SEP)
UNI_SEP = get_token("UNI", _SEP)

# ---------------------------------------------------------------------------
# Uniswap V4 singleton contracts on Sepolia (PoolManager + StateView reader)
# ---------------------------------------------------------------------------

V4_POOL_MANAGER_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V4_POOL_MANAGER", _SEP))
V4_STATE_VIEW_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V4_STATE_VIEW", _SEP))
ZERO_ADDR = Web3.to_checksum_address("0x" + "0" * 40)

# (fee_tier, tick_spacing) pairs to probe when discovering Sepolia V4 pools.
# (500, 20) is listed first — it is the pool seeded at market price.  Standard
# tiers are checked last; they are initialised on Sepolia at a distorted price.
SEPOLIA_V4_FEE_TICK_SPACINGS: list[tuple[int, int]] = [
    (500, 20),
    (100, 1),
    (500, 10),
    (3000, 60),
    (10000, 200),
]

# ---------------------------------------------------------------------------
# Uniswap V2 contracts on Sepolia
# ---------------------------------------------------------------------------

V2_ROUTER_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V2_ROUTER", _SEP))
V2_FACTORY_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V2_FACTORY", _SEP))
#: There is only one V2 factory on Sepolia; alias kept for compatibility.
V2_FACTORY_ALT_SEPOLIA = V2_FACTORY_SEPOLIA

# ---------------------------------------------------------------------------
# Uniswap V3 contracts on Sepolia
# ---------------------------------------------------------------------------

V3_ROUTER_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V3_ROUTER", _SEP))
V3_QUOTER_SEPOLIA: Address = Address(get_address("UNISWAP_V3_QUOTER", _SEP))
V3_FACTORY_SEPOLIA = Web3.to_checksum_address(get_address("UNISWAP_V3_FACTORY", _SEP))


def require_env(*names: str) -> None:
    """Skip the current test if any of the named environment variables is unset.

    Call at the top of a test function in place of repeated ``if not VAR: pytest.skip(...)`` blocks::

        require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")
    """
    for name in names:
        if not os.getenv(name, "").strip():
            pytest.skip(f"{name} not set")


async def connect(rpc_url: str, *, expected_chain_id: int = SEPOLIA_CHAIN_ID) -> AsyncWeb3:
    """Build an ``AsyncWeb3`` for *rpc_url*, failing unless it reports *expected_chain_id*.

    Replaces the per-test ``AsyncWeb3(AsyncHTTPProvider(...))`` + ``chain_id`` guard.
    Defaults to Sepolia; pass :data:`ARBITRUM_SEPOLIA_CHAIN_ID` for the destination lane.
    """
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
    actual = int(await w3.eth.chain_id)
    if actual != expected_chain_id:
        pytest.fail(f"expected chain_id={expected_chain_id}, got {actual}")
    return w3


async def discover_sepolia_v4_weth_usdc_edge(w3: AsyncWeb3) -> V4PoolEdge | None:
    """Return the first initialised Sepolia WETH/USDC V4 pool edge, or None.

    Probes fee/tick-spacing pairs in :data:`SEPOLIA_V4_FEE_TICK_SPACINGS` order
    (seeded market-price pool first) and returns the first pool with non-zero
    ``sqrtPriceX96`` and ``liquidity``.
    """
    state_view = UNISWAP_V4_STATE_VIEW(to=V4_STATE_VIEW_SEPOLIA)
    c0, c1 = (
        (WETH_SEPOLIA, USDC_SEPOLIA) if WETH_SEPOLIA.lower() < USDC_SEPOLIA.lower() else (USDC_SEPOLIA, WETH_SEPOLIA)
    )
    is_weth_c0 = c0.lower() == WETH_SEPOLIA.lower()

    for fee_tier, tick_spacing in SEPOLIA_V4_FEE_TICK_SPACINGS:
        pool_id_bytes = keccak(
            abi_encode(
                ["address", "address", "uint24", "int24", "address"],
                [c0, c1, fee_tier, tick_spacing, ZERO_ADDR],
            )
        )
        try:
            slot0 = await state_view.fns.getSlot0(pool_id_bytes).call(w3)
            liquidity = await state_view.fns.getLiquidity(pool_id_bytes).call(w3)
        except Exception:
            continue
        if slot0[0] == 0 or liquidity == 0:
            continue
        return V4PoolEdge(
            token_in=WETH_SEP,
            token_out=USDC_SEP,
            # Address (bytes), not the checksum str: the DeFiVM V4 lowering passes
            # pool_address straight to call_raw, which only accepts bytes/int/Operand.
            pool_address=Address(V4_POOL_MANAGER_SEPOLIA),
            protocol="UniswapV4",
            fee_bps=fee_tier // 100,
            sqrt_price_x96=slot0[0],
            liquidity=liquidity,
            is_token0_in=is_weth_c0,
            tick_spacing=tick_spacing,
            hooks=ZERO_ADDR,
            pool_id="0x" + pool_id_bytes.hex(),
        )
    return None


# ---------------------------------------------------------------------------
# RPC retry helpers — testnets routinely 429 under load, so wrap rate-limited
# calls with exponential backoff.
# ---------------------------------------------------------------------------

_RPC_RETRY_MAX = 5
_RPC_RETRY_BASE = 2.0  # seconds; actual delay = base * 2^attempt + jitter

T = TypeVar("T")


def is_rate_limit(exc: BaseException) -> bool:
    """Return True when *exc* looks like an HTTP 429 / rate-limit error."""
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return True
    if hasattr(exc, "status") and getattr(exc, "status") == 429:
        return True
    return False


async def retry_rpc(
    coro_fn: Callable[..., Awaitable[T]],
    /,
    *args: Any,
    label: str = "",
    **kwargs: Any,
) -> T:
    """Call ``await coro_fn(*args, **kwargs)`` with exponential-backoff retry on 429."""
    for attempt in range(_RPC_RETRY_MAX):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            if is_rate_limit(exc) and attempt < _RPC_RETRY_MAX - 1:
                delay = _RPC_RETRY_BASE * (2**attempt) + random.uniform(0, 1)
                tag = f" [{label}]" if label else ""
                print(f"[retry{tag}] 429 — attempt {attempt + 1}/{_RPC_RETRY_MAX}, sleeping {delay:.1f}s", flush=True)
                await asyncio.sleep(delay)
            else:
                raise
    raise RuntimeError("retry_rpc: exhausted retries without raising")  # unreachable
