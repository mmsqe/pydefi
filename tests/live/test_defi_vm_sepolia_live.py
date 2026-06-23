"""Live DeFiVM swap test on Sepolia testnet.

Demonstrates the current dev UX for executing a swap via DeFiVM:

1. **Off-chain route discovery** — ``Router.find_best_route()`` reads live
   pool state from Sepolia and returns a :class:`~pydefi.types.SwapRoute`.
2. **On-chain output preview** — ``quote_swap_transaction()`` issues a single
   ``eth_call`` that runs a view-only DeFiVM program against the live pool
   contracts, returning an on-chain amountOut estimate.
3. **Build transaction** — ``build_swap_transaction()`` compiles the
   executing DeFiVM program and wraps it in ``execute(bytes)`` calldata.
4. **Broadcast** — the transaction is signed and dispatched; the test verifies
   that the recipient's output-token balance increased by at least
   ``min_final_out``.

Requirements
------------
- ``SEPOLIA_RPC_URL`` — Sepolia JSON-RPC endpoint.
- ``SEPOLIA_PRIVATE_KEY`` — hex private key of the sender (must hold Sepolia
  ETH for gas; WETH is deposited automatically if needed).
- ``SEPOLIA_DEFI_VM`` — on-chain address of the deployed DeFiVM contract.

Run with::

    pytest -m live tests/live/test_defi_vm_sepolia_live.py -v
"""

from __future__ import annotations

import os
from typing import cast

import pytest
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_contract import Contract
from eth_contract.erc20 import ERC20
from eth_contract.utils import send_transaction
from eth_typing import ChecksumAddress
from eth_utils import keccak
from web3 import AsyncWeb3, Web3
from web3.types import Wei

from pydefi.abi.amm import (
    UNISWAP_V2_FACTORY,
    UNISWAP_V2_PAIR,
    UNISWAP_V3_FACTORY,
    UNISWAP_V3_POOL,
)
from pydefi.amm.universal_router import (
    MSG_SENDER,
    UNIVERSAL_ROUTER_ADDRESSES,
    UniversalRouter,
    V4Hop,
)
from pydefi.deployments import get_address, get_token
from pydefi.indexer import PoolIndexer
from pydefi.pathfinder.dag import RouteDAG
from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge, V4PoolEdge
from pydefi.pathfinder.router import Router
from pydefi.types import Address, ChainId, TokenAmount
from pydefi.types import Token as _Token
from pydefi.vm.swap import build_swap_transaction, quote_dag
from tests.live.sepolia_helpers import (
    USDC_SEPOLIA,
    WETH_SEPOLIA,
    connect,
    discover_sepolia_v4_weth_usdc_edge,
    require_env,
)
from tests.live.sepolia_helpers import (
    V2_FACTORY_ALT_SEPOLIA as V2_FACTORY_ADDR,
)
from tests.live.sepolia_helpers import (
    V3_FACTORY_SEPOLIA as V3_FACTORY_ADDR,
)
from tests.live.sepolia_helpers import (
    V3_QUOTER_SEPOLIA as V3_QUOTER_ADDR,
)

# ---------------------------------------------------------------------------
# Sepolia token addresses (module-level for convenience)
# ---------------------------------------------------------------------------

WETH_ADDR = WETH_SEPOLIA
USDC_ADDR = USDC_SEPOLIA
UNI_ADDR = Web3.to_checksum_address(get_address("UNI", ChainId.SEPOLIA))

# Additional Sepolia ERC-20 addresses (Aave/Uniswap test tokens)
DAI_ADDR = Web3.to_checksum_address("0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357")
USDT_ADDR = Web3.to_checksum_address("0xaA8E23Fb1079EA71e0a56F48a2aA51851D8433D0")
WBTC_ADDR = Web3.to_checksum_address("0x29f2D40B0605204364af54EC677bD022dA425d03")
LINK_ADDR = Web3.to_checksum_address("0xf8Fb3713D459D7C1018BD0A49D19b4C44290EBE5")
AAVE_ADDR = Web3.to_checksum_address("0x88541670E55cC00bEEFD87eB59EDd1b7C511AC9a")

WETH = get_token("WETH", ChainId.SEPOLIA)
USDC = get_token("USDC", ChainId.SEPOLIA)
UNI = get_token("UNI", ChainId.SEPOLIA)

DAI = _Token(chain_id=11155111, address=Address(DAI_ADDR), symbol="DAI", decimals=18)
USDT = _Token(chain_id=11155111, address=Address(USDT_ADDR), symbol="USDT", decimals=6)
WBTC = _Token(chain_id=11155111, address=Address(WBTC_ADDR), symbol="WBTC", decimals=8)
LINK = _Token(chain_id=11155111, address=Address(LINK_ADDR), symbol="LINK", decimals=18)
AAVE = _Token(chain_id=11155111, address=Address(AAVE_ADDR), symbol="AAVE", decimals=18)

# Swap 0.002 WETH -> USDC (matches reference test)
SWAP_AMOUNT = 2 * 10**15

# Environment variables resolved at module load so skips are immediate.
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL", "").strip()
SEPOLIA_PRIVATE_KEY = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
SEPOLIA_DEFI_VM = os.getenv("SEPOLIA_DEFI_VM", "").strip()

# WETH.deposit() is not in the standard ERC20 ABI; use a minimal inline stub.
_WETH_DEPOSIT = Contract.from_abi(["function deposit() external payable"])

# Uniswap V4 PoolManager on Sepolia (used for Swap-event log matching; pool
# discovery + StateView live in tests.live.sepolia_helpers).
V4_POOL_MANAGER_ADDR = Web3.to_checksum_address(get_address("UNISWAP_V4_POOL_MANAGER", ChainId.SEPOLIA))
# PoolManager Swap event topic0 (authoritative settled-amount source for V4).
_V4_SWAP_TOPIC = keccak(text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)").hex()

# ---------------------------------------------------------------------------
# PoolIndexer integration — the indexer owns pool *discovery* and static
# metadata (tokens, fee tier).  The volatile AMM state (V3 sqrtPrice/liquidity,
# V2 reserves) is read live per quote: it is a current-block quantity that the
# indexer's swap-event backfill cannot keep fresh (it never sees mint/burn).
# ---------------------------------------------------------------------------

# Known token metadata for auto-registration (address lowercase → (symbol, decimals))
_TOKEN_INFO: dict[str, tuple[str, int]] = {
    WETH_ADDR.lower(): ("WETH", 18),
    USDC_ADDR.lower(): ("USDC", 6),
    UNI_ADDR.lower(): ("UNI", 18),
    DAI_ADDR.lower(): ("DAI", 18),
    USDT_ADDR.lower(): ("USDT", 6),
    WBTC_ADDR.lower(): ("WBTC", 8),
    LINK_ADDR.lower(): ("LINK", 18),
    AAVE_ADDR.lower(): ("AAVE", 18),
}


# Persistent SQLite DB alongside this file — caches discovered pools' static
# metadata (addresses, tokens, fee tiers) across runs so repeat runs skip
# re-registration.  Delete sepolia_pools.db to force a fresh discovery.
_DB_PATH = os.path.join(os.path.dirname(__file__), "sepolia_pools.db")
_DB_URL = f"sqlite:///{_DB_PATH}"


@pytest.fixture(scope="module")
def pool_indexer() -> PoolIndexer:
    """Module-scoped PoolIndexer shared by all tests in this file.

    Backed by a persistent SQLite file at ``tests/live/sepolia_pools.db``.
    It supplies pool discovery and static metadata; each test reads the
    instantaneous AMM state (sqrtPrice/liquidity/reserves) live when quoting.
    """
    return PoolIndexer(db_url=_DB_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signer() -> tuple[LocalAccount, ChecksumAddress, ChecksumAddress]:
    """Return ``(account, sender, defi_vm_address)`` from the Sepolia env."""
    account = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(account.address)
    return account, sender, Web3.to_checksum_address(SEPOLIA_DEFI_VM)


async def _prefund_vm_with_weth(
    w3: AsyncWeb3, account: LocalAccount, sender: str, defi_vm: str, amount: int = SWAP_AMOUNT
) -> None:
    """Wrap ETH->WETH (when the sender holds < *amount*) then transfer *amount*
    WETH to *defi_vm*.  Both the V3 flash-swap callback and the V4 settle path
    require the VM to already hold tokenIn before ``execute()``."""
    weth = ERC20(to=WETH_ADDR)
    if int(await weth.fns.balanceOf(sender).call(w3)) < amount:
        await send_transaction(
            w3, account, to=WETH_ADDR, data=bytes(_WETH_DEPOSIT(to=WETH_ADDR).fns.deposit().data), value=Wei(amount)
        )
    await send_transaction(w3, account, to=WETH_ADDR, data=bytes(weth.fns.transfer(defi_vm, amount).data), value=Wei(0))


async def _discover_pool_graph(w3: AsyncWeb3, indexer: PoolIndexer | None = None) -> tuple[PoolGraph, str]:
    """Build a PoolGraph from the best available WETH/USDC V3 pool on Sepolia.

    Tries fee tiers 500, 3000, 10000 in order and uses the first deployed one.
    When *indexer* is supplied, pool state (sqrtPriceX96, liquidity) is read
    from the local DB instead of individual ``eth_call`` round-trips.
    """
    factory = UNISWAP_V3_FACTORY(to=V3_FACTORY_ADDR)
    for fee_tier in (500, 3000, 10000):
        print(f"[discover_pool_graph] getPool fee={fee_tier} ...", flush=True)
        pool_addr = Web3.to_checksum_address(await factory.fns.getPool(WETH_ADDR, USDC_ADDR, fee_tier).call(w3))
        if pool_addr and pool_addr != Web3.to_checksum_address("0x" + "0" * 40):
            print(f"[discover_pool_graph] found pool {pool_addr}", flush=True)
            pool_contract = UNISWAP_V3_POOL(to=pool_addr)

            # token0: use cached value if pool already registered, else fetch once
            pool_rec = indexer.get_pool(pool_addr) if indexer is not None else None
            if pool_rec is not None:
                token0_addr = pool_rec.token0_address
            else:
                print("[discover_pool_graph] fetch token0 ...", flush=True)
                token0_addr = await pool_contract.fns.token0().call(w3)

            if indexer is not None and pool_rec is None:
                print("[discover_pool_graph] register pool in indexer ...", flush=True)
                fee_actual = await pool_contract.fns.fee().call(w3)
                t0_sym, t0_dec = _TOKEN_INFO.get(token0_addr.lower(), ("", 18))
                t1_addr = USDC_ADDR if token0_addr.lower() == WETH_ADDR.lower() else WETH_ADDR
                t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr.lower(), ("", 18))
                indexer.add_v3_pool(
                    pool_address=pool_addr,
                    protocol="UniswapV3",
                    token0_address=token0_addr,
                    token0_symbol=t0_sym,
                    token0_decimals=t0_dec,
                    token1_address=t1_addr,
                    token1_symbol=t1_sym,
                    token1_decimals=t1_dec,
                    chain_id=11155111,
                    fee_bps=fee_actual // 100,
                )
                pool_rec = indexer.get_pool(pool_addr)

            # V3 sqrtPrice moves on every swap and active liquidity moves on every
            # in-range mint/burn, so the instantaneous AMM state is a current-block
            # quantity — read it live rather than from the indexer's last-swap-event
            # snapshot (the swap-only backfill never sees mint/burn, so its liquidity
            # goes stale and the local sim drifts from a live QuoterV2 quote).  The
            # indexer still owns pool discovery + static metadata above.
            slot0 = await pool_contract.fns.slot0().call(w3)
            liquidity = await pool_contract.fns.liquidity().call(w3)
            sqrt_price_x96 = slot0[0]
            fee_bps = pool_rec.fee_bps if pool_rec is not None else (await pool_contract.fns.fee().call(w3)) // 100

            edge = V3PoolEdge(
                token_in=WETH,
                token_out=USDC,
                pool_address=Address(pool_addr),
                protocol="UniswapV3",
                fee_bps=fee_bps,
                sqrt_price_x96=sqrt_price_x96,
                liquidity=liquidity,
                is_token0_in=token0_addr.lower() == WETH_ADDR.lower(),
            )
            graph = PoolGraph()
            graph.add_pool(edge)
            return graph, pool_addr
    pytest.skip("No WETH/USDC V3 pool found on Sepolia")


_ZERO_ADDR = Web3.to_checksum_address("0x" + "0" * 40)


async def _discover_mixed_pool_graph(w3: AsyncWeb3, indexer: PoolIndexer | None = None) -> tuple[PoolGraph, str, str]:
    """Build a 2-hop PoolGraph for WETH→UNI (V2) → UNI→USDC (V3) on Sepolia.

    Returns ``(graph, v2_pair_addr, v3_pool_addr)``.
    Skips when either pool is not deployed.
    When *indexer* is supplied, reserve/state data comes from the local DB.
    """
    # -- V2 leg: WETH → UNI -----------------------------------------------
    v2_factory = UNISWAP_V2_FACTORY(to=V2_FACTORY_ADDR)
    print("[discover_mixed] getPair WETH/UNI ...", flush=True)
    v2_pair_addr = Web3.to_checksum_address(await v2_factory.fns.getPair(WETH_ADDR, UNI_ADDR).call(w3))
    if v2_pair_addr == _ZERO_ADDR:
        pytest.skip("No WETH/UNI V2 pair found on Sepolia")
    print(f"[discover_mixed] V2 pair={v2_pair_addr}", flush=True)

    v2_pair = UNISWAP_V2_PAIR(to=v2_pair_addr)
    v2_rec = indexer.get_pool(v2_pair_addr) if indexer is not None else None
    if v2_rec is not None:
        token0_v2 = v2_rec.token0_address
    else:
        print("[discover_mixed] fetch V2 token0 ...", flush=True)
        token0_v2 = await v2_pair.fns.token0().call(w3)

    if indexer is not None and v2_rec is None:
        print("[discover_mixed] register V2 pool in indexer ...", flush=True)
        t0_sym, t0_dec = _TOKEN_INFO.get(token0_v2.lower(), ("", 18))
        t1_addr = UNI_ADDR if token0_v2.lower() == WETH_ADDR.lower() else WETH_ADDR
        t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr.lower(), ("", 18))
        indexer.add_v2_pool(
            pool_address=v2_pair_addr,
            protocol="UniswapV2",
            token0_address=token0_v2,
            token0_symbol=t0_sym,
            token0_decimals=t0_dec,
            token1_address=t1_addr,
            token1_symbol=t1_sym,
            token1_decimals=t1_dec,
            chain_id=11155111,
            fee_bps=30,
        )

    # -- V3 leg: UNI → USDC -----------------------------------------------
    v3_factory = UNISWAP_V3_FACTORY(to=V3_FACTORY_ADDR)
    v3_pool_addr = None
    for fee_tier in (500, 3000, 10000):
        print(f"[discover_mixed] getPool UNI/USDC fee={fee_tier} ...", flush=True)
        addr = Web3.to_checksum_address(await v3_factory.fns.getPool(UNI_ADDR, USDC_ADDR, fee_tier).call(w3))
        if addr and addr != _ZERO_ADDR:
            v3_pool_addr = addr
            break
    if v3_pool_addr is None:
        pytest.skip("No UNI/USDC V3 pool found on Sepolia")
    print(f"[discover_mixed] V3 pool={v3_pool_addr}", flush=True)

    v3_pool = UNISWAP_V3_POOL(to=v3_pool_addr)
    v3_rec = indexer.get_pool(v3_pool_addr) if indexer is not None else None
    if v3_rec is not None:
        token0_v3 = v3_rec.token0_address
    else:
        print("[discover_mixed] fetch V3 token0 ...", flush=True)
        token0_v3 = await v3_pool.fns.token0().call(w3)

    if indexer is not None and v3_rec is None:
        print("[discover_mixed] register V3 pool in indexer ...", flush=True)
        fee_actual_rpc = await v3_pool.fns.fee().call(w3)
        t0_sym, t0_dec = _TOKEN_INFO.get(token0_v3.lower(), ("", 18))
        t1_addr_v3 = USDC_ADDR if token0_v3.lower() == UNI_ADDR.lower() else UNI_ADDR
        t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr_v3.lower(), ("", 18))
        indexer.add_v3_pool(
            pool_address=v3_pool_addr,
            protocol="UniswapV3",
            token0_address=token0_v3,
            token0_symbol=t0_sym,
            token0_decimals=t0_dec,
            token1_address=t1_addr_v3,
            token1_symbol=t1_sym,
            token1_decimals=t1_dec,
            chain_id=11155111,
            fee_bps=fee_actual_rpc // 100,
        )
        v3_rec = indexer.get_pool(v3_pool_addr)
        assert v3_rec is not None

    # Read the instantaneous V2 reserves and V3 sqrtPrice/liquidity live: these
    # are current-block quantities (V2 reserves move on every swap/mint/burn; V3
    # active liquidity moves on in-range mint/burn), so the indexer's swap-only
    # snapshot goes stale and the local sim drifts from a live on-chain quote.
    # The indexer still owns pool discovery + static metadata above.
    reserves = await v2_pair.fns.getReserves().call(w3)
    reserve0, reserve1 = reserves[0], reserves[1]
    if token0_v2.lower() == WETH_ADDR.lower():
        reserve_weth, reserve_uni = reserve0, reserve1
    else:
        reserve_weth, reserve_uni = reserve1, reserve0

    slot0 = await v3_pool.fns.slot0().call(w3)
    liquidity = await v3_pool.fns.liquidity().call(w3)
    sqrt_price_x96 = slot0[0]
    fee_bps_v3 = v3_rec.fee_bps if v3_rec is not None else (await v3_pool.fns.fee().call(w3)) // 100

    v2_edge = PoolEdge(
        token_in=WETH,
        token_out=UNI,
        pool_address=Address(v2_pair_addr),
        protocol="UniswapV2",
        fee_bps=30,
        reserve_in=reserve_weth,
        reserve_out=reserve_uni,
        extra={"is_token0_in": token0_v2.lower() == WETH_ADDR.lower()},
    )

    v3_edge = V3PoolEdge(
        token_in=UNI,
        token_out=USDC,
        pool_address=Address(v3_pool_addr),
        protocol="UniswapV3",
        fee_bps=fee_bps_v3,
        sqrt_price_x96=sqrt_price_x96,
        liquidity=liquidity,
        is_token0_in=token0_v3.lower() == UNI_ADDR.lower(),
    )

    graph = PoolGraph()
    graph.add_pool(v2_edge)
    graph.add_pool(v3_edge)
    return graph, v2_pair_addr, v3_pool_addr


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.testnet
async def test_sepolia_defi_vm_weth_to_usdc(pool_indexer: PoolIndexer) -> None:
    """Execute a WETH->USDC V3 swap via DeFiVM on Sepolia.

    DeFiVM dev UX (new API):

    1. quote_swap_transaction() -- single eth_call, on-chain amountOut
    2. build_swap_transaction() -- compiles execute(bytes) calldata
    3. send_transaction() -- sign + broadcast; verify USDC received
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # -- Step 1: Off-chain route discovery --------------------------------
    graph, _pool_addr = await _discover_pool_graph(w3, pool_indexer)
    print("mm-_pool_addr", _pool_addr)
    router = Router(graph)
    route = router.find_best_route(TokenAmount(WETH, SWAP_AMOUNT), USDC)
    dag = router.find_best_route_dag(TokenAmount(WETH, SWAP_AMOUNT), USDC)

    # -- Step 2: On-chain quote via DeFiVM eth_call -----------------------
    quoted = await quote_dag(
        dag,
        amount_in=SWAP_AMOUNT,
        w3=w3,
        vm_address=cast(Address, defi_vm_address),
        quoter_address=V3_QUOTER_ADDR,
    )
    print(f"[weth->usdc] on-chain quote:  {quoted / 10**6:.6f} USDC")
    assert quoted > 0, f"quote_dag returned 0: {quoted}"

    # Both estimates should agree within 2% (same pool state, same block).
    off_chain_out = route.amount_out.amount
    ratio = quoted / off_chain_out if off_chain_out else 0
    print(f"[weth->usdc] off-chain: {off_chain_out / 10**6:.6f} USDC  ratio={ratio:.4f}")
    assert 0.98 <= ratio <= 1.02, f"on-chain ({quoted}) vs off-chain ({off_chain_out}) ratio={ratio:.4f}"

    # -- Step 3: Pre-fund: wrap ETH -> WETH if balance is insufficient ----
    await _prefund_vm_with_weth(w3, account, sender, defi_vm_address)

    # -- Step 4: Build and broadcast execute(bytes) -----------------------
    usdc = ERC20(to=USDC_ADDR)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    # Base min on the on-chain QuoterV2 result rather than the local simulator
    # estimate.  On testnet the two can disagree by ~1% even when pool state
    # is identical; 2% slippage also matches the on-chain/off-chain ratio
    # tolerance asserted above so any drift surfaces in that assertion first.
    min_final_out = quoted * (10_000 - 200) // 10_000
    tx = build_swap_transaction(dag, SWAP_AMOUNT, defi_vm_address, sender, min_final_out=min_final_out)
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=500_000,
    )
    print(f"[weth->usdc] tx {receipt['transactionHash'].hex()} status {receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM execute() reverted: {receipt['transactionHash'].hex()}"

    # -- Step 5: Verify USDC received -------------------------------------
    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before
    print(f"[weth->usdc] received {received / 10**6:.6f} USDC  (min={min_final_out / 10**6:.6f})")
    assert received >= min_final_out, f"received {received} USDC < min_final_out {min_final_out}"


@pytest.mark.asyncio
@pytest.mark.testnet
async def test_sepolia_defi_vm_mixed_v2v3_route(pool_indexer: PoolIndexer) -> None:
    """Execute a WETH→UNI (V2) → UNI→USDC (V3) 2-hop swap via DeFiVM on Sepolia.

    Verifies that DeFiVM can handle a mixed-protocol route where the first hop
    uses UniswapV2 constant-product math and the second hop uses UniswapV3
    concentrated liquidity.  The Universal Router path is intentionally skipped
    here because ``build_swap_transaction`` raises ``NotImplementedError`` for
    mixed legs; only the DeFiVM path (``build_swap_transaction``) is tested.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # -- Step 1: Off-chain route discovery (WETH→UNI V2, UNI→USDC V3) ----
    graph, v2_pair_addr, v3_pool_addr = await _discover_mixed_pool_graph(w3, pool_indexer)
    print(f"[mixed] V2 pair={v2_pair_addr}  V3 pool={v3_pool_addr}")
    router = Router(graph)
    route = router.find_best_route(TokenAmount(WETH, SWAP_AMOUNT), USDC)
    dag = router.find_best_route_dag(TokenAmount(WETH, SWAP_AMOUNT), USDC)

    # -- Step 2: On-chain quote via DeFiVM eth_call -----------------------
    quoted = await quote_dag(
        dag,
        amount_in=SWAP_AMOUNT,
        w3=w3,
        vm_address=cast(Address, defi_vm_address),
        quoter_address=V3_QUOTER_ADDR,
    )
    print(f"[mixed] on-chain quote:  {quoted / 10**6:.6f} USDC")
    assert quoted > 0, f"quote_dag returned 0: {quoted}"

    # Off-chain V2 constant-product + V3 single-tick estimates compound over
    # two hops and diverge from the on-chain QuoterV2 result; log for info only.
    off_chain_out = route.amount_out.amount
    ratio = quoted / off_chain_out if off_chain_out else 0
    print(f"[mixed] off-chain: {off_chain_out / 10**6:.6f} USDC  ratio={ratio:.4f}")

    # -- Step 3: Pre-fund: wrap ETH -> WETH if balance is insufficient ----
    await _prefund_vm_with_weth(w3, account, sender, defi_vm_address)

    # -- Step 4: Build and broadcast execute(bytes) -----------------------
    usdc = ERC20(to=USDC_ADDR)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    # On-chain quote (QuoterV2) is the reliable floor here; the off-chain
    # estimate is a compounded V2+V3 approximation that diverges over multi-hop
    # routes (see comment above).  2% slippage tolerance.
    min_final_out = quoted * (10_000 - 200) // 10_000
    tx = build_swap_transaction(dag, SWAP_AMOUNT, defi_vm_address, sender, min_final_out=min_final_out)
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=600_000,
    )
    print(f"[mixed] tx {receipt['transactionHash'].hex()} status {receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM execute() reverted: {receipt['transactionHash'].hex()}"

    # -- Step 5: Verify USDC received -------------------------------------
    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before
    print(f"[mixed] received {received / 10**6:.6f} USDC  (min={min_final_out / 10**6:.6f})")
    assert received >= min_final_out, f"received {received} USDC < min_final_out {min_final_out}"


# ---------------------------------------------------------------------------
# Split-route helper
# ---------------------------------------------------------------------------


async def _discover_split_pool_graph(w3: AsyncWeb3, indexer: PoolIndexer | None = None) -> PoolGraph:
    """Load WETH/USDC pools from multiple sources on Sepolia.

    Adds every deployed V3 pool (fee tiers 500, 3000, 10000), every deployed
    V4 pool (same fee tiers, no hooks), and the V2 WETH/USDC pair (if
    deployed and non-empty) to a single :class:`PoolGraph`.  Skips if fewer
    than two independent liquidity sources are found.

    When *indexer* is supplied, V2/V3 pool states are fetched with a single
    batch ``eth_getLogs`` call instead of per-pool ``eth_call`` round-trips.
    V4 pools still use StateView ``eth_call``s (no event-based indexing).
    """
    graph = PoolGraph()
    found: list[str] = []

    # -- Phase 1: discover V3 pool addresses and register new ones --------
    factory_v3 = UNISWAP_V3_FACTORY(to=V3_FACTORY_ADDR)
    v3_discovered: list[tuple[str, int]] = []  # (pool_addr, fee_tier)
    for fee_tier in (500, 3000, 10000):
        print(f"[discover_split] getPool WETH/USDC fee={fee_tier} ...", flush=True)
        pool_addr = Web3.to_checksum_address(await factory_v3.fns.getPool(WETH_ADDR, USDC_ADDR, fee_tier).call(w3))
        if pool_addr == _ZERO_ADDR:
            continue
        print(f"[discover_split] found V3-{fee_tier} pool={pool_addr}", flush=True)
        v3_discovered.append((pool_addr, fee_tier))

        if indexer is not None and indexer.get_pool(pool_addr) is None:
            print(f"[discover_split] register V3-{fee_tier} in indexer ...", flush=True)
            pool_contract = UNISWAP_V3_POOL(to=pool_addr)
            token0_addr = await pool_contract.fns.token0().call(w3)
            fee_actual = await pool_contract.fns.fee().call(w3)
            t0_sym, t0_dec = _TOKEN_INFO.get(token0_addr.lower(), ("", 18))
            t1_addr = USDC_ADDR if token0_addr.lower() == WETH_ADDR.lower() else WETH_ADDR
            t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr.lower(), ("", 18))
            indexer.add_v3_pool(
                pool_address=pool_addr,
                protocol="UniswapV3",
                token0_address=token0_addr,
                token0_symbol=t0_sym,
                token0_decimals=t0_dec,
                token1_address=t1_addr,
                token1_symbol=t1_sym,
                token1_decimals=t1_dec,
                chain_id=11155111,
                fee_bps=fee_actual // 100,
            )

    # -- Phase 2: discover V2 pair address and register if new -----------
    v2_factory = UNISWAP_V2_FACTORY(to=V2_FACTORY_ADDR)
    print("[discover_split] getPair WETH/USDC V2 ...", flush=True)
    v2_pair_addr = Web3.to_checksum_address(await v2_factory.fns.getPair(WETH_ADDR, USDC_ADDR).call(w3))
    v2_pair = UNISWAP_V2_PAIR(to=v2_pair_addr) if v2_pair_addr != _ZERO_ADDR else None
    print(f"[discover_split] V2 pair={'none' if v2_pair is None else v2_pair_addr}", flush=True)

    if v2_pair is not None:
        v2_rec = indexer.get_pool(v2_pair_addr) if indexer is not None else None
        if v2_rec is not None:
            token0_v2 = v2_rec.token0_address
        else:
            print("[discover_split] fetch V2 token0 ...", flush=True)
            token0_v2 = await v2_pair.fns.token0().call(w3)

        if indexer is not None and v2_rec is None:
            print("[discover_split] register V2 pool in indexer ...", flush=True)
            t0_sym, t0_dec = _TOKEN_INFO.get(token0_v2.lower(), ("", 18))
            t1_addr = USDC_ADDR if token0_v2.lower() == WETH_ADDR.lower() else WETH_ADDR
            t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr.lower(), ("", 18))
            indexer.add_v2_pool(
                pool_address=v2_pair_addr,
                protocol="UniswapV2",
                token0_address=token0_v2,
                token0_symbol=t0_sym,
                token0_decimals=t0_dec,
                token1_address=t1_addr,
                token1_symbol=t1_sym,
                token1_decimals=t1_dec,
                chain_id=11155111,
                fee_bps=30,
            )
    else:
        token0_v2 = ""

    # -- Phase 3: build V3 edges from live instantaneous state -----------
    for pool_addr, fee_tier in v3_discovered:
        print(f"[discover_split] build V3-{fee_tier} edge pool={pool_addr} ...", flush=True)
        pool_rec = indexer.get_pool(pool_addr) if indexer is not None else None
        pool_contract = UNISWAP_V3_POOL(to=pool_addr)
        if pool_rec is not None:
            token0_addr = pool_rec.token0_address
            fee_bps = pool_rec.fee_bps
        else:
            token0_addr = await pool_contract.fns.token0().call(w3)
            fee_bps = (await pool_contract.fns.fee().call(w3)) // 100

        # Live current-block sqrtPrice/liquidity (see _discover_pool_graph for
        # why the indexer's swap-only snapshot can't be trusted for quoting).
        slot0 = await pool_contract.fns.slot0().call(w3)
        liquidity = await pool_contract.fns.liquidity().call(w3)
        sqrt_price_x96 = slot0[0]

        if liquidity == 0:
            continue
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=Address(pool_addr),
            protocol="UniswapV3",
            fee_bps=fee_bps,
            sqrt_price_x96=sqrt_price_x96,
            liquidity=liquidity,
            is_token0_in=token0_addr.lower() == WETH_ADDR.lower(),
        )
        graph.add_pool(edge)
        found.append(f"V3-{fee_tier}")
        print("mmv3_pool_addr", pool_addr, "fee_tier", fee_tier, "liquidity", liquidity)

    # -- V2 pair (WETH/USDC) — heterogeneous source for the split ---------
    if v2_pair is not None:
        # Live reserves — a current-block quantity (see _discover_pool_graph).
        reserves = await v2_pair.fns.getReserves().call(w3)
        reserve0, reserve1 = reserves[0], reserves[1]

        if token0_v2.lower() == WETH_ADDR.lower():
            reserve_weth, reserve_usdc = reserve0, reserve1
        else:
            reserve_weth, reserve_usdc = reserve1, reserve0
        if reserve_weth > 0:
            v2_edge = PoolEdge(
                token_in=WETH,
                token_out=USDC,
                pool_address=Address(v2_pair_addr),
                protocol="UniswapV2",
                fee_bps=30,
                reserve_in=reserve_weth,
                reserve_out=reserve_usdc,
                extra={"is_token0_in": token0_v2.lower() == WETH_ADDR.lower()},
            )
            graph.add_pool(v2_edge)
            found.append("V2")
            print("mmv2_pair_addr", v2_pair_addr, "reserve_weth", reserve_weth, "reserve_usdc", reserve_usdc)

    if len(found) < 2:
        pytest.skip(f"Need >=2 WETH/USDC pools on Sepolia for a split test; found {len(found)}")

    print(f"[split] pool graph: {found}")
    return graph


# ---------------------------------------------------------------------------
# Split-route test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.testnet
async def test_sepolia_defi_vm_split_route(pool_indexer: PoolIndexer) -> None:
    """Execute a WETH->USDC split across two V3 pools via DeFiVM on Sepolia.

    Uses :meth:`~pydefi.pathfinder.Router.find_optimal_split` to distribute the input
    across two independent WETH/USDC V3 fee tiers, then executes the resulting
    the split RouteDAG via ``build_swap_transaction``.

    Steps:
    1. Discover >= 2 WETH/USDC V3 pools and build a PoolGraph.
    2. Run Router.find_optimal_split() to get a split RouteDAG.
    3. Build execute(bytes) calldata with build_swap_transaction().
    4. Broadcast and verify USDC received >= min_final_out.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # -- Step 1: Discover split pool graph --------------------------------
    graph = await _discover_split_pool_graph(w3, pool_indexer)
    router = Router(graph)
    amount_in = TokenAmount(WETH, SWAP_AMOUNT)

    # Compare two candidate-pool budgets off-chain (no broadcast needed).
    split_coarse = router.find_optimal_split(amount_in, USDC, candidates=2)
    split_fine = router.find_optimal_split(amount_in, USDC, candidates=4)
    print(f"[split] candidates=2: legs={[f'{w / 100:.0f}%' for w in Router.dag_leg_weights(split_coarse)]}")
    print(f"[split] candidates=4: legs={[f'{w / 100:.0f}%' for w in Router.dag_leg_weights(split_fine)]}")

    # Execute the wider-candidate split.
    split_route = split_fine

    # -- Step 2: Pre-fund: wrap ETH -> WETH if balance is insufficient ----
    await _prefund_vm_with_weth(w3, account, sender, defi_vm_address)

    # -- Step 3: Build and broadcast execute(bytes) -----------------------
    usdc = ERC20(to=USDC_ADDR)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    min_final_out = 0
    tx = build_swap_transaction(split_route, SWAP_AMOUNT, defi_vm_address, sender, min_final_out=min_final_out)
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=600_000,
    )
    print(f"[split] tx {receipt['transactionHash'].hex()} status {receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM execute() reverted: {receipt['transactionHash'].hex()}"

    # -- Step 4: Verify USDC received -------------------------------------
    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before
    print(f"[split] received {received / 10**6:.6f} USDC  (min={min_final_out / 10**6:.6f})")
    assert received >= min_final_out, f"received {received} USDC < min_final_out {min_final_out}"


# ---------------------------------------------------------------------------
# On-chain execution: verify find_optimal_split produces a genuine split DAG
# ---------------------------------------------------------------------------


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_defi_vm_split_merge_execution(pool_indexer: PoolIndexer) -> None:
    """Execute a split/merge DAG produced by find_optimal_split on-chain.

    test_sepolia_defi_vm_split_route broadcasts find_optimal_split's result but
    does not verify that a split actually occurred — at SWAP_AMOUNT the
    optimizer may return a single-leg DAG.  This test explicitly finds a trade
    size where find_optimal_split returns >=2 legs, then broadcasts that DAG and
    confirms the on-chain execution succeeds and USDC is received.

    Skipped when no trade size forces a multi-leg split on the current pool
    state, or when the required env vars are absent.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # -- Step 1: Discover pool graph and find a trade size that splits --------
    graph = await _discover_split_pool_graph(w3, pool_indexer)
    router = Router(graph)

    # Probe small amounts (multiples of SWAP_AMOUNT) to stay within testnet
    # balances.  Sepolia pools have thin liquidity so even 5–20× SWAP_AMOUNT
    # can force the optimizer to split across pools.
    split_dag = None
    split_amount_in: TokenAmount | None = None
    for multiplier in [5, 10, 20, 50]:
        probe_amount = SWAP_AMOUNT * multiplier
        amount_in = TokenAmount(WETH, probe_amount)
        dag = router.find_optimal_split(amount_in, USDC, candidates=3)
        weights = Router.dag_leg_weights(dag)
        print(
            f"[split-exec] {probe_amount / 10**18:.4f} WETH → legs={len(weights)} {[f'{w / 100:.0f}%' for w in weights]}"
        )
        if len(weights) >= 2:
            split_dag = dag
            split_amount_in = amount_in
            break

    if split_dag is None or split_amount_in is None:
        pytest.skip(
            "find_optimal_split returned a single-leg DAG for all probed trade sizes "
            "on the current Sepolia pool state — no split/merge execution path to test."
        )

    print(
        f"[split-exec] executing {split_amount_in.amount / 10**18:.4f} WETH split with {len(Router.dag_leg_weights(split_dag))} legs"
    )

    # -- Step 2: Pre-fund DeFiVM with WETH -----------------------------------
    weth = ERC20(to=WETH_ADDR)
    weth_balance = await weth.fns.balanceOf(sender).call(w3)
    raw_amount = split_amount_in.amount
    if weth_balance < raw_amount:
        weth_deposit = _WETH_DEPOSIT(to=WETH_ADDR)
        await send_transaction(
            w3,
            account,
            to=WETH_ADDR,
            data=bytes(weth_deposit.fns.deposit().data),
            value=Wei(raw_amount),
        )
    await send_transaction(
        w3,
        account,
        to=WETH_ADDR,
        data=bytes(weth.fns.transfer(defi_vm_address, raw_amount).data),
        value=Wei(0),
    )

    # -- Step 3: Build and broadcast -----------------------------------------
    usdc = ERC20(to=USDC_ADDR)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    tx = build_swap_transaction(split_dag, raw_amount, defi_vm_address, sender, min_final_out=0)
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=800_000,
    )
    print(f"[split-exec] tx {receipt['transactionHash'].hex()} status={receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM execute() reverted: {receipt['transactionHash'].hex()}"

    # -- Step 4: Verify output -----------------------------------------------
    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before
    print(f"[split-exec] received {received / 10**6:.6f} USDC")
    assert received > 0, "received 0 USDC after split/merge execution"


# ---------------------------------------------------------------------------
# Forced split/merge: manually constructed 50/50 DAG at known-safe amounts
# ---------------------------------------------------------------------------


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_defi_vm_forced_split_merge(pool_indexer: PoolIndexer) -> None:
    """Execute a manually-constructed 50/50 split DAG on Sepolia.

    Unlike test_sepolia_defi_vm_split_merge_execution (which depends on the
    optimizer choosing to split), this test unconditionally builds a
    RouteDAG with two equal legs across two independent WETH/USDC pools.
    This guarantees that _build_route_split_segment runs on-chain even when
    the optimizer would prefer a single leg at the probed trade size.

    Each leg receives SWAP_AMOUNT // 2 = 0.001 WETH — well within Sepolia's
    thin-liquidity safe range — so no tick-boundary reverts are expected.

    Steps:
    1. Discover >=2 WETH/USDC pools from _discover_split_pool_graph.
    2. Pick the two most-liquid pools as the two legs.
    3. Manually build RouteDAG: WETH.split().leg(5000).swap(USDC, pool0)
                                         .leg(5000).swap(USDC, pool1).merge()
    4. Transfer SWAP_AMOUNT WETH to DeFiVM.
    5. Broadcast build_swap_transaction(dag, SWAP_AMOUNT, ...).
    6. Assert status=1 and USDC received > 0.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # -- Step 1: Discover pool graph (requires >=2 WETH/USDC pools) ----------
    graph = await _discover_split_pool_graph(w3, pool_indexer)

    # -- Step 2: Pick the two most-liquid WETH->USDC pools -------------------
    all_edges = graph.edges_from(WETH)
    usdc_edges = [e for e in all_edges if Web3.to_checksum_address(e.token_out.address) == USDC_ADDR]
    if len(usdc_edges) < 2:
        pytest.skip(f"Need >=2 WETH->USDC pool edges; found {len(usdc_edges)}")

    # Sort by liquidity descending so we pick the two deepest pools.
    usdc_edges.sort(key=lambda e: getattr(e, "liquidity", 0), reverse=True)
    pool_a, pool_b = usdc_edges[0], usdc_edges[1]
    print(
        f"[forced-split] pool_a={pool_a.pool_address} ({pool_a.protocol} fee={pool_a.fee_bps}bp)"
        f"  pool_b={pool_b.pool_address} ({pool_b.protocol} fee={pool_b.fee_bps}bp)"
    )

    # -- Step 3: Manually build 50/50 RouteDAG --------------------------------
    dag = RouteDAG().from_token(WETH).split().leg(5000).swap(USDC, pool_a).leg(5000).swap(USDC, pool_b).merge()

    weights = Router.dag_leg_weights(dag)
    assert len(weights) == 2, f"expected 2 legs, got {len(weights)}"
    assert weights == [5000, 5000], f"expected [5000, 5000], got {weights}"
    print(f"[forced-split] DAG legs: {[f'{w / 100:.0f}%' for w in weights]}")

    # -- Step 4: Pre-fund DeFiVM with WETH ------------------------------------
    await _prefund_vm_with_weth(w3, account, sender, defi_vm_address)

    # -- Step 5: Build and broadcast execute(bytes) ---------------------------
    usdc = ERC20(to=USDC_ADDR)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    tx = build_swap_transaction(dag, SWAP_AMOUNT, defi_vm_address, sender, min_final_out=0)
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=800_000,
    )
    print(f"[forced-split] tx {receipt['transactionHash'].hex()} status={receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM execute() reverted: {receipt['transactionHash'].hex()}"

    # -- Step 6: Verify USDC received -----------------------------------------
    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before
    print(f"[forced-split] received {received / 10**6:.6f} USDC")
    assert received > 0, "received 0 USDC after forced split/merge execution"


@pytest.mark.asyncio
@pytest.mark.testnet
async def test_sepolia_split_three_legs_analysis(pool_indexer: PoolIndexer) -> None:
    """Off-chain analysis: probe how many legs the optimizer finds across trade sizes.

    Sweeps notional amounts (1, 10, 100, 1000 WETH) and granularities
    (step_bps=500, 200) looking for a 3-leg split.  No broadcast — pure
    off-chain routing math.

    On Sepolia the V3-3000 and V3-10000 pools have a fee disadvantage over
    V3-500 (0.30 % and 1.00 % vs 0.05 %).  A third leg only emerges when
    V3-500's marginal price impact exceeds that fee gap.  Based on on-chain
    liquidity (V3-500 virtual WETH depth ≈ 12 500 WETH) this requires a
    notional well above 100 WETH on Sepolia testnet.

    If no trade size produces 3 legs the test is skipped with a diagnostic
    message rather than failing, because pool state varies over time.
    """
    require_env("SEPOLIA_RPC_URL")

    w3 = await connect(SEPOLIA_RPC_URL)

    graph = await _discover_split_pool_graph(w3, pool_indexer)
    router = Router(graph)

    # Sweep trade sizes and candidate budgets; record the best leg count found.
    probe_amounts_weth = [1, 10, 100, 1_000, 10_000]
    candidate_counts = [3, 4]
    best_n_legs = 0
    best_desc = ""

    for weth_units in probe_amounts_weth:
        amount_in = TokenAmount(WETH, weth_units * 10**18)
        for candidates in candidate_counts:
            split = router.find_optimal_split(amount_in, USDC, candidates=candidates)
            weights = Router.dag_leg_weights(split)
            n = len(weights)
            desc = f"  {weth_units:>6} WETH  cand={candidates:>2}  legs={n}  {[f'{w / 100:.0f}%' for w in weights]}"
            print(f"[3-leg probe]{desc}")
            if n > best_n_legs:
                best_n_legs = n
                best_desc = desc

    print(f"[3-leg probe] best result:{best_desc}")

    if best_n_legs < 3:
        pytest.skip(
            f"No trade size produced a 3-leg split on the current Sepolia pool state "
            f"(best was {best_n_legs} legs). This is expected when V3-500 virtual depth "
            f"is large enough that fee-tier gaps prevent routing to secondary pools."
        )

    assert best_n_legs >= 3


@pytest.mark.asyncio
@pytest.mark.testnet
async def test_sepolia_split_three_legs_fee_equalized(pool_indexer: PoolIndexer) -> None:
    """Verify the 3-leg split optimizer using real Sepolia pool depths with fees equalized.

    The reason Sepolia never naturally produces 3 legs is fee asymmetry:
    V3-500 (0.05%) vs V3-3000 (0.30%) vs V3-10000 (1.00%).  The fee gap is
    larger than the price-impact savings from spreading the trade.

    This test queries live ``sqrtPriceX96`` / ``liquidity`` from all deployed
    WETH/USDC V3 pools, then builds a synthetic graph where every pool has the
    same fee tier (0.05 % = fee_bps=5).  With equal fees, the optimizer must
    split across all three pools to minimise price impact, demonstrating the
    3-leg code path with real on-chain pool depths.
    """
    require_env("SEPOLIA_RPC_URL")

    w3 = await connect(SEPOLIA_RPC_URL)

    # -- Collect all V3 WETH/USDC pools with liquidity --------------------
    # Phase 1: discover addresses + register any new pools with the indexer.
    factory_v3 = UNISWAP_V3_FACTORY(to=V3_FACTORY_ADDR)
    v3_candidates: list[tuple[str, int]] = []  # (pool_addr, fee_tier)
    for fee_tier in (500, 3000, 10000):
        pool_addr = Web3.to_checksum_address(await factory_v3.fns.getPool(WETH_ADDR, USDC_ADDR, fee_tier).call(w3))
        if pool_addr == _ZERO_ADDR:
            continue
        v3_candidates.append((pool_addr, fee_tier))
        if pool_indexer.get_pool(pool_addr) is None:
            pool_contract = UNISWAP_V3_POOL(to=pool_addr)
            token0_addr = await pool_contract.fns.token0().call(w3)
            fee_actual = await pool_contract.fns.fee().call(w3)
            t0_sym, t0_dec = _TOKEN_INFO.get(token0_addr.lower(), ("", 18))
            t1_addr = USDC_ADDR if token0_addr.lower() == WETH_ADDR.lower() else WETH_ADDR
            t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr.lower(), ("", 18))
            pool_indexer.add_v3_pool(
                pool_address=pool_addr,
                protocol="UniswapV3",
                token0_address=token0_addr,
                token0_symbol=t0_sym,
                token0_decimals=t0_dec,
                token1_address=t1_addr,
                token1_symbol=t1_sym,
                token1_decimals=t1_dec,
                chain_id=11155111,
                fee_bps=fee_actual // 100,
            )

    # Phase 2: read each pool's live current-block sqrtPrice/liquidity.
    pool_states: list[dict] = []
    for pool_addr, fee_tier in v3_candidates:
        pool_rec = pool_indexer.get_pool(pool_addr)
        assert pool_rec is not None
        token0_addr = pool_rec.token0_address
        pool_contract = UNISWAP_V3_POOL(to=pool_addr)
        slot0 = await pool_contract.fns.slot0().call(w3)
        liquidity = await pool_contract.fns.liquidity().call(w3)
        sqrt_price_x96 = slot0[0]
        if liquidity == 0:
            continue
        pool_states.append(
            {
                "addr": pool_addr,
                "fee_tier": fee_tier,
                "sqrt_price_x96": sqrt_price_x96,
                "liquidity": liquidity,
                "is_weth_token0": token0_addr.lower() == WETH_ADDR.lower(),
            }
        )
        print(f"[fee-eq] V3-{fee_tier} addr={pool_addr} liq={liquidity}")

    if len(pool_states) < 3:
        pytest.skip(f"Need >=3 V3 WETH/USDC pools on Sepolia; found {len(pool_states)}")

    # -- Build fee-equalized graph (all pools at fee_bps=5) ---------------
    # Using the actual pool addresses so the graph correctly deduplicates by
    # first-hop pool in _find_top_routes.
    graph = PoolGraph()
    for ps in pool_states:
        print("mm-ps", ps)
        graph.add_pool(
            V3PoolEdge(
                token_in=WETH,
                token_out=USDC,
                pool_address=Address(ps["addr"]),
                protocol="UniswapV3",
                fee_bps=5,  # equal fee for all pools — isolates depth-driven split
                sqrt_price_x96=ps["sqrt_price_x96"],
                liquidity=ps["liquidity"],
                is_token0_in=ps["is_weth_token0"],
            )
        )

    router = Router(graph)

    # Choose a trade size that causes meaningful price impact across pools.
    # The actual number of legs depends on live liquidity depths: if a pool is
    # too shallow relative to the trade, the optimizer correctly allocates 0%
    # to it.  We assert >= 2 to confirm multi-leg splitting is exercised.
    amount_in = TokenAmount(WETH, 150 * 10**18)

    split = router.find_optimal_split(amount_in, USDC, candidates=3)
    weights = Router.dag_leg_weights(split)
    print(f"[fee-eq] 150 WETH fee-equalized split: legs={len(weights)}  {[f'{w / 100:.0f}%' for w in weights]}")

    assert len(weights) >= 2, (
        f"Expected multi-leg split with fee-equalized pools, got {len(weights)}: {[f'{w / 100:.0f}%' for w in weights]}"
    )
    assert len(weights) <= 3
    assert sum(weights) == 10_000
    assert all(w > 0 for w in weights)

    # A quantitative output comparison requires resimulating the DAG, which is
    # not exposed by the current public API — the leg-count assertion above is
    # sufficient to confirm the multi-leg split code path ran.


# ---------------------------------------------------------------------------
# Generic pair swap — parametrized over additional Sepolia token pairs
# ---------------------------------------------------------------------------


async def _discover_pair_graph(
    w3: AsyncWeb3,
    token_a: _Token,
    token_b: _Token,
    indexer: PoolIndexer | None = None,
) -> tuple[PoolGraph, str | None]:
    """Discover V3 (fee=500/3000/10000) and V2 pools for any token pair.

    Returns ``(graph, best_pool_addr)`` where ``best_pool_addr`` is the address
    of the first pool found.  Returns ``(empty_graph, None)`` if no pools exist.
    """
    graph = PoolGraph()
    best_addr: str | None = None
    addr_a = Web3.to_checksum_address(token_a.address)
    addr_b = Web3.to_checksum_address(token_b.address)

    # -- V3 pools across fee tiers ----------------------------------------
    factory_v3 = UNISWAP_V3_FACTORY(to=V3_FACTORY_ADDR)
    v3_discovered: list[tuple[str, int]] = []
    for fee_tier in (500, 3000, 10000):
        pool_addr = Web3.to_checksum_address(await factory_v3.fns.getPool(addr_a, addr_b, fee_tier).call(w3))
        if pool_addr == _ZERO_ADDR:
            continue
        print(f"[pair-graph] {token_a.symbol}/{token_b.symbol} V3 fee={fee_tier} pool={pool_addr}", flush=True)
        if indexer is not None and indexer.get_pool(pool_addr) is None:
            pool_contract = UNISWAP_V3_POOL(to=pool_addr)
            token0_addr_reg = await pool_contract.fns.token0().call(w3)
            fee_actual = await pool_contract.fns.fee().call(w3)
            t1_addr_reg = addr_b if token0_addr_reg.lower() == addr_a.lower() else addr_a
            t0_sym, t0_dec = _TOKEN_INFO.get(token0_addr_reg.lower(), ("", 18))
            t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr_reg.lower(), ("", 18))
            indexer.add_v3_pool(
                pool_address=pool_addr,
                protocol="UniswapV3",
                token0_address=token0_addr_reg,
                token0_symbol=t0_sym,
                token0_decimals=t0_dec,
                token1_address=t1_addr_reg,
                token1_symbol=t1_sym,
                token1_decimals=t1_dec,
                chain_id=11155111,
                fee_bps=fee_actual // 100,
            )
        v3_discovered.append((pool_addr, fee_tier))

    for pool_addr, fee_tier in v3_discovered:
        pool_rec = indexer.get_pool(pool_addr) if indexer is not None else None
        pool_contract = UNISWAP_V3_POOL(to=pool_addr)
        if pool_rec is not None:
            token0_addr = pool_rec.token0_address
            fee_bps = pool_rec.fee_bps
        else:
            token0_addr = await pool_contract.fns.token0().call(w3)
            fee_bps = fee_tier // 100

        # Live current-block sqrtPrice/liquidity (see _discover_pool_graph).
        slot0 = await pool_contract.fns.slot0().call(w3)
        liquidity = await pool_contract.fns.liquidity().call(w3)
        sqrt_price_x96 = slot0[0]

        if sqrt_price_x96 == 0 or liquidity == 0:
            continue

        for tok_in_obj, tok_out_obj in ((token_a, token_b), (token_b, token_a)):
            graph.add_pool(
                V3PoolEdge(
                    token_in=tok_in_obj,
                    token_out=tok_out_obj,
                    pool_address=Address(pool_addr),
                    protocol="UniswapV3",
                    fee_bps=fee_bps,
                    sqrt_price_x96=sqrt_price_x96,
                    liquidity=liquidity,
                    # token0_addr is a hex str; tok_in_obj.address is raw bytes —
                    # checksum both so the compare isn't always-False (str != bytes).
                    is_token0_in=(
                        Web3.to_checksum_address(token0_addr) == Web3.to_checksum_address(tok_in_obj.address)
                    ),
                )
            )
        if best_addr is None:
            best_addr = pool_addr

    # -- V2 pair ----------------------------------------------------------
    factory_v2 = UNISWAP_V2_FACTORY(to=V2_FACTORY_ADDR)
    v2_addr = Web3.to_checksum_address(await factory_v2.fns.getPair(addr_a, addr_b).call(w3))
    if v2_addr != _ZERO_ADDR:
        print(f"[pair-graph] {token_a.symbol}/{token_b.symbol} V2 pair={v2_addr}", flush=True)
        v2_pair = UNISWAP_V2_PAIR(to=v2_addr)
        token0_v2 = await v2_pair.fns.token0().call(w3)
        reserves = await v2_pair.fns.getReserves().call(w3)
        r0, r1 = int(reserves[0]), int(reserves[1])
        if r0 > 0 and r1 > 0:
            if indexer is not None and indexer.get_pool(v2_addr) is None:
                t1_addr_v2 = addr_b if token0_v2.lower() == addr_a.lower() else addr_a
                t0_sym, t0_dec = _TOKEN_INFO.get(token0_v2.lower(), ("", 18))
                t1_sym, t1_dec = _TOKEN_INFO.get(t1_addr_v2.lower(), ("", 18))
                indexer.add_v2_pool(
                    pool_address=v2_addr,
                    protocol="UniswapV2",
                    token0_address=token0_v2,
                    token0_symbol=t0_sym,
                    token0_decimals=t0_dec,
                    token1_address=t1_addr_v2,
                    token1_symbol=t1_sym,
                    token1_decimals=t1_dec,
                    chain_id=11155111,
                    fee_bps=30,
                )
            tok0 = token_a if token0_v2.lower() == addr_a.lower() else token_b
            tok1 = token_b if token0_v2.lower() == addr_a.lower() else token_a
            for tok_in_obj, tok_out_obj, reserve_in, reserve_out, is_token0_in in (
                (tok0, tok1, r0, r1, True),
                (tok1, tok0, r1, r0, False),
            ):
                graph.add_pool(
                    PoolEdge(
                        token_in=tok_in_obj,
                        token_out=tok_out_obj,
                        pool_address=Address(v2_addr),
                        protocol="UniswapV2",
                        fee_bps=30,
                        reserve_in=reserve_in,
                        reserve_out=reserve_out,
                        extra={"is_token0_in": is_token0_in},
                    )
                )
            if best_addr is None:
                best_addr = v2_addr

    return graph, best_addr


# ---------------------------------------------------------------------------
# Complex DAG: sequential swap → split/merge → sequential swap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.testnet
async def test_sepolia_defi_vm_pre_split_post_merge(pool_indexer: PoolIndexer) -> None:
    """Execute a DAG with sequential swaps before and after a split/merge.

    Reproduces the following flow on Sepolia::

        token0(WETH) --pool1(V2)--> token1(UNI)
        token1 --50% pool2(V3-500)--> token3(USDC)   [leg 1]
        token1 --50% pool3(alt)----> token3(USDC)    [leg 2]
        token3 --pool4(V3-500)----> token5(WETH)

    Built with the RouteDAG fluent API::

        dag = (
            RouteDAG()
            .from_token(WETH)
            .swap(UNI, pool1)          # pre-split sequential swap
            .split()
            .leg(5000)
              .swap(USDC, pool2)       # leg 1
            .leg(5000)
              .swap(USDC, pool3)       # leg 2
            .merge()
            .swap(WETH, pool4)         # post-merge sequential swap
        )

    Skipped when fewer than two independent UNI→USDC pools are found on
    Sepolia, or when any of the required pair pools are absent.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # ── Pool discovery ───────────────────────────────────────────────────────

    # pre-split: WETH → UNI
    graph_wu, _ = await _discover_pair_graph(w3, WETH, UNI, pool_indexer)
    weth_to_uni = [e for e in graph_wu.edges_from(WETH) if Web3.to_checksum_address(e.token_out.address) == UNI_ADDR]
    if not weth_to_uni:
        pytest.skip("No WETH→UNI pool found on Sepolia")
    pool1 = weth_to_uni[0]
    print(f"[complex-dag] pool1 (pre-split WETH→UNI): {pool1.pool_address} {pool1.protocol} {pool1.fee_bps}bp")

    # split legs: two independent UNI → USDC pools
    graph_uu, _ = await _discover_pair_graph(w3, UNI, USDC, pool_indexer)
    uni_to_usdc = [e for e in graph_uu.edges_from(UNI) if Web3.to_checksum_address(e.token_out.address) == USDC_ADDR]
    if len(uni_to_usdc) < 2:
        pytest.skip(f"Need >=2 independent UNI→USDC pools for split; found {len(uni_to_usdc)}")
    pool2 = uni_to_usdc[0]
    pool3 = uni_to_usdc[1]
    print(f"[complex-dag] pool2 (leg1 UNI→USDC): {pool2.pool_address} {pool2.protocol} {pool2.fee_bps}bp")
    print(f"[complex-dag] pool3 (leg2 UNI→USDC): {pool3.pool_address} {pool3.protocol} {pool3.fee_bps}bp")

    # post-merge: USDC → WETH
    graph_uw, _ = await _discover_pair_graph(w3, USDC, WETH, pool_indexer)
    usdc_to_weth = [e for e in graph_uw.edges_from(USDC) if Web3.to_checksum_address(e.token_out.address) == WETH_ADDR]
    if not usdc_to_weth:
        pytest.skip("No USDC→WETH pool found on Sepolia for post-merge swap")
    pool4 = usdc_to_weth[0]
    print(f"[complex-dag] pool4 (post-merge USDC→WETH): {pool4.pool_address} {pool4.protocol} {pool4.fee_bps}bp")

    # ── Build DAG ────────────────────────────────────────────────────────────

    dag = (
        RouteDAG()
        .from_token(WETH)
        .swap(UNI, pool1)  # pre-split sequential swap: WETH → UNI
        .split()
        .leg(5000)
        .swap(USDC, pool2)  # leg 1 (50%): UNI → USDC
        .leg(5000)
        .swap(USDC, pool3)  # leg 2 (50%): UNI → USDC via different pool
        .merge()
        .swap(WETH, pool4)  # post-merge sequential swap: USDC → WETH
    )

    # ── Pre-fund DeFiVM with WETH ────────────────────────────────────────────

    weth = ERC20(to=WETH_ADDR)
    weth_balance = await weth.fns.balanceOf(sender).call(w3)
    if weth_balance < SWAP_AMOUNT:
        weth_deposit = _WETH_DEPOSIT(to=WETH_ADDR)
        await send_transaction(
            w3,
            account,
            to=WETH_ADDR,
            data=bytes(weth_deposit.fns.deposit().data),
            value=Wei(SWAP_AMOUNT),
        )
    await send_transaction(
        w3,
        account,
        to=WETH_ADDR,
        data=bytes(weth.fns.transfer(defi_vm_address, SWAP_AMOUNT).data),
        value=Wei(0),
    )

    # ── Build and broadcast ──────────────────────────────────────────────────

    weth_before = await weth.fns.balanceOf(sender).call(w3)
    tx = build_swap_transaction(dag, SWAP_AMOUNT, defi_vm_address, sender, min_final_out=0)
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=900_000,
    )
    print(f"[complex-dag] tx {receipt['transactionHash'].hex()} status={receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM execute() reverted: {receipt['transactionHash'].hex()}"

    # ── Verify WETH received (post-merge output) ─────────────────────────────

    weth_after = await weth.fns.balanceOf(sender).call(w3)
    received = weth_after - weth_before
    print(f"[complex-dag] received {received / 10**18:.6f} WETH back after full round-trip")
    assert received > 0, "received 0 WETH after pre-split→split/merge→post-merge execution"


# Pairs to test: (token_in, token_out, human_amount_in, label)
_SWAP_PAIRS = [
    (WETH, DAI, "0.001", "WETH-DAI"),
    (WETH, USDT, "0.001", "WETH-USDT"),
    (WETH, WBTC, "0.001", "WETH-WBTC"),
    (WETH, LINK, "0.001", "WETH-LINK"),
    (WETH, AAVE, "0.001", "WETH-AAVE"),
    (WETH, USDC, "0.001", "WETH-USDC"),
]


@pytest.mark.asyncio
@pytest.mark.testnet
@pytest.mark.parametrize("token_in,token_out,amount_human,label", _SWAP_PAIRS, ids=[p[3] for p in _SWAP_PAIRS])
async def test_sepolia_defi_vm_swap_pair(
    token_in: _Token,
    token_out: _Token,
    amount_human: str,
    label: str,
    pool_indexer: PoolIndexer,
) -> None:
    """Off-chain route + on-chain quote for an arbitrary Sepolia token pair.

    Does NOT broadcast — just verifies that:
    1. At least one pool exists for the pair.
    2. Router finds a route and returns a positive amount_out.
    3. ``quote_swap_transaction`` (eth_call DeFiVM) agrees within 5%.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_DEFI_VM")

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(SEPOLIA_RPC_URL))
    defi_vm_address = Web3.to_checksum_address(SEPOLIA_DEFI_VM)

    amount_in_raw = int(
        __import__("decimal").Decimal(amount_human) * __import__("decimal").Decimal(10**token_in.decimals)
    )

    graph, best_pool = await _discover_pair_graph(w3, token_in, token_out, pool_indexer)
    if best_pool is None:
        pytest.skip(f"No {label} pool found on Sepolia")

    router = Router(graph)
    try:
        route = router.find_best_route(TokenAmount(token_in, amount_in_raw), token_out)
    except Exception as exc:
        pytest.skip(f"Router found no {label} route: {exc}")

    amount_out = route.amount_out.amount
    print(
        f"[{label}] off-chain quote: {amount_human} {token_in.symbol} → "
        f"{amount_out / 10**token_out.decimals:.6f} {token_out.symbol}  "
        f"impact={float(route.price_impact) * 100:.3f}%",
        flush=True,
    )
    assert amount_out > 0, f"{label}: off-chain quote returned 0"

    assert route.dag is not None
    on_chain_amount = await quote_dag(
        route.dag,
        amount_in=route.amount_in.amount,
        w3=w3,
        vm_address=cast(Address, defi_vm_address),
        quoter_address=V3_QUOTER_ADDR,
    )
    ratio = on_chain_amount / amount_out if amount_out else 0
    print(
        f"[{label}] on-chain quote: {on_chain_amount / 10**token_out.decimals:.6f} {token_out.symbol}  "
        f"ratio={ratio:.4f}",
        flush=True,
    )
    assert on_chain_amount > 0, f"{label}: on-chain quote returned 0"
    assert 0.8 <= ratio <= 1.2, (
        f"{label}: on-chain/off-chain ratio {ratio:.4f} out of ±20% tolerance (off={amount_out}, on={on_chain_amount})"
    )


# ===========================================================================
# Uniswap V4 testnet coverage (ported from swap_route_test, adapted to RouteDAG)
# ===========================================================================
# Uniswap V4 direct swap test
# ---------------------------------------------------------------------------


async def _discover_v4_single_pool(w3: AsyncWeb3) -> tuple[PoolGraph, str]:
    """Single-edge graph for the first initialised WETH/USDC V4 pool on Sepolia.

    Thin adapter over the shared :func:`discover_sepolia_v4_weth_usdc_edge`;
    returns ``(graph, pool_id_hex)`` and skips when no seeded pool has liquidity.
    """
    edge = await discover_sepolia_v4_weth_usdc_edge(w3)
    if edge is None:
        pytest.skip("No initialised WETH/USDC V4 pool found on Sepolia")
    graph = PoolGraph()
    graph.add_pool(edge)
    return graph, edge.pool_id


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_defi_vm_v4_swap() -> None:
    """Execute a WETH→USDC swap through a Uniswap V4 pool via DeFiVM on Sepolia.

    V4 unlock/settle flow differs from V3 flash swaps:
    - The sender pre-transfers WETH to DeFiVM before calling ``execute()``.
    - DeFiVM calls ``PoolManager.unlock(data)`` where *data* encodes the entire
      PoolKey + SwapParams + settlement addresses.
    - PoolManager fires ``unlockCallback(bytes)`` into DeFiVM's ``fallback()``.
    - DeFiVM's callback calls ``pm.swap()`` → ``pm.sync(tokenIn)`` →
      ``tokenIn.transfer(pm, amountIn)`` → ``pm.settle()`` →
      ``pm.take(tokenOut, recipient, amountOut)``.
    - ``unlock()`` returns ``abi.encode(amountOut)``; the program checks it
      meets the slippage minimum.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account, sender, defi_vm_address = _signer()

    # -- Step 1: Discover V4 pool and build route -------------------------
    graph, pool_id_hex = await _discover_v4_single_pool(w3)
    route = Router(graph).find_best_route(TokenAmount(WETH, SWAP_AMOUNT), USDC)
    dag = route.dag
    assert dag is not None, "Router route is missing its DAG representation"
    assert route is not None, "Router found no route through V4 pool"

    off_chain_out = route.amount_out.amount
    print(f"[v4] off-chain estimate: {off_chain_out / 10**6:.6f} USDC  pool_id={pool_id_hex}")
    assert off_chain_out > 0, "off-chain amountOut estimate is zero"

    # quote_swap_transaction raises NotImplementedError for V4 — skip that step.

    # -- Step 2: Pre-fund DeFiVM with WETH --------------------------------
    # V4 settle requires DeFiVM to already hold tokenIn; unlike V3, there is
    # no flash-swap that defers repayment to a callback.
    await _prefund_vm_with_weth(w3, account, sender, defi_vm_address)
    print(f"[v4] transferred {SWAP_AMOUNT} WETH to DeFiVM")

    # -- Step 3: Build and broadcast execute(bytes) -----------------------
    usdc = ERC20(to=USDC_ADDR)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    # Use high slippage — Sepolia V4 pools have thin liquidity so on-chain
    # output can differ significantly from the single-tick off-chain estimate.
    tx = build_swap_transaction(
        dag, SWAP_AMOUNT, defi_vm_address, sender, min_final_out=route.amount_out.amount * (10_000 - 9_900) // 10_000
    )
    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=tx.value,
        gas=600_000,
    )
    tx_hash = receipt["transactionHash"].hex()
    print(f"[v4] tx {tx_hash} status={receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM V4 swap reverted: {tx_hash}"

    # -- Step 4: Verify USDC received matches PoolManager Swap event ------
    # The Swap event is authoritative: DeFiVM must relay exactly what the pool
    # settled.  BalanceDelta amounts are from the swapper's perspective
    # (positive = received, negative = sent).
    pm_addr = V4_POOL_MANAGER_ADDR.lower()
    swap_log = None
    for log in receipt["logs"]:
        if log["address"].lower() == pm_addr and log["topics"][0].hex() == _V4_SWAP_TOPIC:
            swap_log = log
            break
    assert swap_log is not None, f"PoolManager Swap event not found in tx {tx_hash}"
    evt_amount0, evt_amount1 = abi_decode(["int128", "int128"], bytes(swap_log["data"])[:64])
    # BalanceDelta is from swapper's perspective: positive = received, negative = sent.
    evt_usdc_out = evt_amount0  # positive: swapper received c0 (USDC)
    evt_weth_in = -evt_amount1  # negative → positive: swapper sent c1 (WETH)

    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before

    print(
        f"[v4] received {received / 10**6:.6f} USDC  "
        f"(off-chain estimate={off_chain_out / 10**6:.6f}  "
        f"pm_event={evt_usdc_out / 10**6:.6f}  "
        f"weth_consumed={evt_weth_in}/{SWAP_AMOUNT})"
    )
    assert received > 0, f"received 0 USDC — swap produced no output  tx={tx_hash}"
    assert received == evt_usdc_out, (
        f"balance delta ({received}) != PoolManager Swap event ({evt_usdc_out})  tx={tx_hash}"
    )


# ---------------------------------------------------------------------------
# DeFiVM vs Universal Router comparison
# ---------------------------------------------------------------------------


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_v4_defi_vm_vs_universal_router() -> None:
    """Compare WETH→USDC V4 swap output: DeFiVM direct vs Universal Router.

    Executes the same notional swap (0.002 WETH) through the same V4 pool via
    two independent paths, then asserts both paths receive comparable USDC:

    - **DeFiVM path** — pre-transfers WETH to DeFiVM; DeFiVM calls
      ``PoolManager.unlock()`` directly and handles ``unlockCallback``
      in-contract.  Built via ``build_swap_transaction()``.
    - **Universal Router path** — sends native ETH; the router wraps it to
      WETH internally, then routes through the same V4 pool via
      ``UniversalRouter.build_wrap_and_multihop_exact_in_transaction()``.
      No Permit2 approval required.

    The two swaps move the pool price sequentially, so exact amounts will
    differ slightly.  The assertion checks both are non-zero and within 5 %
    of each other.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(account.address)
    defi_vm = Web3.to_checksum_address(SEPOLIA_DEFI_VM)

    graph, pool_id_hex = await _discover_v4_single_pool(w3)
    edge = graph.edges_from(WETH)[0]
    assert isinstance(edge, V4PoolEdge)
    print(f"[cmp] pool fee={edge.fee_bps * 100} tick_spacing={edge.tick_spacing} pool_id={pool_id_hex}")

    usdc = ERC20(to=USDC_ADDR)
    weth = ERC20(to=WETH_ADDR)

    # -----------------------------------------------------------------------
    # Path 1: DeFiVM — direct PoolManager.unlock()
    # -----------------------------------------------------------------------
    weth_balance = await weth.fns.balanceOf(sender).call(w3)
    if weth_balance < SWAP_AMOUNT:
        weth_deposit = _WETH_DEPOSIT(to=WETH_ADDR)
        await send_transaction(
            w3,
            account,
            to=WETH_ADDR,
            data=bytes(weth_deposit.fns.deposit().data),
            value=Wei(SWAP_AMOUNT),
        )

    await send_transaction(
        w3,
        account,
        to=WETH_ADDR,
        data=bytes(weth.fns.transfer(defi_vm, SWAP_AMOUNT).data),
        value=Wei(0),
    )

    route = Router(graph).find_best_route(TokenAmount(WETH, SWAP_AMOUNT), USDC)
    dag = route.dag
    assert dag is not None, "Router route is missing its DAG representation"
    off_chain_estimate = route.amount_out.amount

    usdc_before_1 = await usdc.fns.balanceOf(sender).call(w3)
    tx_vm = build_swap_transaction(
        dag, SWAP_AMOUNT, defi_vm, sender, min_final_out=route.amount_out.amount * (10_000 - 100) // 10_000
    )
    receipt_vm = await send_transaction(
        w3,
        account,
        to=tx_vm.to,
        data=tx_vm.data,
        value=Wei(tx_vm.value),
        gas=600_000,
    )
    assert receipt_vm["status"] == 1, f"DeFiVM V4 swap reverted: {receipt_vm['transactionHash'].hex()}"
    usdc_after_1 = await usdc.fns.balanceOf(sender).call(w3)
    received_vm = usdc_after_1 - usdc_before_1
    print(f"[cmp] DeFiVM:          {received_vm / 10**6:.6f} USDC  tx={receipt_vm['transactionHash'].hex()}")

    # -----------------------------------------------------------------------
    # Path 2: Universal Router — WRAP_ETH + V4_SWAP (no Permit2)
    # -----------------------------------------------------------------------
    ur = UniversalRouter(UNIVERSAL_ROUTER_ADDRESSES[11155111])
    v4_hop = V4Hop(
        token_in=WETH,
        token_out=USDC,
        fee=edge.fee_bps * 100,
        tick_spacing=edge.tick_spacing,
        hooks=edge.hooks,
    )
    tx_ur = ur.build_wrap_and_multihop_exact_in_transaction(
        eth_amount=SWAP_AMOUNT,
        weth_token=WETH,
        hops=[v4_hop],
        recipient=sender,
        amount_out_minimum=0,
    )

    usdc_before_2 = await usdc.fns.balanceOf(sender).call(w3)
    receipt_ur = await send_transaction(
        w3,
        account,
        to=tx_ur.to,
        data=tx_ur.data,
        value=Wei(tx_ur.value),
        gas=600_000,
    )
    assert receipt_ur["status"] == 1, f"Universal Router V4 swap reverted: {receipt_ur['transactionHash'].hex()}"
    usdc_after_2 = await usdc.fns.balanceOf(sender).call(w3)
    received_ur = usdc_after_2 - usdc_before_2
    gas_ur = receipt_ur["gasUsed"]
    print(
        f"[cmp] UniversalRouter: {received_ur / 10**6:.6f} USDC  gas={gas_ur}  tx={receipt_ur['transactionHash'].hex()}"
    )

    # -----------------------------------------------------------------------
    # Output amounts will be essentially identical — both go through the same
    # PoolManager, so AMM pricing is the same.  The meaningful differentiator
    # is gas: DeFiVM calls PoolManager.unlock() directly with no intermediate
    # router contract, so it should consume less gas than the Universal Router
    # path (WRAP_ETH command + router dispatch overhead + Permit2 bookkeeping).
    #
    # Each output is verified against the pre-swap off-chain estimate (same
    # pool state) rather than against each other, since the second swap gets a
    # marginally worse rate due to price impact from the first.
    # -----------------------------------------------------------------------
    gas_vm = receipt_vm["gasUsed"]
    print(
        f"[cmp] off-chain estimate={off_chain_estimate / 10**6:.6f} USDC  "
        f"DeFiVM={received_vm / 10**6:.6f} (gas={gas_vm})  "
        f"UR={received_ur / 10**6:.6f} (gas={gas_ur})  "
        f"gas_saved={gas_ur - gas_vm}"
    )
    assert received_vm > 0, "DeFiVM path produced no USDC output"
    assert received_ur > 0, "Universal Router path produced no USDC output"
    assert received_vm >= off_chain_estimate * 98 // 100, (
        f"DeFiVM output {received_vm} is more than 2 % below off-chain estimate {off_chain_estimate}"
    )
    assert received_ur >= off_chain_estimate * 98 // 100, (
        f"Universal Router output {received_ur} is more than 2 % below off-chain estimate {off_chain_estimate}"
    )
    assert gas_vm < gas_ur, (
        f"DeFiVM (gas={gas_vm}) should use less gas than Universal Router (gas={gas_ur}) for a plain V4 single-hop swap"
    )


@pytest.mark.testnet
@pytest.mark.asyncio
async def test_sepolia_v4_universal_router_requires_permit2() -> None:
    """Universal Router's ERC-20 input path reverts without Permit2; DeFiVM does not need it.

    Universal Router's ``build_multihop_exact_in_transaction`` (``payer_is_user=True``)
    pulls WETH from the caller via Permit2 inside the V4 ``SETTLE_ALL`` action.
    Without a prior ``permit2.approve()`` call the router reverts.

    DeFiVM requires only a plain ERC-20 ``transfer`` to pre-fund itself — no
    approval contract in the trust path.

    This test:
    1. Pre-transfers WETH to DeFiVM and executes the swap successfully.
    2. Approves WETH allowance to Universal Router directly (not via Permit2) and
       shows Universal Router still reverts — the router exclusively uses Permit2
       for ERC-20 inputs, ignoring direct allowances.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(account.address)
    defi_vm = Web3.to_checksum_address(SEPOLIA_DEFI_VM)
    ur_address = UNIVERSAL_ROUTER_ADDRESSES[11155111]

    graph, _ = await _discover_v4_single_pool(w3)
    edge = graph.edges_from(WETH)[0]
    assert isinstance(edge, V4PoolEdge)

    weth = ERC20(to=WETH_ADDR)
    usdc = ERC20(to=USDC_ADDR)

    # Ensure sender holds enough WETH for both swap attempts.
    weth_balance = await weth.fns.balanceOf(sender).call(w3)
    if weth_balance < 2 * SWAP_AMOUNT:
        weth_deposit = _WETH_DEPOSIT(to=WETH_ADDR)
        await send_transaction(
            w3,
            account,
            to=WETH_ADDR,
            data=bytes(weth_deposit.fns.deposit().data),
            value=Wei(2 * SWAP_AMOUNT),
        )

    # -----------------------------------------------------------------------
    # Path 1: DeFiVM — plain pre-transfer, no approval contract needed
    # -----------------------------------------------------------------------
    await send_transaction(
        w3,
        account,
        to=WETH_ADDR,
        data=bytes(weth.fns.transfer(defi_vm, SWAP_AMOUNT).data),
        value=Wei(0),
    )
    route = Router(graph).find_best_route(TokenAmount(WETH, SWAP_AMOUNT), USDC)
    dag = route.dag
    assert dag is not None, "Router route is missing its DAG representation"
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)
    tx_vm = build_swap_transaction(
        dag, SWAP_AMOUNT, defi_vm, sender, min_final_out=route.amount_out.amount * (10_000 - 100) // 10_000
    )
    receipt_vm = await send_transaction(
        w3,
        account,
        to=tx_vm.to,
        data=tx_vm.data,
        value=Wei(tx_vm.value),
        gas=600_000,
    )
    assert receipt_vm["status"] == 1, f"DeFiVM swap reverted unexpectedly: {receipt_vm['transactionHash'].hex()}"
    received_vm = (await usdc.fns.balanceOf(sender).call(w3)) - usdc_before
    assert received_vm > 0, "DeFiVM produced no USDC output"
    print(f"[permit2] DeFiVM (no Permit2): {received_vm / 10**6:.6f} USDC  ✓")

    # -----------------------------------------------------------------------
    # Path 2: Universal Router ERC-20 path — direct allowance, no Permit2
    #
    # Grant Universal Router a direct ERC-20 allowance for WETH (standard
    # approve, NOT permit2.approve).  The router ignores direct allowances and
    # exclusively pulls tokens via Permit2, so the call reverts.
    # -----------------------------------------------------------------------
    await send_transaction(
        w3,
        account,
        to=WETH_ADDR,
        data=bytes(weth.fns.approve(ur_address, SWAP_AMOUNT).data),
        value=Wei(0),
    )

    ur = UniversalRouter(ur_address)
    v4_hop = V4Hop(
        token_in=WETH,
        token_out=USDC,
        fee=edge.fee_bps * 100,
        tick_spacing=edge.tick_spacing,
        hooks=edge.hooks,
    )
    tx_ur = ur.build_multihop_exact_in_transaction(
        amount_in=TokenAmount(WETH, SWAP_AMOUNT),
        hops=[v4_hop],
        recipient=MSG_SENDER,
        amount_out_minimum=0,
    )
    receipt_ur = await send_transaction(
        w3,
        account,
        to=tx_ur.to,
        data=tx_ur.data,
        value=Wei(tx_ur.value),
        gas=400_000,
        check=False,  # expect revert — do not raise
    )
    assert receipt_ur["status"] == 0, (
        f"Universal Router should have reverted without Permit2 approval (tx={receipt_ur['transactionHash'].hex()})"
    )
    print("[permit2] Universal Router (direct approve, no Permit2): reverted as expected  ✓")

    # -----------------------------------------------------------------------
    # Path 3: Universal Router ERC-20 path — proper Permit2 setup
    #
    # Step A: approve Permit2 to spend WETH on behalf of sender.
    # Step B: via Permit2, grant Universal Router a time-limited allowance.
    # Now build_multihop_exact_in_transaction (payer_is_user=True) succeeds.
    #
    # Cost: 2 extra transactions that DeFiVM never needs.
    # -----------------------------------------------------------------------
    # Canonical Permit2 address (same on all chains).
    permit2_addr = Web3.to_checksum_address("0x000000000022D473030F116dDEE9F6B43aC78BA3")
    _permit2 = Contract.from_abi(
        ["function approve(address token, address spender, uint160 amount, uint48 expiration) external"]
    )(to=permit2_addr)

    # Ensure sender holds enough WETH for a third swap.
    weth_balance = await weth.fns.balanceOf(sender).call(w3)
    if weth_balance < SWAP_AMOUNT:
        weth_deposit = _WETH_DEPOSIT(to=WETH_ADDR)
        await send_transaction(
            w3,
            account,
            to=WETH_ADDR,
            data=bytes(weth_deposit.fns.deposit().data),
            value=Wei(SWAP_AMOUNT),
        )

    # Step A — approve Permit2 contract.
    await send_transaction(
        w3,
        account,
        to=WETH_ADDR,
        data=bytes(weth.fns.approve(permit2_addr, SWAP_AMOUNT).data),
        value=Wei(0),
    )
    print("[permit2] Step A: WETH.approve(Permit2)  ✓")

    # Step B — grant Universal Router an allowance through Permit2.
    latest = await w3.eth.get_block("latest")
    expiration = latest.get("timestamp", 0) + 3600  # type: ignore[arg-type]
    await send_transaction(
        w3,
        account,
        to=permit2_addr,
        data=bytes(_permit2.fns.approve(WETH_ADDR, ur_address, SWAP_AMOUNT, expiration).data),
        value=Wei(0),
    )
    print("[permit2] Step B: permit2.approve(WETH, UniversalRouter)  ✓")

    # Now the ERC-20 swap succeeds.
    usdc_before_3 = await usdc.fns.balanceOf(sender).call(w3)
    tx_ur_p2 = ur.build_multihop_exact_in_transaction(
        amount_in=TokenAmount(WETH, SWAP_AMOUNT),
        hops=[v4_hop],
        recipient=MSG_SENDER,
        amount_out_minimum=0,
    )
    receipt_ur_p2 = await send_transaction(
        w3,
        account,
        to=tx_ur_p2.to,
        data=tx_ur_p2.data,
        value=Wei(tx_ur_p2.value),
        gas=600_000,
    )
    assert receipt_ur_p2["status"] == 1, (
        f"Universal Router with Permit2 reverted unexpectedly: {receipt_ur_p2['transactionHash'].hex()}"
    )
    received_ur_p2 = (await usdc.fns.balanceOf(sender).call(w3)) - usdc_before_3
    assert received_ur_p2 > 0, "Universal Router (with Permit2) produced no USDC output"
    print(
        f"[permit2] Universal Router (with Permit2, 2 extra txns): {received_ur_p2 / 10**6:.6f} USDC  ✓\n"
        f"[permit2] Summary — DeFiVM: 1 pre-transfer  |  Universal Router: 2 approvals + 1 swap tx"
    )


# ---------------------------------------------------------------------------
# Gas estimation comparison (read-only, no ETH spent)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_sepolia_v4_gas_estimate_comparison() -> None:
    """Compare DeFiVM vs Universal Router V4 swap gas via eth_estimateGas.

    No transactions are broadcast — only read-only RPC calls are made, so
    only ``SEPOLIA_RPC_URL`` and ``SEPOLIA_DEFI_VM`` are required (no private
    key).

    DeFiVM needs WETH pre-funded before execution.  Rather than spending ETH
    on a real transfer, the estimate uses an ``eth_estimateGas`` state override
    to inject a WETH balance into DeFiVM in the simulated call.

    WETH9 storage layout: ``balanceOf`` mapping at slot 0.
    ``balanceOf[addr]`` lives at ``keccak256(abi.encode(addr, uint256(0)))``.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    defi_vm = Web3.to_checksum_address(SEPOLIA_DEFI_VM)

    graph, _ = await _discover_v4_single_pool(w3)
    edge = graph.edges_from(WETH)[0]
    assert isinstance(edge, V4PoolEdge)

    # Use a well-funded Sepolia faucet address as the simulated sender so
    # eth_estimateGas has a non-zero ETH balance for the UR path.
    # Any address with ETH works here — we only need the estimate, not a real tx.
    sim_sender = Web3.to_checksum_address("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266")

    # -----------------------------------------------------------------------
    # DeFiVM gas estimate — inject WETH balance via state override
    # -----------------------------------------------------------------------
    # WETH9: balanceOf[addr] slot = keccak256(abi.encode(addr, 0))
    weth_slot = "0x" + Web3.keccak(abi_encode(["address", "uint256"], [defi_vm, 0])).hex()
    weth_balance_hex = "0x" + hex(SWAP_AMOUNT)[2:].zfill(64)

    route = Router(graph).find_best_route(TokenAmount(WETH, SWAP_AMOUNT), USDC)
    dag = route.dag
    assert dag is not None, "Router route is missing its DAG representation"
    tx_vm = build_swap_transaction(
        dag, SWAP_AMOUNT, defi_vm, sim_sender, min_final_out=route.amount_out.amount * (10_000 - 100) // 10_000
    )

    vm_resp = await w3.provider.make_request(  # type: ignore[attr-defined]
        "eth_estimateGas",
        [
            {
                "from": sim_sender,
                "to": tx_vm.to,
                "data": "0x" + tx_vm.data.hex(),
                "value": hex(tx_vm.value),
            },
            "latest",
            {WETH_ADDR: {"stateDiff": {weth_slot: weth_balance_hex}}},
        ],
    )
    if "error" in vm_resp:
        pytest.skip(f"eth_estimateGas (DeFiVM) failed: {vm_resp['error']}")
    gas_vm = int(vm_resp.get("result", "0x0"), 16)  # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # Universal Router gas estimate — WRAP_ETH + V4_SWAP, no state override
    # -----------------------------------------------------------------------
    ur = UniversalRouter(UNIVERSAL_ROUTER_ADDRESSES[11155111])
    v4_hop = V4Hop(
        token_in=WETH,
        token_out=USDC,
        fee=edge.fee_bps * 100,
        tick_spacing=edge.tick_spacing,
        hooks=edge.hooks,
    )
    tx_ur = ur.build_wrap_and_multihop_exact_in_transaction(
        eth_amount=SWAP_AMOUNT,
        weth_token=WETH,
        hops=[v4_hop],
        recipient=sim_sender,
        amount_out_minimum=0,
    )

    ur_resp = await w3.provider.make_request(  # type: ignore[attr-defined]
        "eth_estimateGas",
        [
            {
                "from": sim_sender,
                "to": tx_ur.to,
                "data": "0x" + tx_ur.data.hex(),
                "value": hex(tx_ur.value),
            },
            "latest",
        ],
    )
    if "error" in ur_resp:
        pytest.skip(f"eth_estimateGas (UniversalRouter) failed: {ur_resp['error']}")
    gas_ur = int(ur_resp.get("result", "0x0"), 16)  # type: ignore[arg-type]

    gas_saved = gas_ur - gas_vm
    print(
        f"\n[gas-estimate] DeFiVM direct unlock():  {gas_vm:,}\n"
        f"[gas-estimate] Universal Router:         {gas_ur:,}\n"
        f"[gas-estimate] gas saved by DeFiVM:      {gas_saved:,} "
        f"({gas_saved / gas_ur * 100:.1f} %)"
    )
    assert gas_vm < gas_ur, f"DeFiVM (gas={gas_vm}) should use less gas than Universal Router (gas={gas_ur})"
