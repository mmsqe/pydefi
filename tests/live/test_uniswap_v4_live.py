"""Live integration tests for the UniswapV4 client and V4PoolEdge pricing.

Against the mainnet WETH/USDC V4 pool (fee=500 pips, tickSpacing=10, no
hooks), these read live state via ``StateView`` and assert that local
``V4PoolEdge`` pricing matches the on-chain ``V4Quoter``, that
``build_swap_route`` emits a V4 step, and that the edge routes through the
pathfinder.
"""

import os

import pytest
from eth_account import Account
from eth_contract import Contract
from eth_contract.erc20 import ERC20
from eth_contract.utils import send_transaction
from web3 import Web3
from web3.types import Wei

from pydefi.amm.uniswap_v4 import UniswapV4
from pydefi.pathfinder.graph import PoolGraph
from pydefi.pathfinder.router import Router
from pydefi.types import ZERO_ADDRESS, TokenAmount
from pydefi.vm.swap import build_swap_transaction
from tests.addrs import (
    UNISWAP_V4_POOL_MANAGER,
    UNISWAP_V4_QUOTER,
    UNISWAP_V4_STATE_VIEW,
    USDC,
    WETH,
)
from tests.live.sepolia_helpers import (
    USDC_SEP,
    USDC_SEPOLIA,
    WETH_SEP,
    WETH_SEPOLIA,
    connect,
    discover_sepolia_v4_weth_usdc_edge,
    require_env,
)

_AMOUNT_IN = 10**16  # 0.01 WETH — small enough to stay within the current tick

# Sepolia broadcast env for the DeFiVM V4 swap test appended at the bottom.
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL", "").strip()
SEPOLIA_PRIVATE_KEY = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
SEPOLIA_DEFI_VM = os.getenv("SEPOLIA_DEFI_VM", "").strip()
_WETH_DEPOSIT = Contract.from_abi(["function deposit() external payable"])


@pytest.fixture
def v4(eth_w3) -> UniswapV4:
    """A UniswapV4 client wired to the mainnet PoolManager / StateView / Quoter."""
    return UniswapV4(
        w3=eth_w3,
        pool_manager_address=UNISWAP_V4_POOL_MANAGER,
        state_view_address=UNISWAP_V4_STATE_VIEW,
        quoter_address=UNISWAP_V4_QUOTER,
    )


@pytest.fixture
def amount_in() -> TokenAmount:
    """0.01 WETH input used across the swap tests."""
    return TokenAmount(token=WETH, amount=_AMOUNT_IN)


@pytest.mark.live
class TestUniswapV4Live:
    """Live on-chain tests for the UniswapV4 client and V4PoolEdge pricing."""

    async def test_local_pricing_matches_quoter(self, v4: UniswapV4, amount_in: TokenAmount):
        """Local V4PoolEdge.amount_out should equal the on-chain V4Quoter output.

        For a small input that stays within the current tick, the inherited
        single-tick V3 math reproduces the quoter exactly.
        """
        edge = await v4.get_pool_edge(WETH, USDC)  # fee=500/tick=10/no-hooks defaults
        assert edge.liquidity > 0
        assert edge.pool_address == UNISWAP_V4_POOL_MANAGER
        assert edge.tick_spacing == 10

        local_out = edge.amount_out(_AMOUNT_IN)
        quoted = await v4.quote_exact_input_single(amount_in, USDC)

        assert quoted.token == USDC
        # 0.01 WETH is ~$5–$200 across any realistic ETH price.
        assert 5 * 10**6 < quoted.amount < 200 * 10**6, f"quote out of range: {quoted.amount / 10**6:.2f} USDC"

        rel_err = abs(local_out - quoted.amount) / quoted.amount
        assert rel_err < 1e-4, f"local={local_out} quoter={quoted.amount} ({rel_err:.6%})"

    async def test_build_swap_route(self, v4: UniswapV4, amount_in: TokenAmount):
        """build_swap_route emits a single V4 step carrying the pool key."""
        route = await v4.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 0
        assert len(route.steps) == 1
        step = route.steps[0]
        assert step.protocol == v4.protocol_name
        assert step.pool_address == UNISWAP_V4_POOL_MANAGER
        assert step.tick_spacing == 10
        assert step.hooks == ZERO_ADDRESS

    async def test_pathfinder_routes_v4_edge(self, v4: UniswapV4, amount_in: TokenAmount):
        """A V4PoolEdge should be routable: find_best_route yields a single V4 hop."""
        graph = PoolGraph()
        graph.add_pool(await v4.get_pool_edge(WETH, USDC))

        route = Router(graph).find_best_route(amount_in, USDC)

        assert len(route.steps) == 1
        assert route.steps[0].protocol == v4.protocol_name
        assert route.amount_out.amount > 0

    async def test_calibration_recovers_static_fee(self, v4: UniswapV4):
        """calibrate_hook_fee against the real quoter recovers the static 500-pip fee.

        The WETH/USDC pool is hookless, so the hook-inclusive effective fee is
        exactly the lpFee: the two probes must agree (linear) and imply ~500.
        """
        edge = await v4.get_pool_edge(WETH, USDC)
        assert not edge.hook_affects_pricing  # hookless pool

        result = await v4.calibrate_hook_fee(edge)

        assert result.linear, f"deviation {result.deviation_pips} pips between probes"
        assert abs(result.implied_fee_pips - 500) <= 5, f"implied {result.implied_fee_pips} pips"
        assert edge.hook_fee_calibrated
        assert abs(edge.lp_fee_pips - 500) <= 5


# ===========================================================================
# Sepolia V4 swap via DeFiVM (ported from swap_route_test, adapted to RouteDAG)
# ===========================================================================
@pytest.mark.live
@pytest.mark.asyncio
async def test_v4_weth_to_usdc_via_defi_vm() -> None:
    """Execute a WETH→USDC swap through a Uniswap V4 pool via DeFiVM on Sepolia.

    V4 unlock/settle pattern:
    1. Sender pre-transfers WETH to DeFiVM.
    2. DeFiVM calls ``PoolManager.unlock(data)`` where *data* encodes the
       PoolKey, SwapParams, and settlement addresses.
    3. PoolManager fires ``unlockCallback(bytes)`` into DeFiVM's ``fallback()``.
    4. DeFiVM calls ``pm.swap()`` → ``pm.sync(tokenIn)`` → ``erc20.transfer`` →
       ``pm.settle()`` → ``pm.take(tokenOut, recipient, amountOut)``.
    5. unlock() returns ``abi.encode(amountOut)`` to the DeFiVM program which
       verifies it meets the minimum slippage requirement.
    """
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY", "SEPOLIA_DEFI_VM")

    w3 = await connect(SEPOLIA_RPC_URL)

    account = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(account.address)
    defi_vm = Web3.to_checksum_address(SEPOLIA_DEFI_VM)

    # -- Step 1: Discover V4 pool and build route -------------------------
    edge = await discover_sepolia_v4_weth_usdc_edge(w3)
    if edge is None:
        pytest.skip("No initialised WETH/USDC V4 pool found on Sepolia — pool may not be seeded")
    print(f"[v4] found: pool_id={edge.pool_id} fee_bps={edge.fee_bps} liquidity={edge.liquidity}")

    # Derive swap amount from the pool's current liquidity depth so the swap
    # stays well within the active tick range regardless of the current price.
    # Virtual token1 (WETH) depth at the current tick: L × sqrtP / 2^96.
    # Using 0.1% of that depth gives ~0.01% price impact — dozens of ticks
    # of headroom to the nearest boundary.
    _Q96 = 2**96
    swap_amount = max(10**12, edge.liquidity * edge.sqrt_price_x96 // _Q96 // 1000)
    print(f"[v4] swap_amount={swap_amount} ({swap_amount / 10**18:.8f} WETH)")

    graph = PoolGraph()
    graph.add_pool(edge)
    route = Router(graph).find_best_route(TokenAmount(WETH_SEP, swap_amount), USDC_SEP)
    assert route is not None, "Router found no route through V4 pool"
    dag = route.dag
    assert dag is not None, "Router route is missing its DAG representation"

    off_chain_out = route.amount_out.amount
    print(f"[v4] off-chain estimate: {off_chain_out / 10**6:.6f} USDC  route={route!r}")
    assert off_chain_out > 0, "off-chain amountOut estimate is zero"

    # (quote_swap_transaction raises NotImplementedError for V4; skip quote step)

    # -- Step 2: Pre-fund DeFiVM with WETH --------------------------------
    # V4 unlock/settle requires DeFiVM to already hold tokenIn; unlike V3 there
    # is no flash-swap that defers repayment to a callback.
    weth = ERC20(to=WETH_SEPOLIA)
    weth_balance = await weth.fns.balanceOf(sender).call(w3)
    if weth_balance < swap_amount:
        weth_deposit = _WETH_DEPOSIT(to=WETH_SEPOLIA)
        await send_transaction(
            w3,
            account,
            to=WETH_SEPOLIA,
            data=bytes(weth_deposit.fns.deposit().data),
            value=Wei(swap_amount),
        )
        print(f"[v4] wrapped {swap_amount} wei → WETH")

    await send_transaction(
        w3,
        account,
        to=WETH_SEPOLIA,
        data=bytes(weth.fns.transfer(defi_vm, swap_amount).data),
        value=Wei(0),
    )
    print(f"[v4] transferred {swap_amount} WETH to DeFiVM ({defi_vm})")

    # -- Step 3: Build and broadcast execute(bytes) -----------------------
    usdc = ERC20(to=USDC_SEPOLIA)
    usdc_before = await usdc.fns.balanceOf(sender).call(w3)

    min_final_out = off_chain_out * (10_000 - 100) // 10_000  # 1 % slippage
    tx = build_swap_transaction(dag, swap_amount, defi_vm, sender, min_final_out=min_final_out)
    print(f"[v4] executing unlock() program  min_out={min_final_out / 10**6:.6f} USDC")

    receipt = await send_transaction(
        w3,
        account,
        to=tx.to,
        data=tx.data,
        value=Wei(tx.value),
        gas=600_000,
    )
    tx_hash = receipt["transactionHash"].hex()
    print(f"[v4] tx {tx_hash} status={receipt['status']}")
    assert receipt["status"] == 1, f"DeFiVM V4 swap reverted: {tx_hash}"

    # -- Step 4: Verify USDC received -------------------------------------
    usdc_after = await usdc.fns.balanceOf(sender).call(w3)
    received = usdc_after - usdc_before
    print(f"[v4] received {received / 10**6:.6f} USDC  (min={min_final_out / 10**6:.6f})")
    assert received >= min_final_out, f"received {received} USDC < min_final_out {min_final_out}  tx={tx_hash}"
