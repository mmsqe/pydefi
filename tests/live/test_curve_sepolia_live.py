"""Live Curve checks against real pools on the Sepolia testnet.

Read-only (no broadcasts): loads pool state over ``SEPOLIA_RPC_URL`` and
cross-checks the local :class:`~pydefi.amm.curve.CurvePool` /
:mod:`~pydefi.amm.curve_math` against the deployed pools' own on-chain views,
exercising the Curve client end-to-end on a non-mainnet chain.  Mirrors
:mod:`tests.live.test_curve_live` (mainnet); bit-exact equivalence vs deployed
bytecode is covered by :mod:`tests.live.test_curve_boa`.

A relative tolerance is used rather than exact equality because ``load_state``
and the reference call may land on adjacent Sepolia blocks on a public RPC.
The Sepolia stableswap factory only deploys Stable-NG; legacy/factory-plain
kinds, meta-pools, registry discovery and the math error paths have no testnet
deployment and are exercised here from constructed state instead.

Run: ``SEPOLIA_RPC_URL=... pytest tests/live/test_curve_sepolia_live.py -m live``
"""

import os

import pytest
from web3 import AsyncWeb3

from pydefi.abi.amm import CURVE_POOL, CURVE_V2_POOL
from pydefi.amm import curve_math as m
from pydefi.amm.curve import CurveMetaPool, CurvePool, CurvePoolKind
from pydefi.exceptions import InsufficientLiquidityError, PoolDataError
from pydefi.types import Address, ChainId, Token, TokenAmount
from tests.live.sepolia_helpers import require_env

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL", "").strip()

# Curve Stable-NG factory pool on Sepolia (coins: bUSDC(6), bIBTA(18)).
CURVE_NG_SEPOLIA = "0xa8e58EbC415EEBde0A47dF5A47F8E034Cbb4aB5a"
BUSDC = Token(
    chain_id=ChainId.SEPOLIA, address=Address("0x59fb67f6778cFf089484cF7115906725DFc44293"), symbol="bUSDC", decimals=6
)
BIBTA = Token(
    chain_id=ChainId.SEPOLIA, address=Address("0x51A4Db078859F5a4835A7b184740eD033F682A5b"), symbol="bIBTA", decimals=18
)

# Curve twocrypto-ng factory pool on Sepolia (both coins 18-dec).  Like
# tricrypto-NG it solves y analytically, so local newton_y agrees to ~1e-10.
CURVE_CRYPTO_SEPOLIA = "0x004C167d27ADa24305b76D80762997Fa6EB8d9B2"
CRA = Token(
    chain_id=ChainId.SEPOLIA,
    address=Address("0x2Dc5a9D152dA35581F523a2c077dca079c05bf29"),
    symbol="ePybETH",
    decimals=18,
)
CRB = Token(
    chain_id=ChainId.SEPOLIA, address=Address("0x1f4F5e56aB7e76262d7A07F24a8A5b33058ae600"), symbol="TriPT", decimals=18
)


@pytest.fixture
async def sepolia_w3() -> AsyncWeb3:
    require_env("SEPOLIA_RPC_URL")
    return AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(SEPOLIA_RPC_URL))


@pytest.fixture
async def ng_pool(sepolia_w3) -> CurvePool:
    pool = CurvePool(w3=sepolia_w3, pool_address=CURVE_NG_SEPOLIA, tokens=[BUSDC, BIBTA], kind=CurvePoolKind.STABLE_NG)
    await pool.load_state()
    return pool


@pytest.fixture
async def crypto_pool(sepolia_w3) -> CurvePool:
    pool = CurvePool(w3=sepolia_w3, pool_address=CURVE_CRYPTO_SEPOLIA, tokens=[CRA, CRB], kind=CurvePoolKind.CRYPTO_V2)
    await pool.load_state()
    return pool


def _rel(a: int, b: int) -> float:
    return abs(a - b) / b if b else 0.0


# ===========================================================================
# Live — Stable-NG pool
# ===========================================================================


@pytest.mark.live
class TestCurveSepoliaStableNG:
    async def test_detect_kind(self, sepolia_w3):
        assert await CurvePool.detect_kind(sepolia_w3, CURVE_NG_SEPOLIA) == CurvePoolKind.STABLE_NG

    async def test_load_state_shape(self, ng_pool):
        assert ng_pool.has_local_state and ng_pool.kind is CurvePoolKind.STABLE_NG
        assert ng_pool.protocol_name == "Curve"
        assert [t.symbol for t in ng_pool.tokens] == ["bUSDC", "bIBTA"]
        assert ng_pool._state["ng_d_form"] and ng_pool._state["offpeg_fee_multiplier"] > 0

    async def test_get_dy_local_matches_onchain(self, ng_pool, sepolia_w3):
        for src, dst, i, j in ((BUSDC, BIBTA, 0, 1), (BIBTA, BUSDC, 1, 0)):
            dx = 10**src.decimals
            local = ng_pool.get_dy_local(src, dst, dx)
            onchain = await CURVE_POOL.fns.get_dy(i, j, dx).call(sepolia_w3, to=CURVE_NG_SEPOLIA)
            assert local > 0 and onchain > 0 and _rel(local, onchain) < 1e-4, f"({i}->{j}) {local} vs {onchain}"

    async def test_get_dx_local_matches_onchain(self, ng_pool, sepolia_w3):
        dy = 10**BIBTA.decimals
        local = ng_pool.get_dx_local(BUSDC, BIBTA, dy)
        onchain = await CURVE_POOL.fns.get_dx(0, 1, dy).call(sepolia_w3, to=CURVE_NG_SEPOLIA)
        assert local > 0 and _rel(local, onchain) < 1e-4, f"{local} vs {onchain}"

    async def test_get_dy_prefers_local(self, ng_pool, sepolia_w3):
        dx = 5 * 10**BUSDC.decimals
        assert await ng_pool.get_dy(BUSDC, BIBTA, dx) == ng_pool.get_dy_local(BUSDC, BIBTA, dx)

    async def test_amounts_out_and_in_local(self, ng_pool):
        out = await ng_pool.get_amounts_out(TokenAmount(BUSDC, 10**BUSDC.decimals), [BUSDC, BIBTA])
        assert out[1].token == BIBTA and out[1].amount > 0
        inp = await ng_pool.get_amounts_in(TokenAmount(BIBTA, 10**BIBTA.decimals), [BUSDC, BIBTA])
        assert inp[0].token == BUSDC and inp[0].amount > 0

    async def test_build_swap_route_fee_in_bps(self, ng_pool):
        route = await ng_pool.build_swap_route(TokenAmount(BUSDC, 10**BUSDC.decimals), BIBTA)
        assert route.steps[0].protocol == "Curve"
        assert route.steps[0].fee == ng_pool._state["fee"] * 10_000 // m.FEE_DENOMINATOR

    async def test_build_exchange_tx_executes_shape(self, ng_pool):
        tx = ng_pool.build_exchange_tx(TokenAmount(BUSDC, 10**BUSDC.decimals), BIBTA, slippage_bps=50)
        assert tx["to"] == CURVE_NG_SEPOLIA and tx["value"] == "0"
        i, j, _dx, _min = CURVE_POOL.fns.exchange.decode_input(bytes.fromhex(tx["data"][2:]))
        assert (i, j) == (0, 1)

    async def test_legacy_loader_path(self, sepolia_w3):
        # Sepolia has no a_precision=1 pool, but the legacy _load_stable_state
        # branch (A()/fee()/decimal rates) runs against any pool exposing them.
        legacy = CurvePool(
            w3=sepolia_w3, pool_address=CURVE_NG_SEPOLIA, tokens=[BUSDC, BIBTA], kind=CurvePoolKind.STABLE_LEGACY
        )
        state = await legacy.load_state()
        assert state["a_precision"] == 1 and state["legacy_fee_order"] and legacy.has_local_state

    async def test_onchain_fallback_without_state(self, sepolia_w3):
        # No load_state → get_dy / get_amounts_in hit the on-chain views.
        fresh = CurvePool(
            w3=sepolia_w3, pool_address=CURVE_NG_SEPOLIA, tokens=[BUSDC, BIBTA], kind=CurvePoolKind.STABLE_NG
        )
        dx = 10**BUSDC.decimals
        onchain = await fresh.get_dy(BUSDC, BIBTA, dx)
        assert onchain > 0
        amounts_in = await fresh.get_amounts_in(TokenAmount(BIBTA, 10**BIBTA.decimals // 2), [BUSDC, BIBTA])
        assert amounts_in[0].amount > 0


# ===========================================================================
# Live — twocrypto (Curve V2) pool
# ===========================================================================


@pytest.mark.live
class TestCurveSepoliaCrypto:
    async def test_detect_kind(self, sepolia_w3):
        assert await CurvePool.detect_kind(sepolia_w3, CURVE_CRYPTO_SEPOLIA) == CurvePoolKind.CRYPTO_V2

    async def test_get_dy_local_matches_onchain(self, crypto_pool, sepolia_w3):
        assert crypto_pool.protocol_name == "CurveV2"
        for i, j, src, dst in ((0, 1, CRA, CRB), (1, 0, CRB, CRA)):
            dx = 10**src.decimals
            local = crypto_pool.get_dy_local(src, dst, dx)
            onchain = await CURVE_V2_POOL.fns.get_dy(i, j, dx).call(sepolia_w3, to=CURVE_CRYPTO_SEPOLIA)
            assert local > 0 and _rel(local, onchain) < 1e-8, f"({i}->{j}) {local} vs {onchain}"

    async def test_get_dx_local_search(self, crypto_pool, sepolia_w3):
        # Cryptoswap has no closed-form get_dx; CurvePool binary-searches get_dy.
        dy = crypto_pool.get_dy_local(CRA, CRB, 10**CRA.decimals)
        dx = crypto_pool.get_dx_local(CRA, CRB, dy)
        assert crypto_pool.get_dy_local(CRA, CRB, dx) >= dy

    async def test_get_dx_local_over_capacity_raises(self, crypto_pool):
        with pytest.raises(InsufficientLiquidityError):
            crypto_pool.get_dx_local(CRA, CRB, 10**30)

    async def test_build_exchange_tx_v2_abi(self, crypto_pool):
        tx = crypto_pool.build_exchange_tx(TokenAmount(CRA, 10**CRA.decimals), CRB, min_amount_out=1)
        i, j, _dx, mn = CURVE_V2_POOL.fns.exchange.decode_input(bytes.fromhex(tx["data"][2:]))
        assert (i, j, mn) == (0, 1, 1)


# ===========================================================================
# Live — discovery helper and on-chain error path
# ===========================================================================


@pytest.mark.live
class TestCurveSepoliaClientPaths:
    async def test_discover_tokens_reads_coins(self, sepolia_w3):
        # _discover_tokens reads the pool's coins; pass only one known token so
        # the other is built from its ERC-20 decimals/symbol on-chain.
        tokens = await CurvePool._discover_tokens(sepolia_w3, CURVE_NG_SEPOLIA, [BUSDC])
        assert [t.address for t in tokens] == [BUSDC.address, BIBTA.address]
        assert tokens[0] is BUSDC and tokens[1].decimals == 18  # second built from chain

    async def test_get_dy_onchain_revert_wrapped(self, sepolia_w3):
        # A pool address with no Curve get_dy reverts → InsufficientLiquidityError.
        bad = CurvePool(w3=sepolia_w3, pool_address="0x59fb67f6778cFf089484cF7115906725DFc44293", tokens=[BUSDC, BIBTA])
        with pytest.raises(InsufficientLiquidityError):
            await bad.get_dy(BUSDC, BIBTA, 10**6)


# ===========================================================================
# Constructed-state coverage — paths with no Sepolia deployment
# (legacy / factory-plain kinds, meta-pools, crypto newton_D, error branches).
# Marked `live` to group with this file; these need no network.
# ===========================================================================

_ADDR = "0x" + "11" * 20
_DAI = Token(chain_id=ChainId.SEPOLIA, address=Address("0x" + "d0" * 20), symbol="DAI", decimals=18)
_USDC = Token(chain_id=ChainId.SEPOLIA, address=Address("0x" + "c0" * 20), symbol="USDC", decimals=6)
_USDT = Token(chain_id=ChainId.SEPOLIA, address=Address("0x" + "d7" * 20), symbol="USDT", decimals=6)
_WETH = Token(chain_id=ChainId.SEPOLIA, address=Address("0x" + "ee" * 20), symbol="WETH", decimals=18)

_LEGACY = {  # 3pool-style: a_precision=1, fee taken in coin units
    "balances": [100_000_000 * 10**18, 100_000_000 * 10**6, 100_000_000 * 10**6],
    "rates": [10**18, 10**30, 10**30],
    "amp": 2000,
    "fee": 1_000_000,
    "a_precision": 1,
    "ng_d_form": False,
    "offpeg_fee_multiplier": 0,
    "legacy_fee_order": True,
}
_FACTORY = {  # A_PRECISION, classic D form, flat fee
    "balances": [50_000_000 * 10**18, 50_000_000 * 10**6],
    "rates": [10**18, 10**30],
    "amp": 200_000,
    "fee": 1_000_000,
    "a_precision": 100,
    "ng_d_form": False,
    "offpeg_fee_multiplier": 0,
    "legacy_fee_order": False,
}
_CRYPTO = {
    "balances": [10_000_000 * 10**6, 100 * 10**8, 3000 * 10**18],
    "precisions": [10**12, 10**10, 10**0],
    "price_scale": [10**18, 100_000 * 10**18, 3333 * 10**18],
    "amp": 1_707_629,
    "gamma": 11_809_167_828_997,
    "mid_fee": 3_000_000,
    "out_fee": 30_000_000,
    "fee_gamma": 500_000_000_000_000,
}
_META_BASE = {**_LEGACY, "total_supply": 300_000_000 * 10**18}
_META = {
    "balances": [50_000_000 * 10**18, 50_000_000 * 10**18],
    "rates": [10**18, 10**18],
    "amp": 200_000,
    "fee": 1_000_000,
    "a_precision": 100,
    "ng_d_form": False,
}
_NG_BASE = {
    **_FACTORY,
    "ng_d_form": True,
    "offpeg_fee_multiplier": 20_000_000_000,
    "total_supply": 100_000_000 * 10**18,
}
_NG_META = {**_META, "ng_d_form": True, "offpeg_fee_multiplier": 20_000_000_000}


def _stub(kind: CurvePoolKind, state: dict, tokens: list[Token]) -> CurvePool:
    pool = CurvePool(w3=None, pool_address=_ADDR, tokens=tokens, kind=kind)
    pool._state = dict(state)
    return pool


@pytest.mark.live
class TestCurveConstructedCoverage:
    """Exercises kinds / paths Sepolia does not deploy, from injected state."""

    def test_stable_legacy_and_factory_dy_dx(self):
        # legacy_fee_order branch + a_precision=1, and the factory flat-fee path.
        legacy = _stub(CurvePoolKind.STABLE_LEGACY, _LEGACY, [_DAI, _USDC, _USDT])
        assert legacy.get_dy_local(_DAI, _USDC, 1000 * 10**18) > 990 * 10**6
        assert legacy.get_dx_local(_DAI, _USDC, 1000 * 10**6) > 0
        assert m.stable_get_dy(0, 1, 1000 * 10**18, **_FACTORY) > 990 * 10**6
        assert m.stable_get_dx(0, 1, 1000 * 10**6, **_FACTORY) > 0

    def test_stable_get_D_empty_and_get_y_errors(self):
        assert m.stable_get_D([0, 0, 0], 2000, a_precision=1) == 0
        xp = m._xp_mem(_LEGACY["rates"], _LEGACY["balances"])
        d = m.stable_get_D(xp, 2000, a_precision=1)
        with pytest.raises(ValueError):
            m.stable_get_y(0, 0, xp[0], xp, 2000, d, a_precision=1)
        with pytest.raises(ValueError):
            m.stable_get_y(0, 5, xp[0], xp, 2000, d, a_precision=1)

    def test_stable_get_dx_infeasible_returns_zero(self):
        # dy larger than the pool can give → solved y < 0 → 0.
        assert m.stable_get_dx(0, 1, 10**40, **_FACTORY) == 0

    def test_stable_dy_zero_output_and_flat_fee(self):
        # dx=0 → dy = xp[j] - y - 1 < 0 → 0 (the dy < 0 guard).
        assert m.stable_get_dy(0, 1, 0, **_FACTORY) == 0
        # fee_multiplier <= FEE_DENOMINATOR → flat fee branch.
        assert m.stable_dynamic_fee(100, 100, 4_000_000, m.FEE_DENOMINATOR) == 4_000_000

    def test_stable_ng_dy_dx_offpeg_fee(self):
        # NG config (offpeg_fee_multiplier > 0) drives the dynamic-fee branch in
        # both get_dy and get_dx without needing the Sepolia round-trip.
        ng = {k: _NG_BASE[k] for k in _FACTORY}
        assert m.stable_get_dy(0, 1, 1000 * 10**18, **ng) > 0
        assert m.stable_get_dx(0, 1, 1000 * 10**6, **ng) > 0

    def test_crypto_newton_D_geomean_fee_and_recompute(self):
        xp = [
            _CRYPTO["balances"][k] * _CRYPTO["precisions"][k] * _CRYPTO["price_scale"][k] // m.PRECISION
            for k in range(3)
        ]
        assert m.newton_D(_CRYPTO["amp"], _CRYPTO["gamma"], xp) > 0
        assert m._geometric_mean(sorted(xp, reverse=True)) > 0
        assert _CRYPTO["mid_fee"] <= m.crypto_fee(xp, _CRYPTO["mid_fee"], _CRYPTO["out_fee"], _CRYPTO["fee_gamma"])
        # d=None forces the newton_D recompute branch inside crypto_get_dy.
        assert m.crypto_get_dy(0, 2, 10_000 * 10**6, d=None, **_CRYPTO) > 0

    def test_meta_underlying_factory_base(self):
        # factory meta over legacy base: deposit (½-fee), withdraw, and the
        # both-legs-in-base delegate path.
        assert m.meta_get_dy_underlying(0, 1, 1000 * 10**18, _META, _META_BASE) > 990 * 10**18
        assert m.meta_get_dy_underlying(2, 0, 1000 * 10**6, _META, _META_BASE) > 990 * 10**18
        assert m.meta_get_dy_underlying(1, 2, 1000 * 10**18, _META, _META_BASE) == m.stable_get_dy(
            0, 1, 1000 * 10**18, **{k: _META_BASE[k] for k in _FACTORY}
        )

    def test_meta_underlying_ng_base(self):
        # NG meta over NG base: fee-inclusive ng mint + ng dynamic-fee withdraw.
        assert m.meta_get_dy_underlying(0, 1, 1000 * 10**18, _NG_META, _NG_BASE) > 980 * 10**6
        assert m.meta_get_dy_underlying(2, 0, 1000 * 10**6, _NG_META, _NG_BASE) > 980 * 10**18

    def test_curve_meta_pool_methods(self):
        base = CurvePool(w3=None, pool_address=_ADDR, tokens=[_DAI, _USDC, _USDT])
        meta = CurveMetaPool(w3=None, pool_address=_ADDR, primary_token=_WETH, base_pool=base)
        assert meta.protocol_name == "Curve"
        meta._meta, meta._base = dict(_META), dict(_META_BASE)
        assert meta.get_dy_underlying(_WETH, _USDC, 1000 * 10**18) > 0
        tx = meta.build_exchange_underlying_tx(TokenAmount(_WETH, 10**18), _USDC, min_amount_out=7)
        i, j, _dx, mn = CURVE_POOL.fns.exchange_underlying.decode_input(bytes.fromhex(tx["data"][2:]))
        assert (i, j, mn) == (0, 2, 7)
        # min_amount_out=None → derived from get_dy_underlying + slippage.
        auto = meta.build_exchange_underlying_tx(TokenAmount(_WETH, 10**18), _USDC, slippage_bps=100)
        *_ignore, auto_min = CURVE_POOL.fns.exchange_underlying.decode_input(bytes.fromhex(auto["data"][2:]))
        assert auto_min > 0
        with pytest.raises(ValueError):
            meta._underlying_index(BIBTA)  # token not in this meta-pool

    def test_curve_meta_pool_requires_state(self):
        base = CurvePool(w3=None, pool_address=_ADDR, tokens=[_DAI, _USDC])
        meta = CurveMetaPool(w3=None, pool_address=_ADDR, primary_token=_WETH, base_pool=base)
        with pytest.raises(PoolDataError):
            meta.get_dy_underlying(_WETH, _DAI, 10**18)

    def test_curve_pool_error_and_util_branches(self):
        empty = CurvePool(w3=None, pool_address=_ADDR, tokens=[_DAI, _USDC])
        with pytest.raises(PoolDataError):
            empty.get_dy_local(_DAI, _USDC, 10**18)
        with pytest.raises(PoolDataError):
            empty.get_dx_local(_DAI, _USDC, 10**6)
        legacy = _stub(CurvePoolKind.STABLE_LEGACY, _LEGACY, [_DAI, _USDC, _USDT])
        with pytest.raises(ValueError):
            legacy.get_dy_local(_DAI, _WETH, 10**18)  # _WETH not a coin → _coin_index raise
        assert legacy.get_registry_contract("0x" + "ab" * 20) is not None

    async def test_get_amounts_path_length_validation(self):
        legacy = _stub(CurvePoolKind.STABLE_LEGACY, _LEGACY, [_DAI, _USDC, _USDT])
        with pytest.raises(ValueError):
            await legacy.get_amounts_out(TokenAmount(_DAI, 10**18), [_DAI])
        with pytest.raises(ValueError):
            await legacy.get_amounts_in(TokenAmount(_USDC, 10**6), [_DAI, _USDC, _USDT])
