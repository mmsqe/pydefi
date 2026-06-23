"""Broadcast (sign + send) real Curve swaps on the Sepolia testnet.

Unlike :mod:`tests.live.test_curve_sepolia_live` (read-only price checks), these
tests submit REAL transactions and assert the swap executed on-chain within
slippage of the local quote — the end-to-end execution proof the read-only and
boa-fork suites cannot give: real signing, mempool, gas estimation and state
mutation, exercising :meth:`CurvePool.build_exchange_tx` for both the int128
stableswap and uint256 cryptoswap ABIs.

These are **self-funding**: the signer only needs Sepolia ETH for gas.  The
input coin is acquired permissionlessly per pool —

* USDe/USDC Stable-NG pool: USDC is the **Aave Sepolia faucet** test token,
  minted via ``mint(token, to, amount)`` on 0xC959…3f42D.
* WETH/USDC twocrypto pool: WETH is obtained by wrapping ETH (``deposit()``).

(Curve's other Sepolia pools use minter-gated Backed stablecoins or an expired
Napier principal token — none acquirable by a fresh key — so they are avoided.)

Gated to opt-in by ``tests/conftest.py`` (the ``testnet`` marker is skipped
unless ``-m testnet`` is explicitly passed), so a bare ``pytest`` never
broadcasts.  Run::

    SEPOLIA_RPC_URL=... SEPOLIA_PRIVATE_KEY=<funded with Sepolia ETH> \\
        pytest tests/live/test_curve_sepolia_execution.py -m testnet
"""

import os

import pytest
from eth_account import Account
from eth_contract import ERC20, Contract
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3

from pydefi.amm.curve import CurvePool, CurvePoolKind
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.utils import send_signed, tx_data_bytes
from tests.live.sepolia_helpers import require_env

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL", "").strip()
SEPOLIA_PRIVATE_KEY = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
SEPOLIA_CHAIN_ID = 11155111
SLIPPAGE_BPS = 200  # 2% min-out floor and payout tolerance
MIN_GAS_ETH = 3 * 10**15  # ~0.003 ETH headroom for faucet + approve + exchange


def _t(addr: str, symbol: str, decimals: int) -> Token:
    return Token(chain_id=ChainId.SEPOLIA, address=Address(addr), symbol=symbol, decimals=decimals)


# Stable-NG pool USDe(0)/USDC(1); USDC is the Aave Sepolia faucet token (deep, ~1.1M each).
CURVE_USDE_USDC = "0x9ffA772BBc36a03CFf60A3088E40e9207f4F233A"
USDE = _t("0x45bfdad339Eb9f0636Eb7bB586F2a79dD9ddFF2c", "USDe", 18)
AAVE_USDC = _t("0x94a9D9AC8a22534E3FaCa9F4e7F2E2cf85d5E4C8", "USDC", 6)
AAVE_FAUCET = "0xC959483DBa39aa9E78757139af0e9a2EDEb3f42D"

# Twocrypto pool USDC(0, Circle/gated)/WETH(1); WETH obtained by wrapping ETH.
CURVE_WETH_USDC = "0x011E998D2d794424DE95935d55a6Ca81822Ecb2b"
CIRCLE_USDC = _t("0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238", "USDC", 6)  # coin0 (output only)
WETH_SEP = _t("0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14", "WETH", 18)

_FAUCET = Contract.from_abi(["function mint(address token, address to, uint256 amount) external returns (uint256)"])
_WETH = Contract.from_abi(["function deposit() external payable"])


def _ck(token: Token) -> str:
    return Web3.to_checksum_address(token.address)


async def _connect() -> AsyncWeb3:
    require_env("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY")
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(SEPOLIA_RPC_URL))
    if await w3.eth.chain_id != SEPOLIA_CHAIN_ID:
        pytest.skip("SEPOLIA_RPC_URL is not pointed at Sepolia")
    return w3


async def _balance(w3: AsyncWeb3, token: Token, owner: str) -> int:
    return int(await ERC20.fns.balanceOf(owner).call(w3, to=_ck(token)))


async def _send(w3: AsyncWeb3, sender: str, to: str, data: bytes, value: int = 0) -> None:
    # "pending" nonce so a just-mined prior tx is counted in this sequence.
    nonce = await w3.eth.get_transaction_count(sender, "pending")
    _h, status = await send_signed(
        w3, private_key=SEPOLIA_PRIVATE_KEY, sender=sender, to=to, data=data, nonce=nonce, value=value
    )
    assert status == 1, f"tx to {to} reverted"


async def _self_fund(w3: AsyncWeb3, sender: str, token_in: Token, need: int) -> None:
    """Acquire *need* of *token_in* permissionlessly (Aave faucet or wrap ETH)."""
    if await _balance(w3, token_in, sender) >= need:
        return
    if token_in.address == AAVE_USDC.address:
        await _send(w3, sender, AAVE_FAUCET, tx_data_bytes(_FAUCET.fns.mint(_ck(token_in), sender, need).data))
    elif token_in.address == WETH_SEP.address:
        await _send(w3, sender, _ck(token_in), tx_data_bytes(_WETH.fns.deposit().data), value=need)
    else:  # pragma: no cover - defensive
        pytest.skip(f"no permissionless funding path for {token_in.symbol}")


async def _execute_swap(w3: AsyncWeb3, pool: CurvePool, token_in: Token, token_out: Token, dx: int) -> None:
    account = Account.from_key(SEPOLIA_PRIVATE_KEY)
    sender = Web3.to_checksum_address(account.address)

    # Native ETH covers gas (and the WETH wrap when token_in is WETH).
    wrap = dx if token_in.address == WETH_SEP.address else 0
    if await w3.eth.get_balance(sender) < MIN_GAS_ETH + wrap:
        pytest.skip("signer underfunded with Sepolia ETH — top up via a faucet")

    await _self_fund(w3, sender, token_in, dx)
    assert await _balance(w3, token_in, sender) >= dx, "self-funding did not yield enough input"

    await pool.load_state()
    quote = pool.get_dy_local(token_in, token_out, dx)
    assert quote > 0, "local quote is zero"
    min_out = quote * (10_000 - SLIPPAGE_BPS) // 10_000

    # Curve exchange pulls token_in via transferFrom → approve the pool.
    if int(await ERC20.fns.allowance(sender, pool.router_address).call(w3, to=_ck(token_in))) < dx:
        await _send(w3, sender, _ck(token_in), tx_data_bytes(ERC20.fns.approve(pool.router_address, dx).data))

    out_before = await _balance(w3, token_out, sender)
    tx = pool.build_exchange_tx(TokenAmount(token_in, dx), token_out, min_amount_out=min_out)
    await _send(w3, sender, tx["to"], bytes(HexBytes(tx["data"])), value=int(tx["value"]))

    received = await _balance(w3, token_out, sender) - out_before
    assert received >= min_out, f"received {received} < min_out {min_out}"
    assert abs(received - quote) <= quote * SLIPPAGE_BPS // 10_000, f"received {received} vs quote {quote}"


@pytest.mark.testnet
class TestCurveSepoliaExecution:
    """Real, self-funding approve+exchange broadcasts on Sepolia Curve pools."""

    async def test_execute_stable_ng_swap(self):
        # int128 exchange ABI; USDC (Aave faucet) → USDe on the deep NG pool.
        w3 = await _connect()
        pool = CurvePool(w3=w3, pool_address=CURVE_USDE_USDC, tokens=[USDE, AAVE_USDC], kind=CurvePoolKind.STABLE_NG)
        await _execute_swap(w3, pool, AAVE_USDC, USDE, dx=100 * 10**6)  # 100 USDC

    async def test_execute_crypto_v2_swap(self):
        # uint256 exchange ABI; WETH (wrapped ETH) → USDC on a small twocrypto pool.
        w3 = await _connect()
        pool = CurvePool(
            w3=w3, pool_address=CURVE_WETH_USDC, tokens=[CIRCLE_USDC, WETH_SEP], kind=CurvePoolKind.CRYPTO_V2
        )
        await pool.load_state()
        # Pool WETH liquidity is small; cap dx well under it to keep impact sane.
        weth_reserve = pool._state["balances"][1]
        if weth_reserve < 10**15:
            pytest.skip(f"twocrypto WETH reserve too low ({weth_reserve}) for a safe broadcast")
        await _execute_swap(w3, pool, WETH_SEP, CIRCLE_USDC, dx=min(10**14, weth_reserve // 20))
