from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .builder import DeFiBuilder


class ExtractionMode(str, Enum):
    RET_LAST32 = "ret_last32"
    RET_U256 = "ret_u256"
    RET_SLICE = "ret_slice"


@dataclass(frozen=True)
class ExtractionPlan:
    mode: ExtractionMode
    offset: int | None = None
    size: int | None = None


@dataclass(frozen=True)
class PatchPlan:
    calldata: bytes
    patch_offsets: list[int]
    placeholder_value: int


@dataclass(frozen=True)
class AutoPlan:
    swap: PatchPlan
    bridge: PatchPlan
    extraction: ExtractionPlan


BridgeAmountDecoder = Callable[[bytes], tuple[int, int] | None]


class Analyzer:
    """Generic LiFi-style planner for a two-step eDSL flow.

    The planner focuses on a minimal compose model:
    1) swap call
    2) bridge call that consumes amount extracted from swap returndata

    It intentionally avoids protocol-specific inference and keeps fallback
    behavior predictable for PoC and integration tests.
    """

    @staticmethod
    def _to_uint256_bytes(value: int) -> bytes:
        if value < 0:
            raise ValueError("uint256 value must be non-negative")
        return value.to_bytes(32, "big")

    @staticmethod
    def _find_u256_offsets(buf: bytes, value: int) -> list[int]:
        marker = Analyzer._to_uint256_bytes(value)
        offsets: list[int] = []
        start = 0
        while True:
            idx = buf.find(marker, start)
            if idx < 0:
                break
            offsets.append(idx)
            start = idx + 1
        return offsets

    @staticmethod
    def _selector(buf: bytes) -> bytes:
        if len(buf) < 4:
            return b""
        return buf[:4]

    @staticmethod
    def _as_calldata(raw: bytes | str | dict[str, Any]) -> bytes:
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            return bytes.fromhex(raw.removeprefix("0x"))
        if isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, bytes):
                return data
            if isinstance(data, str):
                return bytes.fromhex(data.removeprefix("0x"))
        raise ValueError("unsupported tx payload for calldata extraction")

    @staticmethod
    def _as_tx_payload(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            if isinstance(raw.get("tx_data"), dict):
                return raw["tx_data"]
            return raw
        tx_data = getattr(raw, "tx_data", None)
        if isinstance(tx_data, dict):
            return tx_data
        raise ValueError("expected tx payload dict or object with tx_data dict")

    @staticmethod
    def _extract_amount_in(raw: Any) -> int | None:
        if isinstance(raw, dict):
            value = raw.get("amount_in")
            if isinstance(value, int):
                return value
            if isinstance(value, dict):
                nested_amount = value.get("amount")
                if isinstance(nested_amount, int):
                    return nested_amount

        if hasattr(raw, "amount_in"):
            amount_in_obj = getattr(raw, "amount_in")
            if isinstance(amount_in_obj, int):
                return amount_in_obj

            amount_value = getattr(amount_in_obj, "amount", None)
            if isinstance(amount_value, int):
                return amount_value

        return None

    @staticmethod
    def _resolve_extraction(
        *,
        extraction_mode: ExtractionMode,
        extraction_offset: int | None,
        extraction_size: int | None,
    ) -> ExtractionPlan:
        def _require_uint16(name: str, value: int | None) -> int:
            if value is None:
                raise ValueError(f"{name} is required for {extraction_mode.name} mode")
            if value < 0 or value > 0xFFFF:
                raise ValueError(f"{name} must fit in uint16, got {value}")
            return value

        if extraction_mode == ExtractionMode.RET_U256:
            offset = _require_uint16("extraction_offset", extraction_offset)
            return ExtractionPlan(
                mode=ExtractionMode.RET_U256,
                offset=offset,
            )

        if extraction_mode == ExtractionMode.RET_SLICE:
            offset = _require_uint16("extraction_offset", extraction_offset)
            size = _require_uint16("extraction_size", extraction_size)
            return ExtractionPlan(
                mode=ExtractionMode.RET_SLICE,
                offset=offset,
                size=size,
            )

        if extraction_offset is not None or extraction_size is not None:
            raise ValueError("extraction_offset/extraction_size require RET_U256 or RET_SLICE mode")

        return ExtractionPlan(mode=ExtractionMode.RET_LAST32)

    @staticmethod
    def _validate_unique_offsets(*, offsets: list[int], role: str, allow_empty: bool) -> list[int]:
        if not offsets:
            if allow_empty:
                return []
            raise ValueError(f"expected exactly one {role} amount placeholder, found []")

        unique = sorted(set(offsets))
        if len(unique) != 1:
            raise ValueError(
                f"expected exactly one {role} amount placeholder, found {unique}; pass explicit template/placeholder"
            )
        return unique

    def analyze_from_quotes(
        self,
        *,
        swap_quote: Any,
        bridge_tx: Any,
        extraction_mode: ExtractionMode = ExtractionMode.RET_LAST32,
        extraction_offset: int | None = None,
        extraction_size: int | None = None,
        bridge_amount_placeholder: int | None = None,
    ) -> AutoPlan:
        swap_tx = self._as_tx_payload(swap_quote)
        bridge_tx_payload = self._as_tx_payload(bridge_tx)

        amount_in = self._extract_amount_in(swap_quote)
        if amount_in is None:
            raise ValueError("swap_quote must expose amount_in for analysis")

        swap_calldata = self._as_calldata(swap_tx)
        bridge_calldata = self._as_calldata(bridge_tx_payload)

        placeholder = amount_in if bridge_amount_placeholder is None else bridge_amount_placeholder
        swap_offsets = self._validate_unique_offsets(
            offsets=self._find_u256_offsets(swap_calldata, amount_in),
            role="swap",
            allow_empty=True,
        )
        bridge_offsets = self._validate_unique_offsets(
            offsets=self._find_u256_offsets(bridge_calldata, placeholder),
            role="bridge",
            allow_empty=False,
        )
        extraction = self._resolve_extraction(
            extraction_mode=extraction_mode,
            extraction_offset=extraction_offset,
            extraction_size=extraction_size,
        )

        return AutoPlan(
            swap=PatchPlan(
                calldata=swap_calldata,
                patch_offsets=swap_offsets,
                placeholder_value=amount_in,
            ),
            bridge=PatchPlan(
                calldata=bridge_calldata,
                patch_offsets=bridge_offsets,
                placeholder_value=placeholder,
            ),
            extraction=extraction,
        )


def _maybe_chain_id(value: Any) -> int | None:
    chain_id = getattr(value, "chain_id", None)
    if chain_id is None:
        return None
    try:
        return int(chain_id)
    except Exception:
        return None


def _validate_chain_token_consistency(*, token_out: Any, dst_chain: Any, dst_token: Any) -> None:
    token_out_chain = _maybe_chain_id(token_out)
    dst_token_chain = _maybe_chain_id(dst_token)
    dst_chain_id = int(dst_chain)

    if dst_token_chain is not None and dst_token_chain != dst_chain_id:
        raise ValueError(f"dst_token.chain_id ({dst_token_chain}) does not match dst_chain ({dst_chain_id})")

    if token_out_chain is not None and token_out_chain == dst_chain_id:
        raise ValueError(
            f"token_out.chain_id ({token_out_chain}) should differ from dst_chain ({dst_chain_id}) for bridge flow"
        )


DEFAULT_BRIDGE_AMOUNT_DECODERS: dict[bytes, BridgeAmountDecoder] = {}


def _decode_bridge_amounts(
    calldata: bytes,
    *,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None,
) -> tuple[int, int] | None:
    decoders = DEFAULT_BRIDGE_AMOUNT_DECODERS if bridge_amount_decoders is None else bridge_amount_decoders
    decoder = decoders.get(Analyzer._selector(calldata))
    if decoder is None:
        return None
    return decoder(calldata)


def _validate_bridge_amount_sanity(
    *,
    bridge_calldata: bytes,
    max_slippage_bps: int | None,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None = None,
) -> None:
    amounts = _decode_bridge_amounts(
        bridge_calldata,
        bridge_amount_decoders=bridge_amount_decoders,
    )
    if amounts is None:
        return

    amount_ld, min_amount_ld = amounts
    if min_amount_ld > amount_ld:
        raise ValueError(f"bridge amount sanity failed: minAmountLD ({min_amount_ld}) exceeds amountLD ({amount_ld})")

    if max_slippage_bps is None:
        return
    if max_slippage_bps < 0 or max_slippage_bps > 10_000:
        raise ValueError("max_slippage_bps must be in [0, 10000]")
    if amount_ld == 0:
        return

    slippage_bps = ((amount_ld - min_amount_ld) * 10_000) // amount_ld
    if slippage_bps > max_slippage_bps:
        raise ValueError(
            f"bridge slippage sanity failed: {slippage_bps} bps exceeds max_slippage_bps ({max_slippage_bps})"
        )


def apply_plan(
    builder: DeFiBuilder,
    *,
    amm: Any,
    token_out: Any,
    bridge: Any,
    dst_chain: Any,
    dst_token: Any,
    plan: AutoPlan,
    swap_value: int = 0,
    bridge_value: int = 0,
    prefer_slice_for_last32: bool = False,
    enforce_chain_token_consistency: bool = True,
    enforce_bridge_amount_sanity: bool = True,
    max_slippage_bps: int | None = None,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None = None,
) -> DeFiBuilder:
    """Apply an analyzed plan into an existing DeFiBuilder flow."""
    if enforce_chain_token_consistency:
        _validate_chain_token_consistency(token_out=token_out, dst_chain=dst_chain, dst_token=dst_token)

    if enforce_bridge_amount_sanity:
        _validate_bridge_amount_sanity(
            bridge_calldata=plan.bridge.calldata,
            max_slippage_bps=max_slippage_bps,
            bridge_amount_decoders=bridge_amount_decoders,
        )

    builder = builder.swap(
        amm,
        token_out,
        plan.swap.calldata,
        patch_offsets=plan.swap.patch_offsets or None,
        amount_placeholder=plan.swap.placeholder_value,
        value=swap_value,
    )

    if plan.extraction.mode == ExtractionMode.RET_LAST32:
        builder = builder.amount_from_prev_call_slice() if prefer_slice_for_last32 else builder.amount_from_prev_call()
    elif plan.extraction.mode == ExtractionMode.RET_U256:
        builder = builder.amount_from_prev_call(offset=plan.extraction.offset)
    elif plan.extraction.mode == ExtractionMode.RET_SLICE:
        builder = builder.amount_from_prev_call_slice(offset=plan.extraction.offset, size=plan.extraction.size)
    else:
        raise ValueError(f"unsupported extraction mode: {plan.extraction.mode}")

    return builder.bridge(
        bridge,
        dst_chain,
        dst_token,
        patch_offsets=plan.bridge.patch_offsets,
        amount_placeholder=plan.bridge.placeholder_value,
        value=bridge_value,
        calldata_template=plan.bridge.calldata,
    )


async def execute_flow_from_quotes_async(
    builder: DeFiBuilder,
    *,
    swap_quote: Any,
    bridge_tx: Any,
    amm: Any,
    token_out: Any,
    bridge: Any,
    dst_chain: Any,
    dst_token: Any,
    extraction_mode: ExtractionMode = ExtractionMode.RET_LAST32,
    extraction_offset: int | None = None,
    extraction_size: int | None = None,
    bridge_amount_placeholder: int | None = None,
    swap_value: int = 0,
    bridge_value: int = 0,
    prefer_slice_for_last32: bool = False,
    enforce_chain_token_consistency: bool = True,
    enforce_bridge_amount_sanity: bool = True,
    max_slippage_bps: int | None = None,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None = None,
    **_ignored: Any,
) -> bytes:
    """Compose and execute a generic two-step swap/bridge flow."""
    analyzer = Analyzer()
    plan = analyzer.analyze_from_quotes(
        swap_quote=swap_quote,
        bridge_tx=bridge_tx,
        extraction_mode=extraction_mode,
        extraction_offset=extraction_offset,
        extraction_size=extraction_size,
        bridge_amount_placeholder=bridge_amount_placeholder,
    )

    applied = apply_plan(
        builder,
        amm=amm,
        token_out=token_out,
        bridge=bridge,
        dst_chain=dst_chain,
        dst_token=dst_token,
        plan=plan,
        swap_value=swap_value,
        bridge_value=bridge_value,
        prefer_slice_for_last32=prefer_slice_for_last32,
        enforce_chain_token_consistency=enforce_chain_token_consistency,
        enforce_bridge_amount_sanity=enforce_bridge_amount_sanity,
        max_slippage_bps=max_slippage_bps,
        bridge_amount_decoders=bridge_amount_decoders,
    )
    return await applied.execute_async()
