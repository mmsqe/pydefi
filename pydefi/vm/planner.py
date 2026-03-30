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
class ActionPlan:
    kind: str
    target: Any | None = None
    calldata: bytes | None = None
    patch_offsets: list[int] | None = None
    patch_offset: int | None = None
    amount_placeholder: int | None = None
    value: int = 0
    auto_amount_from_prev_call: bool = True
    target_candidates: list[str] | None = None
    token_out: Any | None = None
    dst_chain: Any | None = None
    dst_token: Any | None = None
    extraction_mode: ExtractionMode = ExtractionMode.RET_LAST32
    extraction_offset: int | None = None
    extraction_size: int | None = None
    prefer_slice_for_last32: bool = False


BridgeAmountDecoder = Callable[[bytes], tuple[int, int] | None]


DEFAULT_BRIDGE_AMOUNT_DECODERS: dict[bytes, BridgeAmountDecoder] = {}


def _selector(buf: bytes) -> bytes:
    if len(buf) < 4:
        return b""
    return buf[:4]


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


def _as_tx_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        if isinstance(raw.get("tx_data"), dict):
            return raw["tx_data"]
        return raw
    tx_data = getattr(raw, "tx_data", None)
    if isinstance(tx_data, dict):
        return tx_data
    raise ValueError("expected tx payload dict or object with tx_data dict")


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


def _decode_bridge_amounts(
    calldata: bytes,
    *,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None,
) -> tuple[int, int] | None:
    decoders = DEFAULT_BRIDGE_AMOUNT_DECODERS if bridge_amount_decoders is None else bridge_amount_decoders
    decoder = decoders.get(_selector(calldata))
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


def _coerce_extraction_mode(value: Any) -> ExtractionMode:
    if isinstance(value, ExtractionMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for mode in ExtractionMode:
            if normalized == mode.value or normalized == mode.name.lower():
                return mode
    raise ValueError(f"unsupported extraction mode: {value!r}")


def _normalize_action_kind(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("action.kind must be a non-empty string")
    return value.strip().lower()


def _normalize_action(raw: Any) -> ActionPlan:
    if isinstance(raw, ActionPlan):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("each action must be a dict or ActionPlan")

    kind = _normalize_action_kind(raw.get("kind") or raw.get("type"))

    target = raw.get("target")
    if target is None:
        if kind == "swap":
            target = raw.get("amm")
        elif kind == "bridge":
            target = raw.get("bridge")
        elif kind in {"deposit", "stake", "mint", "wrap", "unwrap"}:
            target = raw.get("protocol") or raw.get("wrapper")

    calldata: bytes | None = None
    if "calldata" in raw:
        calldata = _as_calldata(raw["calldata"])
    elif "calldata_template" in raw:
        calldata = _as_calldata(raw["calldata_template"])
    elif "tx" in raw:
        calldata = _as_calldata(_as_tx_payload(raw["tx"]))
    elif "quote" in raw:
        calldata = _as_calldata(_as_tx_payload(raw["quote"]))

    amount_placeholder = raw.get("amount_placeholder")
    if amount_placeholder is None and "quote" in raw:
        inferred_placeholder = _extract_amount_in(raw["quote"])
        if inferred_placeholder is not None:
            amount_placeholder = inferred_placeholder

    patch_offsets_raw = raw.get("patch_offsets")
    patch_offsets: list[int] | None = None
    if patch_offsets_raw is not None:
        if not isinstance(patch_offsets_raw, list):
            raise ValueError("action.patch_offsets must be a list[int]")
        patch_offsets = [int(v) for v in patch_offsets_raw]

    patch_offset_raw = raw.get("patch_offset")
    patch_offset: int | None = None
    if patch_offset_raw is not None:
        patch_offset = int(patch_offset_raw)

    extraction_mode_raw = raw.get("extraction_mode", ExtractionMode.RET_LAST32)
    extraction_mode = _coerce_extraction_mode(extraction_mode_raw)

    target_candidates_raw = raw.get("target_candidates")
    target_candidates: list[str] | None = None
    if target_candidates_raw is not None:
        if not isinstance(target_candidates_raw, list):
            raise ValueError("action.target_candidates must be a list[str]")
        target_candidates = [str(v) for v in target_candidates_raw]

    return ActionPlan(
        kind=kind,
        target=target,
        calldata=calldata,
        patch_offsets=patch_offsets,
        patch_offset=patch_offset,
        amount_placeholder=amount_placeholder,
        value=int(raw.get("value", 0)),
        auto_amount_from_prev_call=bool(raw.get("auto_amount_from_prev_call", True)),
        target_candidates=target_candidates,
        token_out=raw.get("token_out"),
        dst_chain=raw.get("dst_chain"),
        dst_token=raw.get("dst_token"),
        extraction_mode=extraction_mode,
        extraction_offset=raw.get("extraction_offset"),
        extraction_size=raw.get("extraction_size"),
        prefer_slice_for_last32=bool(raw.get("prefer_slice_for_last32", False)),
    )


def _append_explicit_extraction(builder: DeFiBuilder, action: ActionPlan) -> DeFiBuilder:
    if action.extraction_mode == ExtractionMode.RET_LAST32:
        if action.extraction_offset is not None or action.extraction_size is not None:
            raise ValueError("RET_LAST32 extraction does not accept extraction_offset/extraction_size")
        return (
            builder.amount_from_prev_call_slice() if action.prefer_slice_for_last32 else builder.amount_from_prev_call()
        )

    if action.extraction_mode == ExtractionMode.RET_U256:
        if action.extraction_offset is None:
            raise ValueError("extraction_offset is required for RET_U256")
        return builder.amount_from_prev_call(offset=int(action.extraction_offset))

    if action.extraction_mode == ExtractionMode.RET_SLICE:
        if action.extraction_offset is None or action.extraction_size is None:
            raise ValueError("extraction_offset and extraction_size are required for RET_SLICE")
        return builder.amount_from_prev_call_slice(
            offset=int(action.extraction_offset),
            size=int(action.extraction_size),
        )

    raise ValueError(f"unsupported extraction mode: {action.extraction_mode}")


def apply_action_graph(
    builder: DeFiBuilder,
    *,
    actions: list[Any],
    token_out: Any | None = None,
    dst_chain: Any | None = None,
    dst_token: Any | None = None,
    enforce_chain_token_consistency: bool = True,
    enforce_bridge_amount_sanity: bool = True,
    max_slippage_bps: int | None = None,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None = None,
) -> DeFiBuilder:
    """Apply a multi-step action graph into an existing DeFiBuilder flow."""
    if not actions:
        raise ValueError("actions must be a non-empty list")

    for raw_action in actions:
        action = _normalize_action(raw_action)

        if action.kind == "extract":
            builder = _append_explicit_extraction(builder, action)
            continue

        if action.calldata is None:
            raise ValueError(f"action {action.kind!r} requires calldata/calldata_template/tx/quote")
        if action.target is None:
            raise ValueError(f"action {action.kind!r} requires target")

        if action.kind == "swap":
            builder = builder.swap(
                action.target,
                action.token_out if action.token_out is not None else token_out,
                action.calldata,
                patch_offset=action.patch_offset,
                patch_offsets=action.patch_offsets,
                amount_placeholder=action.amount_placeholder,
                value=action.value,
                auto_amount_from_prev_call=action.auto_amount_from_prev_call,
            )
            continue

        if action.kind == "bridge":
            resolved_dst_chain = action.dst_chain if action.dst_chain is not None else dst_chain
            resolved_dst_token = action.dst_token if action.dst_token is not None else dst_token
            if resolved_dst_chain is None:
                raise ValueError("bridge action requires dst_chain (either per-action or function default)")

            if enforce_chain_token_consistency:
                _validate_chain_token_consistency(
                    token_out=action.token_out if action.token_out is not None else token_out,
                    dst_chain=resolved_dst_chain,
                    dst_token=resolved_dst_token,
                )

            if enforce_bridge_amount_sanity:
                _validate_bridge_amount_sanity(
                    bridge_calldata=action.calldata,
                    max_slippage_bps=max_slippage_bps,
                    bridge_amount_decoders=bridge_amount_decoders,
                )

            builder = builder.bridge(
                action.target,
                resolved_dst_chain,
                resolved_dst_token,
                patch_offset=action.patch_offset,
                patch_offsets=action.patch_offsets,
                amount_placeholder=action.amount_placeholder,
                value=action.value,
                calldata_template=action.calldata,
                auto_amount_from_prev_call=action.auto_amount_from_prev_call,
            )
            continue

        if action.kind in {"call", "deposit", "stake", "mint", "wrap", "unwrap"}:
            call_fn = getattr(builder, action.kind)
            builder = call_fn(
                action.target,
                action.calldata,
                patch_offset=action.patch_offset,
                patch_offsets=action.patch_offsets,
                amount_placeholder=action.amount_placeholder,
                value=action.value,
                auto_amount_from_prev_call=action.auto_amount_from_prev_call,
                **({"target_candidates": action.target_candidates} if action.kind == "call" else {}),
            )
            continue

        raise ValueError(f"unsupported action kind: {action.kind}")

    return builder


async def execute_action_graph_async(
    builder: DeFiBuilder,
    *,
    actions: list[Any],
    token_out: Any | None = None,
    dst_chain: Any | None = None,
    dst_token: Any | None = None,
    enforce_chain_token_consistency: bool = True,
    enforce_bridge_amount_sanity: bool = True,
    max_slippage_bps: int | None = None,
    bridge_amount_decoders: dict[bytes, BridgeAmountDecoder] | None = None,
    **_ignored: Any,
) -> bytes:
    """Compose and execute a generic multi-step action graph flow."""
    applied = apply_action_graph(
        builder,
        actions=actions,
        token_out=token_out,
        dst_chain=dst_chain,
        dst_token=dst_token,
        enforce_chain_token_consistency=enforce_chain_token_consistency,
        enforce_bridge_amount_sanity=enforce_bridge_amount_sanity,
        max_slippage_bps=max_slippage_bps,
        bridge_amount_decoders=bridge_amount_decoders,
    )
    return await applied.execute_async()
