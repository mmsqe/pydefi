from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_graph import (
    ActionPlan,
    BridgeAmountDecoder,
    ExtractionMode,
    apply_action_graph,
    coerce_extraction_mode,
    execute_action_graph_async,
)


@dataclass(frozen=True)
class StepPlan:
    """Declarative intent plan represented as a generic step list.

    This model decouples intent description from execution details. Steps are
    plain dict entries (for example ``{"action": "swap", ...}``) and may
    include a ``split`` node for proportion-based branching.
    """

    source_token: Any
    source_amount: int
    receiver: str
    steps: list[Any]
    dst_chain: Any = 1
    token_out: Any | None = None
    dst_token: Any | None = None


class FlowRunner:
    """Execute generic step plans and action graphs on top of DeFiBuilder."""

    def __init__(self, vm: Any):
        self.vm = vm

    def create(
        self,
        *,
        source_token: Any,
        source_amount: int,
        receiver: str,
        steps: list[Any],
        dst_chain: Any = 1,
        token_out: Any | None = None,
        dst_token: Any | None = None,
    ) -> StepPlan:
        """Create a declarative step plan without executing it."""

        return StepPlan(
            source_token=source_token,
            source_amount=int(source_amount),
            receiver=receiver,
            steps=steps,
            dst_chain=dst_chain,
            token_out=token_out,
            dst_token=dst_token,
        )

    async def _execute_actions(
        self,
        *,
        from_token: Any,
        to_token: Any,
        amount_in: int,
        dst_chain: Any,
        receiver: str,
        actions: list[Any],
        token_out: Any | None = None,
        dst_token: Any | None = None,
        enforce_chain_token_consistency: bool = True,
        enforce_bridge_amount_sanity: bool = True,
        max_slippage_bps: int | None = None,
        bridge_amount_decoders: dict[bytes, "BridgeAmountDecoder"] | None = None,
    ) -> bytes:
        resolved_token_out = to_token if token_out is None else token_out
        resolved_dst_token = resolved_token_out if dst_token is None else dst_token

        builder = self.vm.builder().from_token(from_token, amount_in=int(amount_in)).to(receiver)
        result = await execute_action_graph_async(
            builder,
            actions=actions,
            token_out=resolved_token_out,
            dst_chain=dst_chain,
            dst_token=resolved_dst_token,
            enforce_chain_token_consistency=enforce_chain_token_consistency,
            enforce_bridge_amount_sanity=enforce_bridge_amount_sanity,
            max_slippage_bps=max_slippage_bps,
            bridge_amount_decoders=bridge_amount_decoders,
        )
        return bytes(result)

    async def _execute_linear_steps(
        self,
        *,
        from_token: Any,
        amount_in: int,
        receiver: str,
        dst_chain: Any,
        steps: list[Any],
        token_out: Any | None,
        dst_token: Any | None,
    ) -> bytes:
        """Compile linear steps into actions and execute them."""
        actions = [_step_to_action(step) for step in steps]
        resolved_to_token = _resolve_steps_terminal_token(
            steps, default=token_out if token_out is not None else from_token
        )
        return await self._execute_actions(
            from_token=from_token,
            to_token=resolved_to_token,
            amount_in=int(amount_in),
            dst_chain=dst_chain,
            receiver=receiver,
            actions=actions,
            token_out=resolved_to_token,
            dst_token=dst_token if dst_token is not None else resolved_to_token,
        )

    def _decode_steps_output_amount(self, *, steps: list[Any], ret: bytes) -> int:
        """Decode amount-like output from step execution according to terminal extraction mode."""
        extraction_mode, extraction_offset, extraction_size = _resolve_terminal_extraction(steps)
        return self._decode_from_returndata(
            extraction_mode=extraction_mode,
            extraction_offset=extraction_offset,
            extraction_size=extraction_size,
            ret=ret,
        )

    @staticmethod
    def _decode_from_returndata(
        *,
        extraction_mode: ExtractionMode,
        extraction_offset: int | None,
        extraction_size: int | None,
        ret: bytes,
    ) -> int:
        if extraction_mode == ExtractionMode.RET_LAST32:
            if len(ret) < 32:
                raise ValueError("returndata too short for RET_LAST32")
            return int.from_bytes(ret[-32:], "big")

        if extraction_mode == ExtractionMode.RET_U256:
            offset = int(extraction_offset or 0)
            if offset < 0:
                raise ValueError("RET_U256 offset must be non-negative")
            if offset + 32 > len(ret):
                raise ValueError("returndata too short for RET_U256 offset")
            return int.from_bytes(ret[offset : offset + 32], "big")

        if extraction_mode == ExtractionMode.RET_SLICE:
            if extraction_offset is None or extraction_size is None:
                raise ValueError("RET_SLICE requires extraction_offset and extraction_size")
            start = int(extraction_offset)
            size = int(extraction_size)
            if start < 0 or size < 0:
                raise ValueError("RET_SLICE offset/size must be non-negative")
            end = start + size
            if end > len(ret):
                raise ValueError("returndata too short for RET_SLICE range")
            chunk = ret[start:end]
            if len(chunk) > 32:
                raise ValueError("RET_SLICE chunk cannot exceed 32 bytes for uint256 decode")
            return int.from_bytes(chunk.rjust(32, b"\x00"), "big")

        raise ValueError(f"unsupported extraction mode: {extraction_mode}")

    async def _execute_step_sequence(
        self,
        *,
        from_token: Any,
        amount_in: int,
        receiver: str,
        dst_chain: Any,
        steps: list[Any],
        token_out: Any | None,
        dst_token: Any | None,
    ) -> dict[str, Any] | bytes:
        if not steps:
            raise ValueError("steps must be a non-empty list")

        split_idx = _find_split_index(steps)
        if split_idx is None:
            return await self._execute_linear_steps(
                from_token=from_token,
                amount_in=int(amount_in),
                receiver=receiver,
                dst_chain=dst_chain,
                steps=steps,
                token_out=token_out,
                dst_token=dst_token,
            )

        prefix_steps = steps[:split_idx]
        split_step = steps[split_idx]
        suffix_steps = steps[split_idx + 1 :]

        if suffix_steps:
            raise ValueError(
                "steps after split are not supported yet; place downstream actions inside each split branch"
            )
        if not prefix_steps:
            raise ValueError("split requires at least one step before it")

        prefix_to_token = _resolve_steps_terminal_token(prefix_steps, default=from_token)
        prefix_result = await self._execute_linear_steps(
            from_token=from_token,
            amount_in=int(amount_in),
            receiver=receiver,
            dst_chain=dst_chain,
            steps=prefix_steps,
            token_out=prefix_to_token,
            dst_token=dst_token if dst_token is not None else prefix_to_token,
        )
        amount_mid = self._decode_steps_output_amount(steps=prefix_steps, ret=prefix_result)

        split_token = _resolve_split_token(split_step, default=prefix_to_token)
        portions = _normalize_split_portions(split_step)
        split_amounts = [(amount_mid * bps) // 10_000 for bps, _ in portions]

        branch_results: list[dict[str, Any]] = []
        for idx, (bps, branch_steps) in enumerate(portions):
            branch_amount = split_amounts[idx]
            branch_result = await self._execute_step_sequence(
                from_token=split_token,
                amount_in=branch_amount,
                receiver=receiver,
                dst_chain=dst_chain,
                steps=branch_steps,
                token_out=token_out,
                dst_token=dst_token,
            )

            branch_amount_out: int | None = None
            if isinstance(branch_result, (bytes, bytearray)):
                branch_amount_out = self._decode_steps_output_amount(steps=branch_steps, ret=bytes(branch_result))

            branch_results.append(
                {
                    "branch_index": idx,
                    "ratio_bps": bps,
                    "amount_in": branch_amount,
                    "amount_out": branch_amount_out,
                    "result": branch_result,
                }
            )

        return {
            "amount_mid_total": amount_mid,
            "split_amounts": split_amounts,
            "branches": branch_results,
            "split_token": split_token,
        }

    async def execute(
        self,
        *,
        plan: StepPlan,
    ) -> dict[str, Any] | bytes:
        """Execute a declarative step plan."""

        return await self._execute_step_sequence(
            from_token=plan.source_token,
            amount_in=int(plan.source_amount),
            receiver=plan.receiver,
            dst_chain=plan.dst_chain,
            steps=plan.steps,
            token_out=plan.token_out,
            dst_token=plan.dst_token,
        )


def _step_action_name(raw_step: Any) -> str | None:
    if not isinstance(raw_step, dict):
        return None
    value = raw_step.get("action")
    if value is None:
        value = raw_step.get("kind")
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def _find_split_index(steps: list[Any]) -> int | None:
    for idx, step in enumerate(steps):
        if _step_action_name(step) == "split":
            return idx
    return None


def _step_to_action(raw_step: Any) -> Any:
    if isinstance(raw_step, ActionPlan):
        return raw_step
    if not isinstance(raw_step, dict):
        return raw_step

    action_name = _step_action_name(raw_step)
    if action_name == "split":
        raise ValueError("split step cannot be converted into a linear action")

    out = dict(raw_step)
    if "kind" not in out and "action" in out:
        out["kind"] = out["action"]
    out.pop("action", None)

    if "token_out" not in out and "to" in out:
        out["token_out"] = out["to"]

    return out


def _resolve_steps_terminal_token(steps: list[Any], *, default: Any) -> Any:
    for raw_step in reversed(steps):
        if not isinstance(raw_step, dict):
            continue
        action_name = _step_action_name(raw_step)
        if action_name == "split":
            continue
        if "token_out" in raw_step and raw_step["token_out"] is not None:
            return raw_step["token_out"]
        if "to" in raw_step and raw_step["to"] is not None:
            return raw_step["to"]
    return default


def _resolve_terminal_extraction(steps: list[Any]) -> tuple[ExtractionMode, int | None, int | None]:
    for raw_step in reversed(steps):
        if not isinstance(raw_step, dict):
            continue

        action_name = _step_action_name(raw_step)
        if action_name == "split":
            continue

        if action_name == "extract":
            mode = coerce_extraction_mode(raw_step.get("extraction_mode", ExtractionMode.RET_LAST32))
            return mode, raw_step.get("extraction_offset"), raw_step.get("extraction_size")

        mode = coerce_extraction_mode(raw_step.get("extraction_mode", ExtractionMode.RET_LAST32))
        return mode, raw_step.get("extraction_offset"), raw_step.get("extraction_size")

    raise ValueError("cannot resolve extraction mode from empty step sequence")


def _normalize_bps(value: Any) -> int:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            return _normalize_bps(stripped[:-1])
        value = float(stripped)

    if isinstance(value, float):
        if 0 <= value <= 1:
            return int(round(value * 10_000))
        return int(round(value * 100))

    value_int = int(value)
    if 0 <= value_int <= 100:
        return value_int * 100
    return value_int


def _portion_to_steps(raw_portion: dict[str, Any]) -> list[Any]:
    if "steps" in raw_portion:
        raw_steps = raw_portion["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("split portion steps must be a non-empty list")
        return raw_steps

    single_step = {k: v for k, v in raw_portion.items() if k not in {"percent", "bps", "ratio", "steps"}}
    if not single_step:
        raise ValueError("split portion must define either steps or an inline action")
    return [single_step]


def _normalize_split_portions(raw_split_step: Any) -> list[tuple[int, list[Any]]]:
    if not isinstance(raw_split_step, dict):
        raise ValueError("split step must be a dict")
    if _step_action_name(raw_split_step) != "split":
        raise ValueError("expected split step")

    raw_portions = raw_split_step.get("portions")
    if not isinstance(raw_portions, list) or not raw_portions:
        raise ValueError("split step requires non-empty portions list")

    portions: list[tuple[int, list[Any]]] = []
    for raw_portion in raw_portions:
        if not isinstance(raw_portion, dict):
            raise ValueError("each split portion must be a dict")
        bps_raw = raw_portion.get("bps")
        if bps_raw is None:
            bps_raw = raw_portion.get("percent")
        if bps_raw is None:
            bps_raw = raw_portion.get("ratio")
        if bps_raw is None:
            raise ValueError("split portion requires bps/percent/ratio")

        bps = _normalize_bps(bps_raw)
        if bps < 0 or bps > 10_000:
            raise ValueError("split portion ratio must be in [0, 10000] bps")

        steps = _portion_to_steps(raw_portion)
        portions.append((bps, steps))

    if sum(bps for bps, _ in portions) != 10_000:
        raise ValueError("split portions must sum to 10000 bps")

    return portions


def _resolve_split_token(raw_split_step: Any, *, default: Any) -> Any:
    if isinstance(raw_split_step, dict):
        if raw_split_step.get("token") is not None:
            return raw_split_step["token"]
        if raw_split_step.get("from") is not None:
            return raw_split_step["from"]
    return default
