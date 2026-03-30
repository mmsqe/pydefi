from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, cast

from .blueprint import Blueprint
from .vm_command import VMCommand, VMState


class DeFiBuilder:
    """Blueprint-style eDSL builder for VM command assembly."""

    _AMOUNT_REG = 1
    _SWAP_CALLDATA_REG = 2
    _BRIDGE_CALLDATA_REG = 3
    _RET_U256_WORD0_SELECTORS = {
        # Uniswap V3 QuoterV1
        bytes.fromhex("f7729d43"),  # quoteExactInputSingle(address,address,uint24,uint256,uint160)
        bytes.fromhex("cdca1753"),  # quoteExactInput(bytes,uint256)
        bytes.fromhex("30d07f21"),  # quoteExactOutputSingle(address,address,uint24,uint256,uint160)
        bytes.fromhex("2f80bb1d"),  # quoteExactOutput(bytes,uint256)
        # Uniswap V3 QuoterV2
        bytes.fromhex("c6a5026a"),  # quoteExactInputSingle((address,address,uint256,uint24,uint160))
        bytes.fromhex("bd21704a"),  # quoteExactOutputSingle((address,address,uint256,uint24,uint160))
    }

    def __init__(self, vm: Any):
        self.vm = vm
        self.blueprint: Optional[Blueprint] = None

    @staticmethod
    def _to_uint256_bytes(value: int) -> bytes:
        if value < 0:
            raise ValueError("uint256 value must be non-negative")
        return value.to_bytes(32, "big")

    @staticmethod
    def _resolve_target(source: Any, *, candidates: list[str]) -> str:
        if isinstance(source, str):
            return source

        for name in candidates:
            if hasattr(source, name):
                value = getattr(source, name)
                if isinstance(value, str) and value:
                    return value

        tx_data = getattr(source, "tx_data", None)
        if isinstance(tx_data, dict):
            target = tx_data.get("to")
            if isinstance(target, str) and target:
                return target

        raise ValueError(f"could not resolve target address from source using attrs {candidates!r}")

    @staticmethod
    def _resolve_calldata(source: Any) -> bytes:
        if isinstance(source, bytes):
            return source

        for name in ["calldata", "call_data", "data"]:
            if hasattr(source, name):
                value = getattr(source, name)
                if isinstance(value, bytes):
                    return value
                if isinstance(value, str):
                    return bytes.fromhex(value.removeprefix("0x"))

        tx_data = getattr(source, "tx_data", None)
        if isinstance(tx_data, dict):
            data = tx_data.get("data")
            if isinstance(data, bytes):
                return data
            if isinstance(data, str):
                return bytes.fromhex(data.removeprefix("0x"))

        raise ValueError("could not resolve calldata template from source")

    @staticmethod
    def _encode_call_meta(target: str, value: int) -> bytes:
        target_bytes = bytes.fromhex(target.removeprefix("0x"))
        if len(target_bytes) != 20:
            raise ValueError(f"target must be 20-byte address, got {target}")
        return target_bytes + DeFiBuilder._to_uint256_bytes(value)

    @staticmethod
    def _find_u256_offsets(buf: bytes, value: int) -> list[int]:
        marker = DeFiBuilder._to_uint256_bytes(value)
        offsets: list[int] = []
        start = 0
        while True:
            idx = buf.find(marker, start)
            if idx < 0:
                break
            offsets.append(idx)
            start = idx + 1
        return offsets

    def _require_blueprint(self) -> Blueprint:
        if self.blueprint is None:
            raise ValueError("please call from_token() first")
        return self.blueprint

    def _infer_prev_call_amount_extraction(self) -> tuple[str, bytes]:
        """Infer RET_* opcode to extract amount from previous CALL returndata.
        - Known Uniswap V3/V3 Quoter `quoteExact*` → amount at word 0 → RET_U256(0)
        - V2 `getAmountsOut` / `getAmountsIn` → dynamic array → fallback
        - Default → RET_LAST32
        """
        bp = self._require_blueprint()

        if not bp.commands:
            return "RET_LAST32", b""

        for cmd in reversed(bp.commands):
            if cmd.opcode != "CALL":
                continue

            calldata_reg = cmd.registers[0] if cmd.registers else 0
            for prev in reversed(bp.commands):
                if (
                    prev.opcode == "CALLDATA_BUILD"
                    and prev.registers
                    and prev.registers[0] == calldata_reg
                    and prev.data
                    and len(prev.data) >= 4
                    and prev.data[:4] in self._RET_U256_WORD0_SELECTORS
                ):
                    return "RET_U256", b"\x00\x00"

                if prev.opcode == "CALLDATA_BUILD" and prev.registers and prev.registers[0] == calldata_reg:
                    break

            break

        return "RET_LAST32", b""

    def _append_surgery(self, *, calldata_reg: int, patch_offsets: list[int]) -> None:
        bp = self._require_blueprint()
        if not patch_offsets:
            return
        if len(patch_offsets) > 255:
            raise ValueError("patch_offsets length must fit in uint8")
        for offset in patch_offsets:
            if offset < 0 or offset > 0xFFFF:
                raise ValueError(f"patch offset must fit in uint16, got {offset}")

        payload = bytes([len(patch_offsets)]) + b"".join(offset.to_bytes(2, "big") for offset in patch_offsets)
        bp.add_command(
            VMCommand(
                opcode="CALLDATA_SURGERY",
                data=payload,
                registers=[calldata_reg] + [self._AMOUNT_REG] * len(patch_offsets),
            )
        )

    def _resolve_patch_offsets(
        self,
        *,
        template: bytes,
        patch_offset: Optional[int],
        patch_offsets: Optional[list[int]],
        amount_placeholder: Optional[int],
        role: str,
    ) -> list[int]:
        bp = self._require_blueprint()
        offsets = list(patch_offsets or [])
        if patch_offset is not None:
            offsets.append(patch_offset)
        if not offsets:
            placeholder = bp.amount_in if amount_placeholder is None else amount_placeholder
            inferred = self._find_u256_offsets(template, placeholder)
            if len(inferred) == 1:
                offsets = inferred
            elif len(inferred) > 1:
                raise ValueError(
                    f"multiple amount placeholders found in {role} calldata ({inferred}); pass patch_offset(s) explicitly"
                )
        return offsets

    def _maybe_auto_load_prev_output(self, *, offsets: list[int], auto_amount_from_prev_call: bool) -> None:
        bp = self._require_blueprint()
        if not (auto_amount_from_prev_call and offsets and bp.commands and bp.commands[-1].opcode == "CALL"):
            return

        ret_opcode, ret_data = self._infer_prev_call_amount_extraction()
        bp.add_command(
            VMCommand(
                opcode=ret_opcode,
                data=ret_data,
                registers=[self._AMOUNT_REG],
            )
        )

    def _append_call_action(
        self,
        *,
        target: Any,
        target_candidates: list[str],
        calldata_template: bytes,
        calldata_reg: int,
        patch_offset: Optional[int],
        patch_offsets: Optional[list[int]],
        amount_placeholder: Optional[int],
        value: int,
        auto_amount_from_prev_call: bool,
        role: str,
    ) -> "DeFiBuilder":
        bp = self._require_blueprint()
        offsets = self._resolve_patch_offsets(
            template=calldata_template,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            role=role,
        )

        self._maybe_auto_load_prev_output(offsets=offsets, auto_amount_from_prev_call=auto_amount_from_prev_call)

        bp.add_command(
            VMCommand(
                opcode="CALLDATA_BUILD",
                data=calldata_template,
                registers=[calldata_reg],
            )
        )
        self._append_surgery(calldata_reg=calldata_reg, patch_offsets=offsets)

        resolved_target = self._resolve_target(target, candidates=target_candidates)
        bp.add_command(
            VMCommand(
                opcode="CALL",
                data=self._encode_call_meta(resolved_target, value),
                registers=[calldata_reg],
            )
        )
        return self

    def from_token(self, token: Any, *, amount_in: int) -> "DeFiBuilder":
        self.blueprint = Blueprint(from_token=token, amount_in=amount_in, receiver="")
        registers = [b""] * 16
        registers[self._AMOUNT_REG] = self._to_uint256_bytes(amount_in)
        self.blueprint.initial_state = VMState(registers=registers)
        return self

    def to(self, receiver: str) -> "DeFiBuilder":
        bp = self._require_blueprint()
        bp.receiver = receiver
        return self

    def amount_from_prev_call(self, *, offset: Optional[int] = None, register: Optional[int] = None) -> "DeFiBuilder":
        """Load uint256 from previous CALL returndata into a register.

        This is typically used before a subsequent ``CALLDATA_SURGERY`` so the
        patched value can come from a prior quote response rather than the
        initial ``amount_in``.
        """
        bp = self._require_blueprint()

        target_reg = self._AMOUNT_REG if register is None else register
        if target_reg < 0:
            raise ValueError("register must be non-negative")

        if offset is not None and (offset < 0 or offset > 0xFFFF):
            raise ValueError(f"offset must fit in uint16, got {offset}")

        if offset is None:
            bp.add_command(
                VMCommand(
                    opcode="RET_LAST32",
                    data=b"",
                    registers=[target_reg],
                )
            )
            return self

        bp.add_command(
            VMCommand(
                opcode="RET_U256",
                data=offset.to_bytes(2, "big"),
                registers=[target_reg],
            )
        )
        return self

    def amount_from_prev_call_slice(
        self,
        *,
        offset: Optional[int] = None,
        size: Optional[int] = None,
        register: Optional[int] = None,
    ) -> "DeFiBuilder":
        """Load a raw bytes slice from previous CALL returndata into a register.

        This helper keeps RET_SLICE-style semantics while still allowing a
        subsequent ``CALLDATA_SURGERY`` patch to consume the register value.
        """
        bp = self._require_blueprint()

        target_reg = self._AMOUNT_REG if register is None else register
        if target_reg < 0:
            raise ValueError("register must be non-negative")

        if offset is None and size is None:
            bp.add_command(
                VMCommand(
                    opcode="RET_LAST32",
                    data=b"",
                    registers=[target_reg],
                )
            )
            return self

        if offset is None or size is None:
            raise ValueError("offset and size must both be provided when not using auto mode")
        if offset < 0 or offset > 0xFFFF:
            raise ValueError(f"offset must fit in uint16, got {offset}")
        # RET_SLICE output is later interpreted as a uint256 by CALLDATA_SURGERY,
        # so the slice size must not exceed 32 bytes.
        if size < 0 or size > 32:
            raise ValueError(f"size must be between 0 and 32 bytes for uint256-compatible slice, got {size}")

        bp.add_command(
            VMCommand(
                opcode="RET_SLICE",
                data=offset.to_bytes(2, "big") + size.to_bytes(2, "big"),
                registers=[target_reg],
            )
        )
        return self

    def swap(
        self,
        amm: Any,
        token_out: Any,
        dex_calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        _ = token_out
        return self._append_call_action(
            target=amm,
            target_candidates=["router_address", "address", "to"],
            calldata_template=dex_calldata_template,
            calldata_reg=self._SWAP_CALLDATA_REG,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            role="swap",
        )

    def bridge(
        self,
        bridge: Any,
        dst_chain: Any,
        dst_token: Any,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        calldata_template: Optional[bytes] = None,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        _ = dst_token
        bp = self._require_blueprint()
        bp.chain_id = int(dst_chain)
        template = calldata_template if calldata_template is not None else self._resolve_calldata(bridge)
        return self._append_call_action(
            target=bridge,
            target_candidates=["oft_address", "bridge_address", "router_address", "address", "to"],
            calldata_template=template,
            calldata_reg=self._BRIDGE_CALLDATA_REG,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            role="bridge",
        )

    def call(
        self,
        target: Any,
        calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
        target_candidates: Optional[list[str]] = None,
    ) -> "DeFiBuilder":
        candidates = target_candidates or [
            "address",
            "to",
            "router_address",
            "bridge_address",
            "vault_address",
            "staking_address",
            "minter_address",
            "wrapper_address",
            "oft_address",
        ]
        return self._append_call_action(
            target=target,
            target_candidates=candidates,
            calldata_template=calldata_template,
            calldata_reg=self._BRIDGE_CALLDATA_REG,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            role="call",
        )

    def deposit(
        self,
        protocol: Any,
        calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        return self.call(
            protocol,
            calldata_template,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            target_candidates=["vault_address", "pool_address", "address", "to"],
        )

    def stake(
        self,
        protocol: Any,
        calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        return self.call(
            protocol,
            calldata_template,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            target_candidates=["staking_address", "gauge_address", "address", "to"],
        )

    def mint(
        self,
        protocol: Any,
        calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        return self.call(
            protocol,
            calldata_template,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            target_candidates=["minter_address", "address", "to"],
        )

    def wrap(
        self,
        wrapper: Any,
        calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        return self.call(
            wrapper,
            calldata_template,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            target_candidates=["wrapper_address", "address", "to"],
        )

    def unwrap(
        self,
        wrapper: Any,
        calldata_template: bytes,
        *,
        patch_offset: Optional[int] = None,
        patch_offsets: Optional[list[int]] = None,
        amount_placeholder: Optional[int] = None,
        value: int = 0,
        auto_amount_from_prev_call: bool = True,
    ) -> "DeFiBuilder":
        return self.call(
            wrapper,
            calldata_template,
            patch_offset=patch_offset,
            patch_offsets=patch_offsets,
            amount_placeholder=amount_placeholder,
            value=value,
            auto_amount_from_prev_call=auto_amount_from_prev_call,
            target_candidates=["wrapper_address", "address", "to"],
        )

    def build(self) -> Blueprint:
        bp = self._require_blueprint()
        if not bp.receiver:
            raise ValueError("please call to() before build()")
        return bp

    def execute(self) -> bytes:
        bp = self.build()
        if bp.initial_state is None:
            raise ValueError("blueprint.initial_state is missing")
        result, success = self.vm.run(bp.commands, bp.initial_state)
        if not success:
            raise ValueError("blueprint execution failed")
        return result

    async def execute_async(self) -> bytes:
        """Execute blueprint using VM async runtime (e.g. AsyncWeb3-backed CALLs)."""
        bp = self.build()
        if bp.initial_state is None:
            raise ValueError("blueprint.initial_state is missing")
        run_async = getattr(self.vm, "run_async", None)
        if not callable(run_async):
            raise ValueError("attached VM does not support async execution")

        run_async_typed = cast(
            Callable[[list[VMCommand], VMState], Awaitable[tuple[bytes, bool]]],
            run_async,
        )
        result, success = await run_async_typed(bp.commands, bp.initial_state)
        if not success:
            raise ValueError("blueprint execution failed")
        return result

    def execute_transact(self) -> bytes:
        """Execute blueprint through a transact-backed VM runtime."""
        return self.execute()

    async def execute_transact_async(self) -> bytes:
        """Execute blueprint through an async transact-backed VM runtime."""
        return await self.execute_async()
