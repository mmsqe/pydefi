from __future__ import annotations

from collections.abc import Awaitable as AwaitableABC
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Tuple

from .patcher import Patcher
from .vm_command import VMCommand, VMState

if TYPE_CHECKING:
    from .builder import DeFiBuilder


CallExecutor = Callable[[str, bytes, int], Awaitable[Tuple[bool, bytes] | bytes]]
ComposeResolver = Callable[[Any, Any, int, Any], Awaitable[dict[str, Any]] | dict[str, Any]]


class VirtualMachine:
    """Minimal Python virtual machine for Blueprint command execution."""

    _DEFAULT_COMPOSE_RECEIVER = "0x000000000000000000000000000000000000dEaD"
    _COMPOSE_CONTEXT_KEYS = (
        "actions",
        "action_graph",
        "token_out",
        "dst_token",
        "dst_chain",
        "receiver",
        "execution_kwargs",
    )

    def __init__(
        self,
        register_count: int = 16,
        call_executor: CallExecutor | None = None,
        compose_resolver: ComposeResolver | None = None,
    ):
        self._register_count = register_count
        self._call_executor = call_executor
        self._compose_resolver = compose_resolver
        self.registers: List[bytes] = [b""] * self._register_count

    def set_compose_resolver(self, resolver: ComposeResolver | None) -> "VirtualMachine":
        """Set or clear route resolver used by :meth:`compose`."""
        self._compose_resolver = resolver
        return self

    @classmethod
    def from_async_web3(
        cls,
        w3: Any,
        *,
        sender: str | None = None,
        block_identifier: Any = "latest",
        register_count: int = 16,
    ) -> "VirtualMachine":
        """Create a VM wired to an async web3 provider via eth_call."""

        async def _executor(target: str, calldata: bytes, value: int) -> bytes:
            tx = {
                "to": target,
                "data": "0x" + calldata.hex(),
                "value": value,
            }
            if sender is not None:
                tx["from"] = sender

            try:
                result = await w3.eth.call(tx, block_identifier=block_identifier)
            except TypeError:
                result = await w3.eth.call(tx, block_identifier)

            if isinstance(result, str):
                return bytes.fromhex(result.removeprefix("0x"))
            return bytes(result)

        return cls(register_count=register_count, call_executor=_executor)

    @classmethod
    def from_async_web3_transact(
        cls,
        w3: Any,
        *,
        sender: str,
        gas: int | None = None,
        register_count: int = 16,
    ) -> "VirtualMachine":
        """Create a VM wired to an async web3 provider via send_transaction."""

        async def _executor(target: str, calldata: bytes, value: int) -> tuple[bool, bytes]:
            tx: dict[str, Any] = {
                "to": target,
                "data": "0x" + calldata.hex(),
                "value": value,
                "from": sender,
            }
            if gas is not None:
                tx["gas"] = gas

            tx_hash = await w3.eth.send_transaction(tx)
            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
            status = receipt.get("status", 1)
            tx_hash_bytes = bytes(tx_hash)
            return bool(status == 1), tx_hash_bytes

        return cls(register_count=register_count, call_executor=_executor)

    def builder(self) -> "DeFiBuilder":
        from .builder import DeFiBuilder

        return DeFiBuilder(self)

    async def compose(
        self,
        *,
        from_token: Any,
        to_token: Any,
        amount_in: int,
        dst_chain: Any,
        receiver: str | None = None,
        compose_resolver: ComposeResolver | None = None,
        **execution_overrides: Any,
    ) -> bytes:
        """Compose and execute a flow using high-level token inputs.

        Route context can be provided either by a resolver or inline kwargs.
        Action-graph mode required key: ``actions`` (or ``action_graph``).
        Optional keys: ``token_out``, ``dst_token``, ``receiver``,
        ``execution_kwargs``.
        """

        # Strip route-context keys from execution_overrides to avoid duplicate kwargs.
        route_context_overrides: dict[str, Any] = {}
        for key in self._COMPOSE_CONTEXT_KEYS:
            if key in execution_overrides:
                route_context_overrides[key] = execution_overrides.pop(key)

        resolver = compose_resolver or self._compose_resolver
        context: dict[str, Any]
        if resolver is not None:
            resolved = resolver(from_token, to_token, amount_in, dst_chain)
            if isinstance(resolved, AwaitableABC):
                context = await resolved
            else:
                context = resolved

            if route_context_overrides:
                context = {**context, **route_context_overrides}
        else:
            context = route_context_overrides

            if not context:
                raise ValueError(
                    "compose requires either compose_resolver or inline route context keys: actions/action_graph"
                )

        actions_context = context.get("actions")
        if actions_context is None:
            actions_context = context.get("action_graph")

        if actions_context is None:
            raise ValueError("compose route context missing required key: actions (or action_graph)")

        receiver_addr = receiver or context.get("receiver") or self._DEFAULT_COMPOSE_RECEIVER

        compose_builder = self.builder().from_token(from_token, amount_in=amount_in).to(receiver_addr)

        execute_kwargs = dict(context.get("execution_kwargs", {}))
        execute_kwargs.update(execution_overrides)

        planner_dst_chain = context.get("dst_chain", dst_chain)

        from .planner import execute_action_graph_async

        if not isinstance(actions_context, list) or not actions_context:
            raise ValueError("compose action graph mode requires a non-empty list in actions/action_graph")

        result = await execute_action_graph_async(
            compose_builder,
            actions=actions_context,
            token_out=context.get("token_out", to_token),
            dst_chain=planner_dst_chain,
            dst_token=context.get("dst_token", to_token),
            **execute_kwargs,
        )
        return result

    def _require_register(self, reg_idx: int) -> None:
        if reg_idx < 0 or reg_idx >= self._register_count:
            raise ValueError(f"register index out of range: {reg_idx}")

    def _normalize_registers(self, initial_state: VMState) -> list[bytes]:
        """Normalize registers to exactly ``_register_count`` entries."""
        base_registers = list(initial_state.registers)
        if len(base_registers) > self._register_count:
            base_registers = base_registers[: self._register_count]
        elif len(base_registers) < self._register_count:
            base_registers.extend([b""] * (self._register_count - len(base_registers)))
        return base_registers

    def run(self, commands: List[VMCommand], initial_state: VMState) -> Tuple[bytes, bool]:
        """Execute a full command sequence using the provided initial state."""
        self.registers = self._normalize_registers(initial_state)

        for cmd in commands:
            if cmd.opcode == "CALLDATA_BUILD":
                self._calldata_build(cmd)
            elif cmd.opcode == "CALLDATA_SURGERY":
                self._calldata_surgery(cmd)
            elif cmd.opcode == "RET_U256":
                self._ret_u256(cmd)
            elif cmd.opcode == "RET_SLICE":
                self._ret_slice(cmd)
            elif cmd.opcode == "RET_LAST32":
                self._ret_last32(cmd)
            elif cmd.opcode == "CALL":
                self._perform_call(cmd)
            elif cmd.opcode == "RETURN":
                if cmd.registers:
                    reg_idx = cmd.registers[0]
                    self._require_register(reg_idx)
                    return self.registers[reg_idx], True
                return cmd.data, True
            else:
                raise ValueError(f"unknown opcode: {cmd.opcode}")

        # Default output register.
        return self.registers[0], True

    async def run_async(self, commands: List[VMCommand], initial_state: VMState) -> Tuple[bytes, bool]:
        """Async variant of run that supports async call executors."""
        self.registers = self._normalize_registers(initial_state)

        for cmd in commands:
            if cmd.opcode == "CALLDATA_BUILD":
                self._calldata_build(cmd)
            elif cmd.opcode == "CALLDATA_SURGERY":
                self._calldata_surgery(cmd)
            elif cmd.opcode == "RET_U256":
                self._ret_u256(cmd)
            elif cmd.opcode == "RET_SLICE":
                self._ret_slice(cmd)
            elif cmd.opcode == "RET_LAST32":
                self._ret_last32(cmd)
            elif cmd.opcode == "CALL":
                await self._perform_call_async(cmd)
            elif cmd.opcode == "RETURN":
                if cmd.registers:
                    reg_idx = cmd.registers[0]
                    self._require_register(reg_idx)
                    return self.registers[reg_idx], True
                return cmd.data, True
            else:
                raise ValueError(f"unknown opcode: {cmd.opcode}")

        return self.registers[0], True

    def _decode_call(self, cmd: VMCommand) -> tuple[str, int, bytes]:
        if len(cmd.data) < 52:
            raise ValueError("CALL payload requires at least 20-byte target + 32-byte value")

        target = "0x" + cmd.data[:20].hex()
        value = int.from_bytes(cmd.data[20:52], "big")
        calldata_reg = cmd.registers[0] if cmd.registers else 0
        self._require_register(calldata_reg)
        calldata = self.registers[calldata_reg]
        return target, value, calldata

    @staticmethod
    def _normalize_exec_result(exec_result: Any) -> tuple[bool, bytes]:
        if isinstance(exec_result, tuple):
            success, return_data = exec_result
        else:
            success, return_data = True, exec_result

        if isinstance(return_data, str):
            return_data = bytes.fromhex(return_data.removeprefix("0x"))
        elif isinstance(return_data, bytearray):
            return_data = bytes(return_data)
        return success, bytes(return_data)

    def _store_call_result(self, success: bool, return_data: bytes) -> None:
        if not success:
            raise ValueError("CALL execution failed")

        # register[0] stores raw return data from the last CALL.
        self.registers[0] = return_data

    def _calldata_build(self, cmd: VMCommand):
        """Load a calldata template into a register."""
        if not cmd.registers:
            raise ValueError("CALLDATA_BUILD requires registers[0] as target register")
        reg_idx = cmd.registers[0]
        self._require_register(reg_idx)
        self.registers[reg_idx] = cmd.data

    def _calldata_surgery(self, cmd: VMCommand):
        """Apply dynamic uint256 patches to a calldata register buffer."""
        if not cmd.registers:
            raise ValueError("CALLDATA_SURGERY requires registers for source and value inputs")
        source_reg = cmd.registers[0]
        self._require_register(source_reg)
        template = self.registers[source_reg]

        # data layout: 1-byte patch count + repeated 2-byte offsets.
        if len(cmd.data) < 1:
            raise ValueError("CALLDATA_SURGERY requires a non-empty payload with 1-byte patch count")
        surgery_count = cmd.data[0]
        if len(cmd.data) != 1 + surgery_count * 2:
            raise ValueError("CALLDATA_SURGERY payload length does not match patch count")
        if len(cmd.registers) != 1 + surgery_count:
            raise ValueError("CALLDATA_SURGERY registers length is insufficient")
        offset = 1
        for _ in range(surgery_count):
            patch_offset = int.from_bytes(cmd.data[offset : offset + 2], "big")
            value_reg = cmd.registers[1 + _]
            self._require_register(value_reg)
            dynamic_value = int.from_bytes(self.registers[value_reg], "big")

            template = Patcher.patch_calldata(template, [patch_offset], [dynamic_value])
            offset += 2

        self.registers[source_reg] = template

    def _ret_u256(self, cmd: VMCommand) -> None:
        """Read uint256 from last CALL returndata at the given offset."""
        if len(cmd.data) != 2:
            raise ValueError("RET_U256 requires 2-byte offset payload")

        target_reg = cmd.registers[0] if cmd.registers else 1
        self._require_register(target_reg)

        retdata = self.registers[0]
        offset = int.from_bytes(cmd.data, "big")
        if offset + 32 > len(retdata):
            raise ValueError("RET_U256 out of bounds")

        self.registers[target_reg] = retdata[offset : offset + 32]

    def _ret_slice(self, cmd: VMCommand) -> None:
        """Read bytes slice from last CALL returndata using offset and size."""
        if len(cmd.data) != 4:
            raise ValueError("RET_SLICE requires 4-byte payload (offset + size)")

        target_reg = cmd.registers[0] if cmd.registers else 1
        self._require_register(target_reg)

        retdata = self.registers[0]
        offset = int.from_bytes(cmd.data[:2], "big")
        size = int.from_bytes(cmd.data[2:], "big")
        end = offset + size
        if end > len(retdata):
            raise ValueError("RET_SLICE out of bounds")

        self.registers[target_reg] = retdata[offset:end]

    def _ret_last32(self, cmd: VMCommand) -> None:
        """Read the last 32 bytes from previous CALL returndata."""
        if len(cmd.data) != 0:
            raise ValueError("RET_LAST32 does not accept payload")

        target_reg = cmd.registers[0] if cmd.registers else 1
        self._require_register(target_reg)

        retdata = self.registers[0]
        if len(retdata) < 32:
            raise ValueError("RET_LAST32 requires returndata length >= 32")

        self.registers[target_reg] = retdata[-32:]

    def _perform_call(self, cmd: VMCommand):
        """Execute CALL-like command via injected executor or local fallback."""
        _, _, calldata = self._decode_call(cmd)

        if self._call_executor is not None:
            raise RuntimeError(
                "Synchronous CALL execution is not supported when a call executor is configured. "
                "Use the async execution path (e.g. execute_async)."
            )

        success, return_data = True, calldata

        self._store_call_result(success, return_data)

    async def _perform_call_async(self, cmd: VMCommand) -> None:
        """Async CALL execution for AsyncWeb3-backed workflows."""
        target, value, calldata = self._decode_call(cmd)

        if self._call_executor is not None:
            success, return_data = self._normalize_exec_result(await self._call_executor(target, calldata, value))
        else:
            success, return_data = True, calldata

        self._store_call_result(success, return_data)
