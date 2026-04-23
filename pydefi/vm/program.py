"""DeFiVM Program — SSA-style builder over Vyper's Venom IR.

A Pythonic value-flow API: methods that produce values return :class:`Value`
handles that the user threads through subsequent calls, and Venom's stack
allocator generates DUP / SWAP / POP automatically.  Quick example::

    from pydefi.vm import Program

    prog = Program()
    success = prog.call_contract(ROUTER, calldata)
    prog.assert_(success)
    prog.stop()
    bytecode = prog.build()

Patched call (the calldata pattern)::

    prog = Program()
    quote_ok = prog.call_contract(QUOTER, quote_calldata)
    prog.assert_(quote_ok)
    amount = prog.returndata_word(0)
    swap_ok = prog.call_contract(
        ROUTER,
        swap_template,
        patches={36: amount},  # write amount into the calldata at offset 36
    )
    prog.assert_(swap_ok)
    prog.stop()
    bytecode = prog.build()

Design notes
------------
``Program`` owns one :class:`vyper.venom.context.IRContext` and one
``main`` :class:`IRFunction` for its lifetime.  Each method that emits IR
appends instructions to the *current* basic block, exposed via
``self._builder.current_block()``.

**Memory model (DELEGATECALL execution).**  Programs are run by the
Analog-Labs interpreter via ``DELEGATECALL``, which means we manage
free-memory ourselves rather than relying on Venom's compile-time
``alloca``/``memtop``.  Layout matches the legacy builder:

* Registers:        ``mem[0x80 + i*32]`` for ``i`` in 0..15
* Free-mem pointer: ``mem[0x40]`` (defaults to 0x280 on first use)
* Dynamic buffers:  start at ``mem[0x280]``

**Calldata buffers.**  Static calldata for external calls is appended to a
Venom data section; the body emits ``codecopy`` from the section into a
fresh memory allocation, applies optional patches via ``mstore``, then
``call``.  Venom resolves the data-section label to its absolute byte
position in the compiled output, so no post-processing patches are needed.

**Termination.**  A ``Program`` is "open" until the user calls one of
:meth:`stop`, :meth:`return_`, :meth:`return_word`, :meth:`revert`, or a
control-flow primitive that terminates the current BB.  :meth:`build` adds
an implicit ``stop`` if the current BB is still open.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

from eth_abi.abi import encode, encode_with_hooks
from eth_contract.contract import ContractFunction
from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.basicblock import IRBasicBlock, IRLabel, IRLiteral, IRVariable
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

if TYPE_CHECKING:
    from eth_abi.hooks import EncodingContext

    from pydefi.types import Address


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Value:
    """Opaque handle to an SSA value (an EVM word produced at runtime).

    Created by methods on :class:`Program` (e.g. :meth:`Program.const`,
    :meth:`Program.add`, :meth:`Program.call_contract`).  Pass these handles
    as arguments to other ``Program`` methods to build a value-flow graph
    that Venom compiles to EVM bytecode.

    Users should treat :class:`Value` as opaque — direct operand access is
    intentionally not exposed.
    """

    __slots__ = ("_op",)

    def __init__(self, op: Union[IRVariable, IRLiteral]) -> None:
        self._op = op

    def __repr__(self) -> str:
        return f"Value({self._op!r})"


class Label:
    """Opaque handle to a basic block (jump target).

    Created by :meth:`Program.label`.  Use :meth:`Program.jump`,
    :meth:`Program.branch`, or :meth:`Program.goto` to direct control flow.
    """

    __slots__ = ("_bb", "name")

    def __init__(self, bb: IRBasicBlock, name: str) -> None:
        self._bb = bb
        self.name = name

    def __repr__(self) -> str:
        return f"Label({self.name!r})"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Anything acceptable in a ``Value`` slot — runtime SSA handle, plain ``int``
#: literal, or 20-byte ``bytes`` address (interpreted as big-endian uint256).
ValueLike = Union[Value, int, bytes]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pad32(data: bytes) -> bytes:
    """Return *data* zero-padded to the next 32-byte boundary."""
    return data.ljust((len(data) + 31) & ~31, b"\x00")


#: Placeholders that satisfy eth_abi's per-type encoders.  We can't use ``0``
#: for everything — the AddressEncoder rejects ``int`` and the BytesEncoder
#: needs ``bytes``.  Mapping covers the common static types used in DeFi
#: function signatures; unknown types fall back to ``0`` and surface an
#: encoding error if that's not accepted.
_ABI_PLACEHOLDERS: dict[str, object] = {
    "address": b"\x00" * 20,
    "bool": False,
    "bytes32": b"\x00" * 32,
    "bytes20": b"\x00" * 20,
    "bytes16": b"\x00" * 16,
    "bytes8": b"\x00" * 8,
    "bytes4": b"\x00" * 4,
}


class _ValuePlaceholder:
    """Internal placeholder that records the calldata offset of a Value during
    ABI encoding via :func:`eth_abi.encode_with_hooks`.
    """

    def __init__(self, value: Value, abi_type: str) -> None:
        self.value = value
        self.abi_type = abi_type
        self.offset: int | None = None

    def __call__(self, ctx: "EncodingContext") -> object:
        # 4 = function selector prefix.  ctx.offset is the byte position of
        # the encoded value within the args (start of its 32-byte slot for
        # static types — which is where Program.call_contract MSTOREs).
        self.offset = 4 + ctx.offset
        return _ABI_PLACEHOLDERS.get(self.abi_type, 0)


def _replace_values_with_placeholders(arg: object, abi_type: str, sink: list[_ValuePlaceholder]) -> object:
    """Recursively replace :class:`Value` instances with :class:`_ValuePlaceholder`.

    *abi_type* is the type string for *arg* (e.g. ``"uint256"``, ``"address"``,
    ``"(uint256,address)"`` for a struct).  Used to select an encoder-friendly
    placeholder when the current leaf is a :class:`Value`.

    Returns a fresh structure suitable for :func:`eth_abi.encode_with_hooks`.
    Non-Value leaves (int, str, bool, bytes) pass through unchanged.

    Container handling: tuple/list types are walked recursively but for
    Value leaves we currently only annotate with the outer type — sufficient
    for the static-scalar parameter shapes pydefi uses.  Patches inside
    nested tuples/lists fall back to the generic ``0`` placeholder.
    """
    if isinstance(arg, Value):
        ph = _ValuePlaceholder(arg, abi_type)
        sink.append(ph)
        return ph
    if isinstance(arg, tuple):
        return tuple(_replace_values_with_placeholders(item, "", sink) for item in arg)
    if isinstance(arg, list):
        return [_replace_values_with_placeholders(item, "", sink) for item in arg]
    return arg


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------


class Program:
    """SSA-style fluent builder over Vyper's Venom IR.

    Each instance owns one :class:`IRContext` and one ``main``
    :class:`IRFunction`; method calls append IR to the current basic block.
    Call :meth:`build` to compile the accumulated IR to EVM bytecode.

    Methods either:

    * **Produce a value** — return a :class:`Value` handle.  Examples:
      :meth:`const`, :meth:`add`, :meth:`call_contract`, :meth:`load_reg`.
    * **Have a side effect** — return ``None``.  Examples: :meth:`store_reg`,
      :meth:`assert_`, :meth:`stop`, :meth:`jump`.

    Wherever a method takes a ``ValueLike`` argument, you may pass a
    :class:`Value` returned by another method, a Python ``int`` (auto-wrapped
    as a constant), or 20 raw bytes (interpreted as a big-endian address).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._ctx = IRContext()  # type: ignore[operator]
        self._fn = self._ctx.create_function("main")
        self._ctx.entry_function = self._fn
        self._builder = VenomBuilder(self._ctx, self._fn)  # type: ignore[operator]
        self._data_section_counter = 0  # for unique label names

    # ------------------------------------------------------------------
    # Operand coercion
    # ------------------------------------------------------------------

    def _to_operand(self, v: ValueLike) -> Union[IRVariable, IRLiteral, int]:
        """Coerce a :class:`ValueLike` into something acceptable as a Venom operand.

        * :class:`Value`  → its underlying SSA / literal operand.
        * ``int``         → Venom accepts plain ``int`` literals directly.
        * ``bytes`` (len 20) → big-endian ``int`` (an EVM address).
        """
        if isinstance(v, Value):
            return v._op
        if isinstance(v, int):
            if v < 0:
                raise ValueError(f"operand must be non-negative, got {v}")
            return v
        if isinstance(v, (bytes, bytearray, memoryview)):
            if len(v) != 20:
                raise ValueError(
                    f"operand: bytes must be 20 (an address), got {len(v)} bytes; "
                    "use Program.const(int.from_bytes(...)) for arbitrary widths"
                )
            return int.from_bytes(bytes(v), "big")
        raise TypeError(f"operand: unsupported type {type(v).__name__}")

    def _wrap(self, op: Union[IRVariable, IRLiteral]) -> Value:
        return Value(op)

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def const(self, n: int) -> Value:
        """Return a :class:`Value` wrapping the constant uint256 *n*."""
        if n < 0:
            raise ValueError(f"const: value must be non-negative, got {n}")
        if n >= 1 << 256:
            raise ValueError(f"const: value {n} does not fit in uint256")
        return Value(IRLiteral(n))  # type: ignore[operator]

    def addr(self, a: "Address") -> Value:
        """Return a :class:`Value` for a 20-byte EVM address."""
        if len(a) != 20:
            raise ValueError(f"addr: address must be 20 bytes, got {len(a)}")
        return Value(IRLiteral(int.from_bytes(bytes(a), "big")))  # type: ignore[operator]

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def add(self, a: ValueLike, b: ValueLike) -> Value:
        """Wrapping uint256 ``a + b``."""
        return self._wrap(self._builder.add(self._to_operand(a), self._to_operand(b)))

    def sub(self, a: ValueLike, b: ValueLike) -> Value:
        """Saturating ``max(a - b, 0)``.

        Implemented as ``(a - b) * (a >= b)`` since EVM SUB wraps modulo 2^256.
        """
        a_op = self._to_operand(a)
        b_op = self._to_operand(b)
        # raw_diff = a - b (wraps if a < b)
        raw_diff = self._builder.sub(a_op, b_op)
        # not_underflow = 1 if a >= b else 0  ==  iszero(a < b)
        not_underflow = self._builder.iszero(self._builder.lt(a_op, b_op))
        return self._wrap(self._builder.mul(raw_diff, not_underflow))

    def mul(self, a: ValueLike, b: ValueLike) -> Value:
        """Wrapping uint256 ``a * b``."""
        return self._wrap(self._builder.mul(self._to_operand(a), self._to_operand(b)))

    def div(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``a // b``; EVM DIV returns 0 when ``b == 0``."""
        return self._wrap(self._builder.div(self._to_operand(a), self._to_operand(b)))

    def mod(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``a % b``; EVM MOD returns 0 when ``b == 0``."""
        return self._wrap(self._builder.mod(self._to_operand(a), self._to_operand(b)))

    # ------------------------------------------------------------------
    # Comparison / boolean
    # ------------------------------------------------------------------

    def lt(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``1 if a < b else 0``."""
        return self._wrap(self._builder.lt(self._to_operand(a), self._to_operand(b)))

    def gt(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``1 if a > b else 0``."""
        return self._wrap(self._builder.gt(self._to_operand(a), self._to_operand(b)))

    def eq(self, a: ValueLike, b: ValueLike) -> Value:
        """``1 if a == b else 0``."""
        return self._wrap(self._builder.eq(self._to_operand(a), self._to_operand(b)))

    def is_zero(self, a: ValueLike) -> Value:
        """``1 if a == 0 else 0``."""
        return self._wrap(self._builder.iszero(self._to_operand(a)))

    # ------------------------------------------------------------------
    # Bitwise
    # ------------------------------------------------------------------

    def bit_and(self, a: ValueLike, b: ValueLike) -> Value:
        return self._wrap(self._builder.and_(self._to_operand(a), self._to_operand(b)))

    def bit_or(self, a: ValueLike, b: ValueLike) -> Value:
        return self._wrap(self._builder.or_(self._to_operand(a), self._to_operand(b)))

    def bit_xor(self, a: ValueLike, b: ValueLike) -> Value:
        return self._wrap(self._builder.xor(self._to_operand(a), self._to_operand(b)))

    def bit_not(self, a: ValueLike) -> Value:
        return self._wrap(self._builder.not_(self._to_operand(a)))

    def shl(self, value: ValueLike, shift: ValueLike) -> Value:
        """``value << shift`` (Venom signature: ``shl(bits, val)``)."""
        return self._wrap(self._builder.shl(self._to_operand(shift), self._to_operand(value)))

    def shr(self, value: ValueLike, shift: ValueLike) -> Value:
        """``value >> shift`` (Venom signature: ``shr(bits, val)``)."""
        return self._wrap(self._builder.shr(self._to_operand(shift), self._to_operand(value)))

    # ------------------------------------------------------------------
    # Memory / Registers
    # ------------------------------------------------------------------

    def load_reg(self, i: int) -> Value:
        """Load 32-byte word from register *i* (memory slot ``0x80 + i*32``)."""
        if not 0 <= i <= 15:
            raise ValueError(f"load_reg: register index must be 0..15, got {i}")
        addr = 0x80 + i * 32
        return self._wrap(self._builder.mload(IRLiteral(addr)))  # type: ignore[arg-type]

    def store_reg(self, i: int, v: ValueLike) -> None:
        """Store *v* into register *i* (memory slot ``0x80 + i*32``)."""
        if not 0 <= i <= 15:
            raise ValueError(f"store_reg: register index must be 0..15, got {i}")
        addr = 0x80 + i * 32
        self._builder.mstore(IRLiteral(addr), self._to_operand(v))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Self / context
    # ------------------------------------------------------------------

    def self_addr(self) -> Value:
        """EVM ``ADDRESS`` — the running program's own address."""
        return self._wrap(self._builder.address())

    def gas_left(self) -> Value:
        """EVM ``GAS`` — remaining gas."""
        return self._wrap(self._builder.gas())

    # ------------------------------------------------------------------
    # Internal: free-memory and data-section helpers
    # ------------------------------------------------------------------

    def _alloc_calldata(self, calldata: bytes) -> tuple[IRVariable, int]:
        """Append *calldata* as a Venom data section, copy into free memory.

        Returns ``(base_fp, blen)`` where ``base_fp`` is the SSA pointer to
        the start of the buffer in memory and ``blen`` is the unpadded byte
        length (the value to pass as ``argsLen`` to ``CALL``).
        """
        if len(calldata) > 0xFFFF:
            raise ValueError(f"calldata too large ({len(calldata)} bytes, max 65535)")
        blen = len(calldata)
        blen_padded = (blen + 31) & ~31

        label = IRLabel(f"pydefi_calldata_{self._data_section_counter}", is_symbol=True)  # type: ignore[operator]
        self._data_section_counter += 1
        self._ctx.append_data_section(label)
        self._ctx.append_data_item(_pad32(calldata))

        # base_fp = mem[0x40] | (iszero(mem[0x40]) * 0x280)  — defaults to 0x280
        fp = self._builder.mload(IRLiteral(0x40))  # type: ignore[arg-type]
        default_fp = self._builder.mul(self._builder.iszero(fp), 0x280)
        base_fp = self._builder.or_(fp, default_fp)
        # codecopy(base_fp, &calldata, blen_padded)
        data_src = self._builder.offset(0, label)
        self._builder.codecopy(base_fp, data_src, blen_padded)
        # mem[0x40] = base_fp + blen_padded
        self._builder.mstore(IRLiteral(0x40), self._builder.add(base_fp, blen_padded))  # type: ignore[arg-type]
        return base_fp, blen

    def _alloc(self, size: int) -> IRVariable:
        """Reserve *size* bytes in free memory, advance ``mem[0x40]``, return base."""
        fp = self._builder.mload(IRLiteral(0x40))  # type: ignore[arg-type]
        default_fp = self._builder.mul(self._builder.iszero(fp), 0x280)
        base = self._builder.or_(fp, default_fp)
        self._builder.mstore(IRLiteral(0x40), self._builder.add(base, size))  # type: ignore[arg-type]
        return base

    # ------------------------------------------------------------------
    # External calls
    # ------------------------------------------------------------------

    def call_contract(
        self,
        to: ValueLike,
        calldata: bytes,
        *,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
        patches: Mapping[int, ValueLike] | None = None,
    ) -> Value:
        """Emit a CALL with static calldata and optional runtime patches.

        Args:
            to:        Target contract address.
            calldata:  Pre-encoded calldata template; stored in a Venom data
                       section and copied into memory before the call.
            value:     ETH value to forward (wei).
            gas:       Gas limit; ``None`` forwards all remaining gas.
            patches:   Optional ``{calldata_offset: value}`` overlay.  Each
                       entry overwrites a 32-byte word in the in-memory
                       calldata buffer at the given offset before CALL.

        Returns:
            A :class:`Value` holding the CALL success flag (1 on success,
            0 on failure).  Use :meth:`assert_` to revert on failure.
        """
        base_fp, blen = self._alloc_calldata(calldata)

        if patches:
            for offset, val in patches.items():
                if not 0 <= offset <= blen - 32:
                    raise ValueError(
                        f"call_contract: patch offset {offset} out of bounds "
                        f"(calldata length {blen}, must satisfy 0 <= offset <= len-32)"
                    )
                target_addr = self._builder.add(base_fp, offset)
                self._builder.mstore(target_addr, self._to_operand(val))

        gas_op = self._builder.gas() if gas is None else self._to_operand(gas)
        success = self._builder.call(
            gas_op,
            self._to_operand(to),
            self._to_operand(value),
            base_fp,
            blen,
            0,
            0,
        )
        return self._wrap(success)

    def call_contract_abi(
        self,
        to: ValueLike,
        abi_sig: str,
        *args: object,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
    ) -> Value:
        """Emit a CALL with calldata built from a human-readable ABI signature.

        Plain Python values (``int``, ``str`` address, ``bool``, ``bytes``,
        nested ``tuple``/``list``) are statically encoded.  :class:`Value`
        instances anywhere in the argument tree become *runtime patches* —
        the ABI placeholder is encoded as ``0`` and ``mstore`` overwrites it
        in the in-memory calldata buffer with the SSA value before CALL.

        The ``function`` keyword in *abi_sig* is optional; both bare
        ``"transfer(address,uint256)"`` and qualified ``"function transfer(...)"``
        forms are accepted.

        Returns:
            A :class:`Value` holding the CALL success flag.
        """
        normalised = abi_sig if abi_sig.lstrip().startswith("function ") else "function " + abi_sig
        fn = ContractFunction.from_abi(normalised)
        param_types = fn.input_types

        if len(args) != len(param_types):
            raise ValueError(
                f"call_contract_abi: expected {len(param_types)} argument(s) for signature {abi_sig!r}, got {len(args)}"
            )

        # Walk the arg tree, replacing each Value with a placeholder hook that
        # records the value's calldata offset during ABI encoding.
        placeholders: list[_ValuePlaceholder] = []
        encoded_args: list[object] = [
            _replace_values_with_placeholders(a, t, placeholders) for a, t in zip(args, param_types)
        ]

        if not placeholders:
            calldata = bytes(fn.selector) + encode(param_types, encoded_args)
            return self.call_contract(to, calldata, value=value, gas=gas)

        encoded = encode_with_hooks(param_types, encoded_args)
        calldata = bytes(fn.selector) + encoded
        patches: dict[int, ValueLike] = {p.offset: p.value for p in placeholders if p.offset is not None}
        return self.call_contract(to, calldata, value=value, gas=gas, patches=patches)

    # ------------------------------------------------------------------
    # Returndata
    # ------------------------------------------------------------------

    def returndata_word(self, offset: int = 0) -> Value:
        """Read 32 bytes from the last call's returndata at *offset*."""
        if offset < 0:
            raise ValueError(f"returndata_word: offset must be non-negative, got {offset}")
        scratch = self._alloc(32)
        self._builder.returndatacopy(scratch, offset, 32)
        return self._wrap(self._builder.mload(scratch))

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_(self, cond: ValueLike, msg: str = "") -> None:
        """Revert if *cond* is zero.

        With ``msg=""`` (default) emits a bare REVERT(0, 0).  With a non-empty
        message ≤ 32 bytes, encodes ``Error(string)`` ABI in memory and
        REVERTs with that payload — matching Solidity's
        ``require(cond, "message")``.
        """
        if not msg:
            self._builder.assert_(self._to_operand(cond))
            return

        raw = msg.encode()
        if len(raw) > 32:
            raise ValueError(f"assert_: message too long ({len(raw)} bytes, max 32)")

        # Conditional revert: if cond == 0 → revert with Error(string), else fall through.
        ok_bb = self._builder.create_block("assert_ok")
        revert_bb = self._builder.create_block("assert_revert")
        self._builder.append_block(ok_bb)
        self._builder.append_block(revert_bb)
        self._builder.jnz(self._to_operand(cond), ok_bb.label, revert_bb.label)

        # Build Error(string) payload in revert_bb:
        #   mem[fp+0..4]     = selector 0x08c379a0
        #   mem[fp+4..36]    = offset 0x20
        #   mem[fp+36..68]   = msg length
        #   mem[fp+68..68+padded] = msg bytes
        self._builder.set_block(revert_bb)
        msglen = len(raw)
        msg_word = int.from_bytes(raw.ljust(32, b"\x00"), "big")
        # Selector left-aligned in a 32-byte slot at fp+0; mstore(fp, selector_word)
        # writes the selector at fp+0..4 with zeros at fp+4..32 — but we then
        # overwrite fp+4..36 with the offset, so net layout is correct.
        selector_word = 0x08C379A000000000000000000000000000000000000000000000000000000000
        base = self._alloc(100)  # 4 + 32 + 32 + 32
        self._builder.mstore(base, selector_word)
        self._builder.mstore(self._builder.add(base, 4), 0x20)
        self._builder.mstore(self._builder.add(base, 36), msglen)
        self._builder.mstore(self._builder.add(base, 68), msg_word)
        self._builder.revert(base, 100)

        # Continue at ok_bb
        self._builder.set_block(ok_bb)

    def assert_ge(self, a: ValueLike, b: ValueLike, msg: str = "") -> None:
        """Revert if ``a < b`` (i.e. require ``a >= b``)."""
        a_op = self._to_operand(a)
        b_op = self._to_operand(b)
        not_lt = self._builder.iszero(self._builder.lt(a_op, b_op))
        self.assert_(self._wrap(not_lt), msg)

    def assert_le(self, a: ValueLike, b: ValueLike, msg: str = "") -> None:
        """Revert if ``a > b`` (i.e. require ``a <= b``)."""
        a_op = self._to_operand(a)
        b_op = self._to_operand(b)
        not_gt = self._builder.iszero(self._builder.gt(a_op, b_op))
        self.assert_(self._wrap(not_gt), msg)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def label(self, name: str) -> Label:
        """Create a fresh basic block and return its :class:`Label`.

        Does NOT switch the insertion point — call :meth:`goto` to start
        emitting into the new block.  This separation lets you create a label
        that you'll jump to from multiple places before populating its body.

        The block is appended to the function immediately so that it can be
        referenced by :meth:`jump` / :meth:`branch` before being populated.
        """
        bb = self._builder.create_block(name)
        self._builder.append_block(bb)
        return Label(bb, name)

    def goto(self, target: Label) -> None:
        """Switch the insertion point to *target*'s basic block.

        The previously-current block must already be terminated (e.g. via
        :meth:`jump`, :meth:`branch`, :meth:`stop`, …).
        """
        self._builder.set_block(target._bb)

    def jump(self, target: Label) -> None:
        """Terminate the current block with an unconditional ``jmp`` to *target*."""
        self._builder.jmp(target._bb.label)

    def branch(self, cond: ValueLike, *, true: Label, false: Label) -> None:
        """Terminate the current block with conditional ``jnz``.

        Jumps to *true* if ``cond != 0``, else to *false*.
        """
        self._builder.jnz(self._to_operand(cond), true._bb.label, false._bb.label)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Terminate the current block with ``STOP`` (halt, no return data)."""
        self._builder.stop()

    def return_(self, offset: ValueLike, length: ValueLike) -> None:
        """Terminate with ``RETURN(offset, length)`` — return memory slice."""
        self._builder.return_(self._to_operand(offset), self._to_operand(length))

    def return_word(self, value: ValueLike) -> None:
        """Convenience: store *value* at ``mem[0]`` and ``RETURN(0, 32)``."""
        self._builder.mstore(IRLiteral(0), self._to_operand(value))  # type: ignore[arg-type]
        self._builder.return_(IRLiteral(0), IRLiteral(32))  # type: ignore[arg-type]

    def revert(self, offset: ValueLike = 0, length: ValueLike = 0) -> None:
        """Terminate with ``REVERT(offset, length)`` — revert with memory slice."""
        self._builder.revert(self._to_operand(offset), self._to_operand(length))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        optimize: OptimizationLevel = OptimizationLevel.GAS,  # type: ignore[valid-type]
        disable_constant_folding: bool = False,
    ) -> bytes:
        """Compile the accumulated IR via Venom and return EVM bytecode.

        If the current basic block is not terminated, an implicit ``STOP`` is
        inserted before compilation.

        Args:
            optimize: Venom optimization level.  Defaults to ``GAS`` for
                production builds.
            disable_constant_folding: Disable SCCP, algebraic optimization,
                and assert elimination.  Useful in tests that exercise
                runtime behavior with constant inputs — without this,
                ``assert_(prog.const(0))`` raises a Venom
                ``StaticAssertionException`` at compile time because SCCP
                proves the assertion will fail.  Real programs whose
                conditions depend on call returndata or other runtime data
                are unaffected.
        """
        if not self._builder.is_terminated():
            self._builder.stop()

        if disable_constant_folding:
            flags = VenomOptimizationFlags(  # type: ignore[call-arg]
                level=optimize,
                disable_sccp=True,
                disable_algebraic_optimization=True,
                disable_assert_elimination=True,
                disable_branch_optimization=True,
            )
        else:
            flags = VenomOptimizationFlags(level=optimize)  # type: ignore[call-arg]
        run_passes_on(self._ctx, flags, disable_mem_checks=True)  # type: ignore[misc]
        asm = generate_assembly_experimental(self._ctx, optimize=optimize)  # type: ignore[misc]
        bytecode, _ = generate_bytecode(asm)  # type: ignore[misc]
        return bytecode

    # ------------------------------------------------------------------
    # Convenience dunders
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        return self.build()

    def __repr__(self) -> str:
        return f"Program(data_sections={self._data_section_counter})"
