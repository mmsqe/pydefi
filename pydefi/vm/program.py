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
    swap_tmpl = prog.template(
        "function swapExactTokensForTokens(uint256 amountIn, uint256 amountOutMin,"
        " address[] path, address to, uint256 deadline)"
    )
    swap_ok = prog.call_contract(
        ROUTER,
        swap_tmpl(amountIn=amount, amountOutMin=0, path=PATH, to=RECIPIENT, deadline=DL),
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

**Calldata buffers.**  Static calldata for external calls is appended to a
Venom data section; the body emits ``codecopy`` from the section into a
fresh allocation, applies optional patches via ``mstore``, then ``call``.
Venom resolves the data-section label to its absolute byte position in
the compiled output, so no post-processing patches are needed.

**Termination.**  A ``Program`` is "open" until the user calls one of
:meth:`stop`, :meth:`return_`, :meth:`return_word`, :meth:`revert`, or a
control-flow primitive that terminates the current BB.  :meth:`build` adds
an implicit ``stop`` if the current BB is still open.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

from eth_abi.abi import encode as _abi_encode
from eth_abi.abi import encode_with_hooks
from eth_abi.encoding import AddressEncoder, BooleanEncoder, BytesEncoder
from eth_contract.contract import ContractFunction
from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
from vyper.evm.assembler.instructions import (
    CONST,
    DATA_ITEM,
    PUSH_OFST,
    PUSHLABEL,
    DataHeader,
)
from vyper.evm.assembler.instructions import Label as _AsmLabel
from vyper.evm.assembler.symbols import SYMBOL_SIZE
from vyper.evm.opcodes import OPCODES
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.basicblock import IRLabel, IRLiteral, IRVariable
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

if TYPE_CHECKING:
    from eth_abi.hooks import EncodingContext

    from pydefi.types import Address


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: Handle to an SSA value (an EVM word produced at runtime).
#:
#: Returned by methods on :class:`Program` (e.g. :meth:`Program.const`,
#: :meth:`Program.add`, :meth:`Program.call_contract`).  Pass these handles
#: as arguments to other ``Program`` methods to build a value-flow graph
#: that Venom compiles to EVM bytecode.
#:
#: Currently an alias for Venom's operand types — an ``IRVariable`` for
#: computed values, an ``IRLiteral`` for constants.  The union is what the
#: Venom builder accepts as an operand, so no wrapper is needed.
Value = Union[IRVariable, IRLiteral]

#: Anything acceptable in a ``Value`` slot — runtime SSA handle, plain ``int``
#: literal, or 20-byte ``bytes`` address (interpreted as big-endian uint256).
ValueLike = Union[Value, int, bytes]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _placeholder_for_encoder(encoder: object) -> object:
    """Return a value that the given ``eth_abi`` encoder will accept.

    We can't use ``0`` for everything — ``AddressEncoder`` rejects ``int``
    and ``BytesEncoder`` needs ``bytes`` of the correct width.  Unknown
    encoders fall back to ``0`` and surface an encoding error if that's
    not accepted.
    """
    if isinstance(encoder, AddressEncoder):
        return b"\x00" * 20
    if isinstance(encoder, BooleanEncoder):
        return False
    if isinstance(encoder, BytesEncoder):
        return b"\x00" * (encoder.value_bit_size // 8)
    return 0


#: Types that occupy exactly one 32-byte head slot and can therefore be
#: lifted out of the blob into an MSTORE patch without disturbing ``eth_abi``'s
#: head/tail layout.  Matches ``uint{N}``, ``int{N}``, ``address``, ``bool``,
#: and ``bytes1..bytes32``.  Deliberately excludes ``bytes``, ``string``,
#: tuples, and arrays — those carry dynamic tails and must bake into the blob
#: to keep offsets consistent across invocations.
_SCALAR_STATIC_RE = re.compile(r"^(u?int\d*|address|bool|bytes([1-9]|[12]\d|3[012]))$")


def _is_scalar_static_type(abi_type: str) -> bool:
    return bool(_SCALAR_STATIC_RE.match(abi_type))


def _encode_static_to_slot_int(val: object, abi_type: str) -> int:
    """Encode *val* as its 32-byte ABI head slot and return it as a uint256.

    Delegates to ``eth_abi.encode`` so every static type (signed ints,
    addresses, bytesN padding, bool) is handled exactly as eth_abi would
    handle it — the patched MSTORE is byte-identical to what a baked-in
    blob would have contained.
    """
    # _normalise_static handles the int→20-byte address coercion that
    # eth_abi's AddressEncoder otherwise rejects.
    coerced = _normalise_static(val, abi_type)
    encoded = _abi_encode([abi_type], [coerced])
    assert len(encoded) == 32, f"expected 32-byte scalar-static encoding for {abi_type!r}, got {len(encoded)}"
    return int.from_bytes(encoded, "big")


class Placeholder:
    """Encoding hook that zeros out an ABI head slot in the encoded blob and
    remembers where to MSTORE its real value before ``CALL``.

    :class:`CalldataTemplate` wraps any top-level scalar-static arg (runtime
    :class:`Value` or non-zero literal) in a ``Placeholder``, and wraps
    runtime :class:`Value` leaves nested inside dynamic containers. The
    encoder hook invokes ``__call__(ctx)`` at each slot position; we record
    ``4 + ctx.offset`` (skipping the 4-byte selector) and return a
    type-appropriate zero so ``eth_abi`` accepts the slot.

    The ``value`` field is a :data:`ValueLike` — an SSA :class:`Value`
    handle for runtime patches, or a ``uint256`` int for literal patches.
    Either way :meth:`Program.call_contract` pipes it through ``_to_operand``
    when emitting the MSTORE, so the same code path handles both.
    """

    def __init__(self, value: ValueLike) -> None:
        self.value = value
        self.offset: int | None = None

    def __call__(self, ctx: "EncodingContext") -> object:
        # 4 = function selector prefix.  ctx.offset is the byte position of
        # the encoded value within the args (start of its 32-byte slot for
        # static types — which is where Program.call_contract MSTOREs).
        self.offset = 4 + ctx.offset
        return _placeholder_for_encoder(ctx.encoder)


class CalldataPayload:
    """Bundled ``(blob, patches)`` pair produced by :class:`CalldataTemplate`.

    :meth:`Program.call_contract` accepts this in place of a raw ``bytes``
    payload: the blob goes through the normal data-section + CODECOPY path
    (with dedup) and the patches are MSTORE'd over the 32-byte slots.
    """

    __slots__ = ("blob", "patches")

    def __init__(self, blob: bytes, patches: "dict[int, ValueLike]") -> None:
        self.blob = blob
        self.patches = patches


class CalldataTemplate:
    """Reusable ABI calldata template.

    Parse a signature once, invoke it many times with mixed static and
    runtime values. Each invocation runs :func:`eth_abi.encode_with_hooks`
    to produce a :class:`CalldataPayload`:

    * Static Python values (``int``, ``bytes``, tuples, addresses, nested
      arrays) are encoded directly into the blob.
    * :class:`Value` arguments become :class:`Placeholder` hooks whose
      calldata offsets are captured during encoding; they appear as
      MSTORE patches applied to the in-memory buffer before ``CALL``.

    Two invocations of the same template share one Venom data section
    whenever their encoded blobs are identical (i.e. same static arg
    values + same number and placement of Value placeholders).

    Supports both static and dynamic ABI types — anything
    ``eth_abi.encode_with_hooks`` accepts.

    Example::

        xfer = p.template("transfer(address to, uint256 amount)")
        for recipient, amount in rows:
            p.call_contract(weth, xfer(to=recipient, amount=amount))
    """

    __slots__ = ("_signature", "_selector", "_param_types", "_param_names")

    def __init__(self, signature: str, names: "list[str] | None" = None) -> None:
        normalised = signature if signature.lstrip().startswith("function ") else "function " + signature
        fn = ContractFunction.from_abi(normalised)
        self._signature = fn.signature
        self._selector = bytes(fn.selector)
        self._param_types: list[str] = list(fn.input_types)

        # Prefer names from the ABI sig (Solidity style:
        # ``transfer(address to, uint256 amount)``); fall back to *names*
        # kwarg; else the params are positional-only.
        abi_names = [inp.get("name") or "" for inp in fn.abi["inputs"]]
        if names is not None:
            if len(names) != len(self._param_types):
                raise ValueError(f"template: {len(self._param_types)} parameters but {len(names)} names provided")
            self._param_names: list[str] = list(names)
        elif all(abi_names):
            self._param_names = abi_names
        else:
            self._param_names = []

    @property
    def signature(self) -> str:
        return self._signature

    @property
    def param_types(self) -> "list[str]":
        return list(self._param_types)

    @property
    def param_names(self) -> "list[str]":
        return list(self._param_names)

    def _bind_args(self, args: tuple[object, ...], kwargs: dict[str, object]) -> list[object]:
        """Resolve positional / kwarg input against the template signature.

        Accepts either all-positional or all-kwarg form, not a mix.  For
        kwargs, every declared param name must be supplied and no extras
        are allowed.  The returned list is in declaration order.
        """
        n = len(self._param_types)
        if args and kwargs:
            raise TypeError("template: pass positional args or kwargs, not both")
        if n == 0:
            if args or kwargs:
                raise TypeError(f"template {self._signature!r} takes no arguments")
            return []
        if args:
            if len(args) != n:
                raise TypeError(f"template {self._signature!r} takes {n} positional argument(s), got {len(args)}")
            return list(args)
        if not self._param_names:
            raise TypeError(
                f"template {self._signature!r} has no parameter names; "
                "call with positional args or provide names= when creating it"
            )
        extra = [k for k in kwargs if k not in self._param_names]
        if extra:
            raise TypeError(f"template {self._signature!r} got unexpected keyword argument(s): {extra}")
        missing = [nm for nm in self._param_names if nm not in kwargs]
        if missing:
            raise TypeError(f"template {self._signature!r} missing keyword argument(s): {missing}")
        return [kwargs[name] for name in self._param_names]

    def __call__(self, *args: object, **kwargs: object) -> CalldataPayload:
        values = self._bind_args(args, kwargs)

        # Blob dedup strategy, uniformly via :class:`Placeholder` hooks:
        #   - Scalar-static slots (uint/int/address/bool/bytesN) with either
        #     a runtime Value or a non-zero literal are wrapped in a
        #     Placeholder. Its hook zeros out the head slot in the blob and
        #     records the slot's calldata offset; the real value is MSTORE'd
        #     before CALL. Zero literals bake a type-appropriate zero into
        #     the blob directly — the blob already carries the right word,
        #     so we save an MSTORE. Either way the blob stays byte-identical
        #     across invocations that differ only in scalar-static args, so
        #     :attr:`Program._blob_labels` shares one data section.
        #   - Dynamic / container slots (bytes, string, T[], tuples) bake
        #     into the blob via encode_with_hooks; nested runtime Values
        #     still become Placeholders via _replace_values_with_placeholders.
        placeholders: list[Placeholder] = []
        encoded_args: list[object] = []
        for val, t in zip(values, self._param_types):
            if _is_scalar_static_type(t):
                if isinstance(val, Value):
                    ph = Placeholder(val)
                else:
                    slot_int = _encode_static_to_slot_int(val, t)
                    if slot_int == 0:
                        # Blob already zero at this slot — don't track the
                        # Placeholder (no patch) but still let its hook run
                        # so the encoder gets a type-correct zero stand-in.
                        encoded_args.append(Placeholder(0))
                        continue
                    ph = Placeholder(slot_int)
                placeholders.append(ph)
                encoded_args.append(ph)
            else:
                encoded_args.append(_replace_values_with_placeholders(val, t, placeholders))
        encoded = encode_with_hooks(self._param_types, encoded_args)
        blob = self._selector + encoded

        # Every Placeholder must have had its offset recorded by the encoder
        # hook.  An unresolved one would silently leave the ABI zero stand-in
        # in the blob at runtime and produce a subtly wrong call — fail loud.
        unresolved = [i for i, p in enumerate(placeholders) if p.offset is None]
        if unresolved:
            raise ValueError(
                f"template {self._signature!r}: failed to resolve calldata offset "
                f"for {len(unresolved)} Placeholder(s) — possibly nested in a "
                "tuple/list type the encoder didn't traverse with hooks"
            )
        patches: dict[int, ValueLike] = {p.offset: p.value for p in placeholders if p.offset is not None}
        return CalldataPayload(blob, patches)


def _asm_item_size(item: object) -> int:
    """Byte size that *item* contributes when assembled to EVM bytecode.

    Mirrors :func:`vyper.evm.assembler.core._assembly_to_evm` for the asm item
    types Venom emits.  Used to walk the asm list and locate byte positions of
    label-resolved PUSH_OFST / PUSHLABEL instructions when post-processing the
    bytecode.
    """
    if item == "DEBUG" or isinstance(item, (CONST, DataHeader)):
        return 0
    if isinstance(item, PUSHLABEL):
        return 1 + SYMBOL_SIZE
    if isinstance(item, _AsmLabel):
        return 1  # JUMPDEST
    if isinstance(item, PUSH_OFST):
        if isinstance(item.label, _AsmLabel):
            return 1 + SYMBOL_SIZE
        raise NotImplementedError(f"_asm_item_size: CONSTREF PUSH_OFST not supported: {item!r}")
    if isinstance(item, int):
        return 1
    if isinstance(item, str):
        up = item.upper()
        if up.startswith("PUSH") and up != "PUSH":
            return 1
        if up.startswith("DUP") or up.startswith("SWAP"):
            return 1
        if up in OPCODES:
            return 1
        raise ValueError(f"_asm_item_size: unrecognised asm string item: {item!r}")
    if isinstance(item, DATA_ITEM):
        if isinstance(item.data, bytes):
            return len(item.data)
        if isinstance(item.data, _AsmLabel):
            return SYMBOL_SIZE
        raise ValueError(f"_asm_item_size: unrecognised DATA_ITEM payload: {item.data!r}")
    raise ValueError(f"_asm_item_size: unknown asm item type: {type(item).__name__}")


def _shift_label_pushes(asm: list, bytecode: bytes, shift: int) -> bytes:
    """Add *shift* to every label-resolved ``PUSH_OFST`` / ``PUSHLABEL`` immediate.

    Used when the compiled program will be embedded inside a larger bytecode
    buffer at runtime — e.g. CCTP / OFT composer contracts prepend a 66-byte
    PUSH32 prologue before our program runs.  Venom resolves data-section and
    jump-target labels to absolute byte offsets at compile time; if the
    program is then shifted, those references read / jump to the wrong place.
    Adding *shift* to each one keeps them correct.
    """
    if shift == 0:
        return bytecode
    buf = bytearray(bytecode)
    pos = 0
    for item in asm:
        size = _asm_item_size(item)
        if isinstance(item, (PUSH_OFST, PUSHLABEL)) and isinstance(getattr(item, "label", None), _AsmLabel):
            imm_pos = pos + 1  # immediate sits after the 1-byte PUSH<SYMBOL_SIZE> opcode
            current = int.from_bytes(buf[imm_pos : imm_pos + SYMBOL_SIZE], "big")
            shifted = current + shift
            if shifted >> (SYMBOL_SIZE * 8) != 0:
                raise ValueError(
                    f"_shift_label_pushes: shifted offset {shifted} does not fit in "
                    f"{SYMBOL_SIZE} bytes (label={item.label!r}, shift={shift})"
                )
            buf[imm_pos : imm_pos + SYMBOL_SIZE] = shifted.to_bytes(SYMBOL_SIZE, "big")
        pos += size
    return bytes(buf)


def _normalise_static(val: object, abi_type: str) -> object:
    """Coerce user-friendly static values to shapes eth_abi accepts.

    Only the *scalar* forms that pydefi users commonly pass are handled:
    ``int`` for an ``address`` slot is turned into a 20-byte big-endian
    ``bytes``. Everything else (including runtime :class:`Value` handles
    and containers) passes through unchanged for downstream hooks.
    """
    if abi_type == "address" and isinstance(val, int) and not isinstance(val, bool):
        if val < 0 or val >= (1 << 160):
            raise ValueError(f"template: address literal {val} does not fit in 20 bytes")
        return val.to_bytes(20, "big")
    return val


def _element_type(abi_type: str) -> str:
    """Return the element type of an ABI array type (``T[]`` or ``T[N]``).

    ``"address[]"`` → ``"address"``, ``"uint256[3]"`` → ``"uint256"``,
    ``"bytes32[][]"`` → ``"bytes32[]"`` (strips one level only).  If
    *abi_type* is not an array type, returns it unchanged.
    """
    if abi_type.endswith("]"):
        idx = abi_type.rfind("[")
        if idx > 0:
            return abi_type[:idx]
    return abi_type


def _tuple_component_types(abi_type: str) -> list[str]:
    """Split a tuple ABI type string ``(T1,T2,...)`` into its components.

    ``"(address,uint256)"`` → ``["address", "uint256"]``,
    ``"(address,(uint256,uint256))"`` → ``["address", "(uint256,uint256)"]``.
    Returns ``[]`` if *abi_type* is not a tuple type.

    Respects nesting — commas inside nested tuples / fixed arrays do not
    split components at the outer level.
    """
    if not (abi_type.startswith("(") and abi_type.endswith(")")):
        return []
    inner = abi_type[1:-1]
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur += ch
    if cur:
        parts.append(cur)
    return parts


def _replace_values_with_placeholders(arg: object, abi_type: str, sink: list[Placeholder]) -> object:
    """Recursively replace :class:`Value` instances with :class:`Placeholder`.

    *abi_type* is the type string for *arg* (e.g. ``"uint256"``,
    ``"address"``, ``"(uint256,address)"`` for a struct, ``"address[]"``
    for a dynamic array).  Threaded down through nested tuples/lists so
    that nested ``int`` leaves at ``address`` slots are coerced to 20-byte
    form via :func:`_normalise_static`.

    The :class:`Placeholder` itself derives its encoder-friendly stand-in
    from :class:`EncodingContext` at hook-call time, so the type is not
    needed to pick a type-appropriate placeholder — it is threaded here
    only for the static-leaf normalisation path.

    Returns a fresh structure suitable for :func:`eth_abi.encode_with_hooks`.
    Non-Value leaves (int, str, bool, bytes) pass through ``_normalise_static``.
    """
    if isinstance(arg, Value):
        ph = Placeholder(arg)
        sink.append(ph)
        return ph
    if isinstance(arg, tuple):
        component_types = _tuple_component_types(abi_type)
        # Fall back to empty-string typing only if the type string didn't
        # look like a tuple — this preserves the old behavior for callers
        # that pass non-tuple types but still happen to use a tuple arg.
        if len(component_types) != len(arg):
            component_types = [""] * len(arg)
        return tuple(_replace_values_with_placeholders(item, t, sink) for item, t in zip(arg, component_types))
    if isinstance(arg, list):
        elem_type = _element_type(abi_type)
        return [_replace_values_with_placeholders(item, elem_type, sink) for item in arg]
    # Non-Value scalar leaf: normalise before handing off to eth_abi.  This
    # matters in nested containers too — e.g. an ``int`` in an ``address[]``
    # element must be coerced to 20-byte bytes, same as a top-level
    # ``address`` arg.
    return _normalise_static(arg, abi_type)


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
      :meth:`const`, :meth:`add`, :meth:`call_contract`, :meth:`load_slot`.
    * **Have a side effect** — return ``None``.  Examples: :meth:`store_slot`,
      :meth:`assert_`, :meth:`stop`, :meth:`jump`.

    Wherever a method takes a ``ValueLike`` argument, you may pass a
    :class:`Value` returned by another method, a Python ``int`` (auto-wrapped
    as a constant), or 20 raw bytes (interpreted as a big-endian address).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._ctx = IRContext()
        self._fn = self._ctx.create_function("main")
        self._ctx.entry_function = self._fn
        self._builder = VenomBuilder(self._ctx, self._fn)
        self._data_section_counter = 0  # for unique label names
        # Dedup identical calldata blobs — same bytes → same data section →
        # same IRLabel. Saves bytecode size when the same template is reused
        # across many call_contract / template invocations.
        self._blob_labels: dict[bytes, IRLabel] = {}

    # ------------------------------------------------------------------
    # Operand coercion
    # ------------------------------------------------------------------

    def _to_operand(self, v: ValueLike) -> Union[IRVariable, IRLiteral, int]:
        """Coerce a :class:`ValueLike` into something acceptable as a Venom operand.

        * :class:`Value` (``IRVariable`` / ``IRLiteral``) → returned as-is.
        * ``int``            → Venom accepts plain ``int`` literals directly.
        * ``bytes`` (len 20) → big-endian ``int`` (an EVM address).
        """
        if isinstance(v, (IRVariable, IRLiteral)):
            return v
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

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def const(self, n: int) -> Value:
        """Return a :class:`Value` wrapping the constant uint256 *n*."""
        if n < 0:
            raise ValueError(f"const: value must be non-negative, got {n}")
        if n >= 1 << 256:
            raise ValueError(f"const: value {n} does not fit in uint256")
        return IRLiteral(n)

    def addr(self, a: "Address") -> Value:
        """Return a :class:`Value` for a 20-byte EVM address."""
        if len(a) != 20:
            raise ValueError(f"addr: address must be 20 bytes, got {len(a)}")
        return IRLiteral(int.from_bytes(bytes(a), "big"))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def stack_param(self) -> Value:
        """Consume a value that was already on the EVM stack when the program started.

        Used when the caller (e.g. the CCTP / OFT composer prologue in
        ``DeFiVM.sol``) pushes values onto the stack before dispatching to
        the user program.  Each call consumes one pre-existing stack slot
        in **Venom ``param`` order**: the first call returns the *deepest*
        slot (the value pushed first by the prologue), the second call
        returns the slot above it, and so on — last call returns TOS.
        Callers must therefore request ``stack_param()`` values in the same
        order the prologue pushed them.

        Lowers to Venom's ``param`` instruction which emits no bytecode —
        it's a dataflow marker telling Venom's stack allocator that the
        variable is already on the stack.  Must be called before emitting
        any other value-producing instruction in the entry basic block.
        """
        return self._builder.param()

    def add(self, a: ValueLike, b: ValueLike) -> Value:
        """Wrapping uint256 ``a + b``."""
        return self._builder.add(self._to_operand(a), self._to_operand(b))

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
        return self._builder.mul(raw_diff, not_underflow)

    def mul(self, a: ValueLike, b: ValueLike) -> Value:
        """Wrapping uint256 ``a * b``."""
        return self._builder.mul(self._to_operand(a), self._to_operand(b))

    def div(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``a // b``; EVM DIV returns 0 when ``b == 0``."""
        return self._builder.div(self._to_operand(a), self._to_operand(b))

    def mod(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``a % b``; EVM MOD returns 0 when ``b == 0``."""
        return self._builder.mod(self._to_operand(a), self._to_operand(b))

    # ------------------------------------------------------------------
    # Comparison / boolean
    # ------------------------------------------------------------------

    def lt(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``1 if a < b else 0``."""
        return self._builder.lt(self._to_operand(a), self._to_operand(b))

    def gt(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``1 if a > b else 0``."""
        return self._builder.gt(self._to_operand(a), self._to_operand(b))

    def eq(self, a: ValueLike, b: ValueLike) -> Value:
        """``1 if a == b else 0``."""
        return self._builder.eq(self._to_operand(a), self._to_operand(b))

    def is_zero(self, a: ValueLike) -> Value:
        """``1 if a == 0 else 0``."""
        return self._builder.iszero(self._to_operand(a))

    # ------------------------------------------------------------------
    # Bitwise
    # ------------------------------------------------------------------

    def bit_and(self, a: ValueLike, b: ValueLike) -> Value:
        return self._builder.and_(self._to_operand(a), self._to_operand(b))

    def bit_or(self, a: ValueLike, b: ValueLike) -> Value:
        return self._builder.or_(self._to_operand(a), self._to_operand(b))

    def bit_xor(self, a: ValueLike, b: ValueLike) -> Value:
        return self._builder.xor(self._to_operand(a), self._to_operand(b))

    def bit_not(self, a: ValueLike) -> Value:
        return self._builder.not_(self._to_operand(a))

    def shl(self, value: ValueLike, shift: ValueLike) -> Value:
        """``value << shift`` (Venom signature: ``shl(bits, val)``)."""
        return self._builder.shl(self._to_operand(shift), self._to_operand(value))

    def shr(self, value: ValueLike, shift: ValueLike) -> Value:
        """``value >> shift`` (Venom signature: ``shr(bits, val)``)."""
        return self._builder.shr(self._to_operand(shift), self._to_operand(value))

    # ------------------------------------------------------------------
    # Memory slots
    # ------------------------------------------------------------------

    def alloc_slot(self) -> IRVariable:
        """Allocate a 32-byte memory slot and return a handle to its address.

        Use :meth:`store_slot` and :meth:`load_slot` to read and write it.
        """
        return self._builder.alloca(32)

    def load_slot(self, slot: IRVariable) -> Value:
        """Load the 32-byte word stored at *slot*."""
        return self._builder.mload(slot)

    def store_slot(self, slot: IRVariable, v: ValueLike) -> None:
        """Store *v* (32 bytes) into *slot*."""
        self._builder.mstore(slot, self._to_operand(v))

    # ------------------------------------------------------------------
    # Self / context
    # ------------------------------------------------------------------

    def self_addr(self) -> Value:
        """EVM ``ADDRESS`` — the running program's own address."""
        return self._builder.address()

    def gas_left(self) -> Value:
        """EVM ``GAS`` — remaining gas."""
        return self._builder.gas()

    def eth_balance(self, account: ValueLike) -> Value:
        """EVM ``BALANCE(account)`` — ETH balance of *account*."""
        return self._builder.balance(self._to_operand(account))

    def erc20_balance_of(self, token: ValueLike, account: ValueLike) -> Value:
        """ERC-20 ``token.balanceOf(account)`` via STATICCALL.

        Reverts if the staticcall fails.  Use :meth:`eth_balance` for native
        ETH balance (calls the BALANCE opcode directly, no staticcall).
        """
        # ABI: balanceOf(address) selector 0x70a08231, arg at offset 4..36.
        selector = 0x70A08231 << (28 * 8)  # left-align to byte 0..3 of a 32-byte word
        scratch = self._alloc(36)
        # mem[scratch] = selector word (bytes 0..3 hold the selector, 4..31 are zero)
        self._builder.mstore(scratch, selector)
        # mem[scratch+4] = account (right-aligned in a 32-byte word; MSTORE at
        # scratch+4 writes bytes scratch+4..scratch+36, with 12 leading zeros
        # then the 20-byte address).
        self._builder.mstore(self._builder.add(scratch, 4), self._to_operand(account))
        # STATICCALL(gas, token, scratch, 36, scratch, 32) — reuse scratch for retdata.
        success = self._builder.staticcall(
            self._builder.gas(),
            self._to_operand(token),
            scratch,
            36,
            scratch,
            32,
        )
        self._builder.assert_(success)
        return self._builder.mload(scratch)

    # ------------------------------------------------------------------
    # Internal: memory + data-section helpers (Venom alloca)
    # ------------------------------------------------------------------

    def _alloc_calldata(self, calldata: bytes) -> tuple[IRVariable, int]:
        """Append *calldata* as a Venom data section, copy into a fresh buffer.

        Returns ``(base, blen)`` where ``base`` is the SSA pointer to the
        start of the buffer in memory and ``blen`` is the unpadded byte
        length (the value to pass as ``argsLen`` to ``CALL``).
        """
        if len(calldata) > 0xFFFF:
            raise ValueError(f"calldata too large ({len(calldata)} bytes, max 65535)")
        blen = len(calldata)

        # Dedup: identical blobs share one data section.
        label = self._blob_labels.get(calldata)
        if label is None:
            label = IRLabel(f"pydefi_calldata_{self._data_section_counter}", is_symbol=True)
            self._data_section_counter += 1
            self._ctx.append_data_section(label)
            self._ctx.append_data_item(calldata)
            self._blob_labels[calldata] = label

        base = self._alloc(blen)
        self._builder.codecopy(base, label, blen)
        return base, blen

    def _alloc(self, size: int) -> IRVariable:
        """Allocate *size* bytes (rounded up to 32) and return the base pointer."""
        return self._builder.alloca((size + 31) & ~31)

    # ------------------------------------------------------------------
    # External calls
    # ------------------------------------------------------------------

    def template(self, signature: str, *, names: "list[str] | None" = None) -> CalldataTemplate:
        """Build a reusable ABI calldata template.

        The returned :class:`CalldataTemplate` encodes a zero-filled blob once
        and records per-parameter byte offsets. Invoking it produces a
        :class:`CalldataPayload` whose blob is identical across invocations,
        so :meth:`call_contract`'s blob dedup cache reuses one data section
        regardless of how many times the template is called.

        Example::

            xfer = p.template("transfer(address to, uint256 amount)")
            for recipient, amount in rows:
                p.call_contract(weth, xfer(to=recipient, amount=amount))
        """
        return CalldataTemplate(signature, names=names)

    def call_contract(
        self,
        to: ValueLike,
        calldata: "bytes | CalldataPayload",
        *,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
        patches: Mapping[int, ValueLike] | None = None,
    ) -> Value:
        """Emit a CALL with static calldata and optional runtime patches.

        Args:
            to:        Target contract address.
            calldata:  Either pre-encoded calldata bytes, or a
                       :class:`CalldataPayload` from a
                       :class:`CalldataTemplate`. In the latter case, patches
                       are auto-derived; any explicit *patches* argument is
                       merged over the template's (caller wins on offset
                       collision).
            value:     ETH value to forward (wei).
            gas:       Gas limit; ``None`` forwards all remaining gas.
            patches:   Optional ``{calldata_offset: value}`` overlay.  Each
                       entry overwrites a 32-byte word in the in-memory
                       calldata buffer at the given offset before CALL.

        Returns:
            A :class:`Value` holding the CALL success flag (1 on success,
            0 on failure).  Use :meth:`assert_` to revert on failure.
        """
        if isinstance(calldata, CalldataPayload):
            # Explicit caller *patches* win on offset collision.
            patches = {**calldata.patches, **(patches or {})}
            calldata = calldata.blob

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
        return success

    # ------------------------------------------------------------------
    # Returndata
    # ------------------------------------------------------------------

    def returndata_word(self, offset: int = 0) -> Value:
        """Read 32 bytes from the last call's returndata at *offset*."""
        if offset < 0:
            raise ValueError(f"returndata_word: offset must be non-negative, got {offset}")
        scratch = self._alloc(32)
        self._builder.returndatacopy(scratch, offset, 32)
        return self._builder.mload(scratch)

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
        self.assert_(not_lt, msg)

    def assert_le(self, a: ValueLike, b: ValueLike, msg: str = "") -> None:
        """Revert if ``a > b`` (i.e. require ``a <= b``)."""
        a_op = self._to_operand(a)
        b_op = self._to_operand(b)
        not_gt = self._builder.iszero(self._builder.gt(a_op, b_op))
        self.assert_(not_gt, msg)

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
        """Convenience: store *value* into a fresh slot and ``RETURN`` it (32 bytes)."""
        slot = self._builder.alloca(32)
        self._builder.mstore(slot, self._to_operand(value))
        self._builder.return_(slot, IRLiteral(32))

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
        prefix_length: int = 0,
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
            prefix_length: When the compiled program will be embedded at
                byte offset *prefix_length* of a larger bytecode buffer at
                runtime (e.g. the CCTP / OFT composer prepends a 66-byte
                ``PUSH32`` prologue before our program), every absolute
                label reference Venom resolved (``CODECOPY`` data-section
                offsets, ``jmp`` / ``jnz`` targets) is shifted by
                *prefix_length* so it stays correct.
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
        if prefix_length:
            bytecode = _shift_label_pushes(asm, bytecode, prefix_length)
        return bytecode

    # ------------------------------------------------------------------
    # Convenience dunders
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        return self.build()

    def __repr__(self) -> str:
        return f"Program(data_sections={self._data_section_counter})"
