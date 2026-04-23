"""DeFiVM fluent program builder.

:class:`Program` provides a method-chaining interface over the low-level
instruction builders in :mod:`pydefi.vm.program`.  It adds three higher-level
features that are awkward with the raw byte-concatenation approach:

1. **Label-based jumps** — define named positions with :meth:`label` and
   reference them in :meth:`jump` / :meth:`jumpi` without computing byte
   offsets by hand.  Labels are resolved when :meth:`build` is called.

2. **``call_contract`` helper** — wraps the four-item stack protocol required
   by the ``CALL`` opcode into a single method call.

3. **Program composition** — combine independent sub-programs with
   :meth:`extend` / ``+`` / ``+=`` or :meth:`compose`.

4. **Calldata surgery** — :meth:`call_with_patches` embeds runtime values
   (static, from returndata, or from a register) into a calldata template
   before dispatching the external call.

Basic usage::

    from pydefi.vm import Program
    from eth_contract.erc20 import ERC20

    ROUTER  = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    TOKEN   = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    AMOUNT  = 10 ** 18

    bytecode = (
        Program()
        # approve router to spend tokens
        .call_contract(TOKEN, ERC20.fns.approve(ROUTER, AMOUNT).data)
        .pop()  # consume CALL success flag
        # swap (pre-built calldata)
        .call_contract(ROUTER, swap_calldata, value=0, gas=0)
        .pop()  # consume CALL success flag
        # check minimum output
        .push_addr(RECIPIENT)
        .push_addr(TOKEN)
        .push_u256(MIN_OUT)
        .assert_ge("slippage: amount_out too low")
        .build()
    )

Label-based conditional example::

    bytecode = (
        Program()
        .push_u256(condition_value)
        .jumpi("skip")          # jump if condition != 0
        .push_bytes(calldata_a)
        .push_u256(0).push_addr(CONTRACT_A).push_u256(0)
        .call()
        .pop()
        .label("skip")
        .build()
    )

Composition example::

    from eth_contract.erc20 import ERC20

    approve = Program().call_contract(TOKEN, ERC20.fns.approve(ROUTER, MAX_U256).data).pop()
    swap    = Program().call_contract(ROUTER, swap_calldata).pop()

    full = approve + swap            # returns a new Program
    # or: approve.extend(swap)       # in-place
    # or: Program.compose([approve, swap])

Calldata surgery — push the patch value onto the stack first, then call::

    from pydefi.vm.program import ret_u256

    # double_sel(5) → 10; patch that into double_sel(0) template → double_sel(10) → 20
    bytecode = (
        Program()
        .call_contract(ADAPTER, double_calldata)
        .pop()
        .ret_u256(0)                         # push last retdata[0:32] (the amount) → TOS
        .call_with_patches(
            ADAPTER,
            template_calldata,               # double(0) placeholder template
            patches=[(4, 32)],               # offset 4, size 32 — value from TOS
        )
        .pop()
        .build()
    )

Calldata surgery with a register source::

    # Amount was saved to reg 0 earlier in the program
    bytecode = (
        Program()
        .push_u256(0)                         # push register 0 value → TOS
        .call_with_patches(
            ROUTER,
            swap_template,
            patches=[(36, 32)],              # offset 36, size 32 — value from TOS
        )
        .pop()
        .build()
    )

Split-swap example — swap token0 → token1, then split the output and route to
two separate destinations using arithmetic and composition::

    from pydefi.vm.program import load_reg

    # Prerequisite: swap01_template produces token1 from token0 (amount in reg 1 from
    # the CCTPComposer / OFTComposer prologue, or a prior STORE_REG).
    #
    # Program structure:
    #   1. swap token0→token1, store amount1 in reg 0
    #   2. share0 = amount1 * NUMERATOR / DENOMINATOR  (60% example)
    #   3. share1 = amount1 - share0
    #   4. swap token1 → token2 using share0
    #   5. swap token1 → token3 using share1

    NUMERATOR   = 60
    DENOMINATOR = 100

    # ── Step 1: swap token0 → token1 ────────────────────────────────────────
    step1 = (
        Program()
        # call swap adapter; retdata[0] = amount1
        .call_with_patches(SWAP01, swap01_template, []).pop()
        .ret_u256(0)          # push amount1
        .store_reg(0)         # reg[0] = amount1
    )

    # ── Step 2-3: compute shares ─────────────────────────────────────────────
    split = (
        Program()
        .load_reg(0)          # [amount1]
        .push_u256(NUMERATOR) # [amount1, 60]
        .mul()                # [amount1 * 60]
        .push_u256(DENOMINATOR)
        .div()                # [share0 = amount1*60//100]
        .store_reg(1)         # reg[1] = share0
        .load_reg(0)          # [amount1]
        .load_reg(1)          # [amount1, share0]
        .sub()                # [share1 = amount1 - share0]
        .store_reg(2)         # reg[2] = share1
    )

    # ── Step 4: swap token1 → token2 (share0 from reg 1) ────────────────────
    step4 = (
        Program()
        .load_reg(1)          # push share0 → TOS (consumed by patch)
        .call_with_patches(
            SWAP12, swap12_template,
            patches=[(AMOUNT_OFFSET, 32)],
        )
        .pop()
    )

    # ── Step 5: swap token1 → token3 (share1 from reg 2) ────────────────────
    step5 = (
        Program()
        .load_reg(2)          # push share1 → TOS (consumed by patch)
        .call_with_patches(
            SWAP13, swap13_template,
            patches=[(AMOUNT_OFFSET, 32)],
        )
        .pop()
    )

    bytecode = Program.compose([step1, split, step4, step5]).build()
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from eth_abi.abi import encode_with_hooks

if TYPE_CHECKING:
    from eth_abi.hooks import EncodingContext

from eth_contract.contract import ContractFunction
from hexbytes import HexBytes
from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
from vyper.evm.assembler.instructions import (
    CONST,
    DATA_ITEM,
    PUSH_OFST,
    PUSHLABEL,
    DataHeader,
    Label,
)
from vyper.evm.assembler.symbols import SYMBOL_SIZE
from vyper.evm.opcodes import OPCODES
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.basicblock import IRLabel
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

from pydefi.types import Address
from pydefi.vm.abi import emit_abi_encode, emit_abi_encode_packed
from pydefi.vm.program import (
    OP_JUMPDEST,
    REQUIRE_SUCCESS_BLOCK,
    add,
    assert_ge,
    assert_le,
    balance_of,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    call,
    div,
    dup,
    dup_n,
    eq,
    gas_opcode,
    gt,
    iszero,
    jump,
    jumpi,
    load_reg,
    lt,
    mod,
    mul,
    patch_value,
    pop,
    push_addr,
    push_u256,
    ret_slice,
    ret_u256,
    revert_if,
    self_addr,
    shl,
    shr,
    store_reg,
    sub,
    swap,
)

# ---------------------------------------------------------------------------
# Patch source types
# ---------------------------------------------------------------------------

#: A single patch descriptor: ``(calldata_offset, size)`` where the patch
#: value is consumed from the runtime stack (must be pushed by the caller
#: before :meth:`~Program.call_with_patches` is called):
#:
#: - *calldata_offset*: byte offset inside the calldata template to overwrite.
#: - *size*: number of bytes to overwrite (0, 32].
PatchSpec = tuple[int, int]


class Patch:
    """Marks a runtime-patched argument for :meth:`Program.call_contract_abi`,
    the value is fetched from stack, see :meth:`Program.call_with_patches`.

    ``call_contract_abi`` glues :func:`eth_abi.encode_with_hooks` and
    :meth:`~Program.call_with_patches`, see their docs for details.

    Args:
        placeholder: Value to use as a placeholder for ABI encoding, default to 0,
            which works for numeric types.  For non-numeric types like ``address``
            or ``bool``, you may need to provide a different placeholder value that
            successfully encodes.

    Example::

        from pydefi.vm import Program, Patch

        # Patch uint256 amountIn from register 1
        bytecode = (
            Program()
            .push_u256(1)
            .push_u256(2)
            .call_contract_abi(
                ROUTER,
                "function swap(uint256 amountIn, uint256 minOut)",
                Patch(),
                Patch(),
            )
            .pop()
            .build()
        )
    """

    def __init__(self, placeholder: object = 0) -> None:
        self.placeholder = placeholder
        self.offset: int | None = None  # set by __call__ during encode_with_hooks
        self.size: int | None = None  # set by __call__ during encode_with_hooks

    def __call__(self, ctx: EncodingContext) -> object:
        """Hook called by ``eth_abi.encode_with_hooks`` with the encoding context.

        Stores the absolute calldata offset (selector + encoded-args offset) and
        the size of the encoded value, then returns the placeholder value for the
        ABI encoder.
        """
        self.offset = 4 + ctx.offset  # add 4 bytes for the function selector prefix
        self.size = ctx.size
        return self.placeholder


# ---------------------------------------------------------------------------
# Patch-offset helpers — used by call_contract_abi to locate patch positions
# ---------------------------------------------------------------------------


def _collect_patches(arg: object, patches: list["Patch"]) -> bool:
    """Recursively collect :class:`Patch` objects from an argument tree.

    Traverses the value structure directly — no type-string parsing needed.
    Supports :class:`Patch` leaves at any depth inside nested :class:`tuple`
    and :class:`list` containers.

    Returns:
        ``True`` if any :class:`Patch` was found, ``False`` otherwise.
    """
    if isinstance(arg, Patch):
        patches.append(arg)
        return True

    found = False
    if isinstance(arg, (tuple, list)):
        for item in arg:
            found |= _collect_patches(item, patches)
    return found


# ---------------------------------------------------------------------------
# Built bytecode wrapper — refuses concat for programs with data sections
# ---------------------------------------------------------------------------


class _BuiltBytecode(bytes):
    """A :class:`bytes` subclass returned by :meth:`Program.build` when the
    program contains data sections.

    Refuses concatenation with ``+`` so the data-section offsets baked into
    ``CODECOPY`` cannot be silently invalidated by:

    - Prepending bytes (shifts the data sections to a higher absolute offset
      while ``CODECOPY`` still targets the original offset).
    - Appending another built program with its own data sections (shifts the
      second program's data sections so its ``CODECOPY`` targets the wrong
      bytes).
    - Appending raw bytes after the data sections (offsets remain valid but
      the appended code is unreachable — execution halts at the ``STOP``
      sentinel inserted by :meth:`Program.build` before the data sections).

    All three cases are real footguns; raise rather than choose silently.
    Programs without data sections are returned as plain :class:`bytes` and
    behave normally.

    To compose programs that contain data sections, use :class:`Program`-level
    composition (:meth:`Program.extend`, ``prog_a + prog_b``,
    :meth:`Program.compose`) and call :meth:`Program.build` once on the
    merged program.
    """

    __slots__ = ()

    _CONCAT_ERROR = (
        "Cannot concatenate a built Program containing data sections with `+`. "
        "Splicing would invalidate the CODECOPY data-section offsets baked in by build(). "
        "Compose at the Program level instead — Program.extend, prog_a + prog_b, or "
        "Program.compose — and call .build() once on the merged Program."
    )

    def __add__(self, _other: object) -> bytes:  # type: ignore[override]
        raise TypeError(self._CONCAT_ERROR)

    def __radd__(self, _other: object) -> bytes:
        raise TypeError(self._CONCAT_ERROR)


# ---------------------------------------------------------------------------
# Program builder
# ---------------------------------------------------------------------------


class Program:
    """Fluent DeFiVM bytecode builder with label support.

    All instruction methods return ``self`` so calls can be chained.
    Call :meth:`build` at the end to obtain the final ``bytes`` bytecode.
    """

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str]] = []  # (u16 offset in _buf, label name)
        self._data_sections: list[bytes] = []  # raw data appended after code at build time
        # (offset in _buf, data_section_index, fixup_size_bytes) — fixup_size is
        # 4 for hand-rolled push_bytes (PUSH4 placeholder) and SYMBOL_SIZE for
        # Venom-emitted PUSH_OFST placeholders.
        self._data_fixups: list[tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, data: bytes) -> "Program":
        self._buf.extend(data)
        return self

    @classmethod
    def _snapshot(cls, source: "Program") -> "Program":
        """Return a detached copy of *source* that shares no mutable state."""
        snap = cls()
        snap._buf.extend(source._buf)
        snap._labels.update(source._labels)
        snap._fixups.extend(source._fixups)
        snap._data_sections.extend(source._data_sections)
        snap._data_fixups.extend(source._data_fixups)
        return snap

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def label(self, name: str) -> "Program":
        """Mark the current program position with *name* and emit a JUMPDEST.

        Use the same name as the target in :meth:`jump` or :meth:`jumpi`
        to create a labelled branch without computing byte offsets.

        Raises :exc:`ValueError` if the label has already been defined.
        """
        if name in self._labels:
            raise ValueError(f"Program: duplicate label {name!r}")
        self._labels[name] = len(self._buf)
        self._buf.append(OP_JUMPDEST)  # JUMPDEST
        return self

    # ------------------------------------------------------------------
    # Stack / register instructions
    # ------------------------------------------------------------------

    def push_u256(self, n: int) -> "Program":
        """Emit PUSH_U256."""
        if n < 0:
            raise ValueError(f"push_u256: value must be non-negative, got {n}")
        return self._emit(push_u256(n))

    def push_addr(self, a: Address) -> "Program":
        """Emit PUSH_ADDR."""
        if len(a) != 20:
            raise ValueError(f"push_addr: bad address length: {a!r}")
        return self._emit(push_addr(a))

    def push_bytes(self, data: bytes) -> "Program":
        """Copy *data* into free memory via CODECOPY and leave ``[argsOffset(TOS), argsLen(2nd)]``.

        Unlike the standalone :func:`~pydefi.vm.program.push_bytes` which embeds
        data inline as ``PUSH32``/``MSTORE`` chains, this version appends *data*
        as a data section after the code and emits a single ``CODECOPY`` sequence.
        The code overhead is constant (~30 bytes) regardless of data size, making
        it O(1) in bytecode growth for the code section.

        Implementation: the CODECOPY sequence is compiled by Venom — see
        :func:`compile_venom_push_bytes_fragment`.  The Venom fragment uses an
        :class:`IRLabel` placeholder for the data-section source operand, which
        :meth:`build` patches with the actual absolute offset.

        The *data* in the appended section is zero-padded to a 32-byte boundary
        so that the copied region is fully initialised.
        """
        if len(data) > 0xFFFF:
            raise ValueError(f"push_bytes: data too large ({len(data)} bytes, max 65535)")
        blen = len(data)
        blen_padded = (blen + 31) & ~31

        # Register data section (padded to 32-byte boundary).
        ds_index = len(self._data_sections)
        self._data_sections.append(data.ljust(blen_padded, b"\x00") if blen_padded > blen else data)

        # Compile via Venom and splice the fragment in; record a fixup so build()
        # patches the data-section offset into the placeholder.
        code_frag, fixup_pos = compile_venom_push_bytes_fragment(blen)
        base = len(self._buf)
        self._buf.extend(code_frag)
        self._data_fixups.append((base + fixup_pos, ds_index, SYMBOL_SIZE))
        return self

    def dup(self) -> "Program":
        """Emit DUP1 — duplicate the top stack item."""
        return self._emit(dup())

    def dup_n(self, n: int) -> "Program":
        """Emit DUPn — duplicate the stack item *n* positions from the top.

        Args:
            n: Stack depth (1 = TOS, …, 16 = sixteenth item).
        """
        return self._emit(dup_n(n))

    def swap(self) -> "Program":
        """Emit SWAP."""
        return self._emit(swap())

    def pop(self) -> "Program":
        """Emit POP."""
        return self._emit(pop())

    def load_reg(self, i: int) -> "Program":
        """Emit LOAD_REG *i*."""
        return self._emit(load_reg(i))

    def store_reg(self, i: int) -> "Program":
        """Emit STORE_REG *i*."""
        return self._emit(store_reg(i))

    # ------------------------------------------------------------------
    # Control flow instructions
    # ------------------------------------------------------------------

    def jump(self, target: str | int) -> "Program":
        """Emit JUMP.

        *target* may be either a raw byte offset (``int``) or a label name
        (``str``).  Label references are resolved at :meth:`build` time.
        """
        if isinstance(target, int):
            return self._emit(jump(target))
        self._buf.append(0x61)  # PUSH2
        self._fixups.append((len(self._buf), target))
        self._buf.extend(b"\x00\x00")  # 2-byte placeholder
        self._buf.append(0x56)  # JUMP
        return self

    def jumpi(self, target: str | int) -> "Program":
        """Emit JUMPI.

        *target* may be a raw byte offset (``int``) or a label name (``str``).
        JUMPI pops the condition from the top of the stack and jumps if it is
        non-zero.
        """
        if isinstance(target, int):
            return self._emit(jumpi(target))
        self._buf.append(0x61)  # PUSH2
        self._fixups.append((len(self._buf), target))
        self._buf.extend(b"\x00\x00")  # 2-byte placeholder
        self._buf.append(0x57)  # JUMPI
        return self

    def revert_if(self, msg: str) -> "Program":
        """Emit REVERT_IF with message *msg*."""
        return self._emit(revert_if(msg))

    def assert_ge(self, msg: str = "") -> "Program":
        """Emit ASSERT_GE — revert if top-of-stack ``a < b``."""
        return self._emit(assert_ge(msg))

    def assert_le(self, msg: str = "") -> "Program":
        """Emit ASSERT_LE — revert if top-of-stack ``a > b``."""
        return self._emit(assert_le(msg))

    # ------------------------------------------------------------------
    # External / introspection instructions
    # ------------------------------------------------------------------

    def call(self, require_success: bool = True) -> "Program":
        """Emit CALL.

        The caller must have pushed (top to bottom):
        ``gasLimit``, ``to``, ``value``, ``calldataBufIdx``.

        After execution, CALL pushes a single success flag (``1`` on success,
        ``0`` on failure) onto the stack.  Callers that do not rely on an
        automatic revert (for example, when ``require_success=False``) must
        explicitly :meth:`pop` or otherwise consume this flag to avoid stack
        mismanagement.
        """
        return self._emit(call(require_success))

    def balance_of(self) -> "Program":
        """Emit BALANCE_OF — pop ``token``, ``account``; push ERC-20 balance."""
        return self._emit(balance_of())

    def self_addr(self) -> "Program":
        """Emit SELF_ADDR — push the VM contract's own address."""
        return self._emit(self_addr())

    def sub(self) -> "Program":
        """Emit SUB — pop ``a`` (top), ``b``; push ``a - b`` (saturates to 0)."""
        return self._emit(sub())

    def add(self) -> "Program":
        """Emit ADD — pop ``a`` (top), ``b``; push ``a + b`` (wrapping uint256)."""
        return self._emit(add())

    def mul(self) -> "Program":
        """Emit MUL — pop ``a`` (top), ``b``; push ``a * b`` (wrapping uint256)."""
        return self._emit(mul())

    def div(self) -> "Program":
        """Emit DIV — pop ``a`` (top), ``b``; push ``a / b`` (0 if ``b == 0``)."""
        return self._emit(div())

    def mod(self) -> "Program":
        """Emit MOD — pop ``a`` (top), ``b``; push ``a % b`` (0 if ``b == 0``)."""
        return self._emit(mod())

    def lt(self) -> "Program":
        """Emit LT — pop ``a`` (top), ``b``; push ``1`` if ``a < b`` else ``0``."""
        return self._emit(lt())

    def gt(self) -> "Program":
        """Emit GT — pop ``a`` (top), ``b``; push ``1`` if ``a > b`` else ``0``."""
        return self._emit(gt())

    def eq(self) -> "Program":
        """Emit EQ — pop ``a`` (top), ``b``; push ``1`` if ``a == b`` else ``0``."""
        return self._emit(eq())

    def iszero(self) -> "Program":
        """Emit ISZERO — pop ``a``; push ``1`` if ``a == 0`` else ``0``."""
        return self._emit(iszero())

    def bitwise_and(self) -> "Program":
        """Emit AND — pop ``a`` (top), ``b``; push ``a & b``."""
        return self._emit(bitwise_and())

    def bitwise_or(self) -> "Program":
        """Emit OR — pop ``a`` (top), ``b``; push ``a | b``."""
        return self._emit(bitwise_or())

    def bitwise_xor(self) -> "Program":
        """Emit XOR — pop ``a`` (top), ``b``; push ``a ^ b``."""
        return self._emit(bitwise_xor())

    def bitwise_not(self) -> "Program":
        """Emit NOT — pop ``a``; push ``~a`` (bitwise complement)."""
        return self._emit(bitwise_not())

    def shl(self) -> "Program":
        """Emit SHL — pop ``shift`` (top), ``value``; push ``value << shift``."""
        return self._emit(shl())

    def shr(self) -> "Program":
        """Emit SHR — pop ``shift`` (top), ``value``; push ``value >> shift``."""
        return self._emit(shr())

    # ------------------------------------------------------------------
    # ABI / data instructions
    # ------------------------------------------------------------------

    def patch_u256(self, offset: int) -> "Program":
        """Emit PATCH_U256 at *offset*."""
        return self._emit(patch_value(offset, 32))

    def patch_addr(self, offset: int) -> "Program":
        """Emit PATCH_ADDR at *offset*."""
        return self._emit(patch_value(offset, 20))

    def patch_bytes_from_stack(self, patches: list[PatchSpec]) -> "Program":
        """Apply calldata patches consuming values from the stack.

        This is the **stand-alone patching** helper used internally by
        :meth:`call_with_patches`.  Use it when you want to patch a calldata
        buffer already on the stack and then issue the external call separately
        (or conditionally).

        The caller must have pushed the calldata buffer with
        :meth:`push_bytes` (or equivalent) so that the stack is::

            [argsOffset(TOS), argsLen(2nd), patch1_val(3rd), …, patchN_val(2+N)]

        Each ``(offset, size)`` entry in *patches* consumes the next value
        from below ``argsLen`` and writes it into the calldata buffer.

        Stack before: ``[argsOffset(TOS), argsLen(2nd), val1(3rd), …, valN]``
        Stack after:  ``[argsOffset(TOS), argsLen(2nd)]``

        Per-patch bytecode emitted::

            SWAP1   ; [argsLen, argsOffset, val_i, …]
            SWAP2   ; [val_i, argsOffset, argsLen, …]
            <patch_value(offset, size)>  ; [argsOffset, argsLen, …]

        Args:
            patches: List of ``(offset, size)`` descriptors.  The first entry
                consumes the value currently at stack position 3 (just below
                ``argsLen``), the second consumes position 3 again (after the
                previous arg was consumed), and so on.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If any patch *size* is not in the range ``(0, 32]``.
        """
        for offset, size in patches:
            if not (0 < size <= 32):
                raise ValueError(f"patch_bytes_from_stack: patch size {size!r} not supported; expected 0 < size <= 32")
            # SWAP1+SWAP2 rotates the next arg to TOS with argsOffset at 2nd —
            # exactly the layout patch_value expects — consuming the arg directly.
            self._emit(bytes([0x90]))  # SWAP1: [argsLen, argsOffset, val_i, …]
            self._emit(bytes([0x91]))  # SWAP2: [val_i, argsOffset, argsLen, …]
            self._emit(patch_value(offset, size))  # [argsOffset, argsLen, …]
        return self

    def ret_u256(self, offset: int) -> "Program":
        """Emit RET_U256 — push uint256 from last returndata at *offset*."""
        return self._emit(ret_u256(offset))

    def ret_slice(self, offset: int, length: int) -> "Program":
        """Emit RET_SLICE — push bytes slice from last returndata."""
        return self._emit(ret_slice(offset, length))

    # ------------------------------------------------------------------
    # In-VM ABI encoding
    # ------------------------------------------------------------------

    def abi_encode(
        self,
        types: list[str],
        *,
        selector: bytes | None = None,
    ) -> "Program":
        """ABI-encode N runtime values from the stack into memory.

        Generates EVM opcodes that, when executed inside the VM, perform
        canonical ABI encoding of values currently on the stack.  This is the
        in-VM equivalent of Solidity's ``abi.encode()`` — types are known at
        compile time, values come from the stack at runtime.

        The caller must have pushed N values onto the stack in type-list
        order: the first type's value pushed first (deepest on stack),
        the last type's value pushed last (TOS).  Static tuples and
        fixed-size arrays are flattened, so each leaf scalar consumes one
        stack slot.

        After execution the stack contains
        ``[argsOffset(TOS), argsLen(2nd)]``, compatible with
        :meth:`call`, :meth:`patch_u256`, and the ``CALL`` opcode.

        Example — encode two runtime values and call a contract::

            bytecode = (
                Program()
                .load_reg(0)                        # arg 0: address (deepest)
                .load_reg(1)                        # arg 1: uint256 (TOS)
                .abi_encode(
                    ["address", "uint256"],
                    selector=bytes.fromhex("a9059cbb"),  # transfer(address,uint256)
                )
                .push_u256(0)                       # value
                .push_addr(TARGET)
                .gas_opcode()
                .call()
                .pop()
                .build()
            )

        Args:
            types:    ABI type strings (static scalars, tuples, fixed arrays).
            selector: Optional 4-byte function selector prefix.

        Returns:
            ``self`` for chaining.
        """
        return self._emit(emit_abi_encode(types, selector=selector))

    def abi_encode_packed(self, types: list[str]) -> "Program":
        """Packed-encode N runtime values from the stack into memory.

        Generates EVM opcodes that perform packed ABI encoding (Solidity's
        ``abi.encodePacked()``) of values on the stack.

        Same stack convention as :meth:`abi_encode`: values pushed in
        type-list order, TOS = last type's value.  After execution the
        stack contains ``[argsOffset(TOS), argsLen(2nd)]``.

        Args:
            types: Static scalar ABI type strings.  Max 15 leaf values.

        Returns:
            ``self`` for chaining.
        """
        return self._emit(emit_abi_encode_packed(types))

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def call_contract(
        self,
        to: Address,
        calldata: bytes,
        *,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit a complete external-call sequence for a pre-built calldata buffer.

        Pushes the seven items required by the EVM ``CALL`` opcode in the correct
        stack order::

            push_u256(0)           # retSize  (deepest)
            push_u256(0)           # retOffset
            push_bytes(calldata)   # argsOffset (TOS after push_bytes), argsLen (below)
            push_u256(value)
            push_addr(to)
            push_u256(gas) or gas_opcode()  # gasLimit (top); gas_opcode() when gas==0
            CALL

        Args:
            to: Target contract address as :class:`~hexbytes.HexBytes` (``Address``).
            calldata: Pre-encoded ABI calldata.
            value: ETH value to forward with the call (wei), default 0.
            gas: Gas limit for the sub-call (0 = forward all remaining gas).
            require_success: If ``True`` (default), revert if the sub-call fails.

        Returns:
            ``self`` for chaining.
        """
        code_frag, fixup_pos = compile_venom_call_contract_fragment(
            to,
            calldata,
            value=value,
            gas=gas,
        )
        # Fragment ends at CALL — success flag is already on the EVM stack.
        blen_padded = (len(calldata) + 31) & ~31
        data = calldata.ljust(blen_padded, b"\x00") if blen_padded > len(calldata) else bytes(calldata)
        ds_index = len(self._data_sections)
        self._data_sections.append(data)
        base = len(self._buf)
        self._buf.extend(code_frag)
        # SYMBOL_SIZE-byte fixup: Venom emits PUSH<SYMBOL_SIZE> for label-resolved offsets.
        self._data_fixups.append((base + fixup_pos, ds_index, SYMBOL_SIZE))
        if require_success:
            # Append the same PC-relative revert block used by program.call(require_success=True).
            # PC-relative so the check stays correct after Program.compose() shifts code offsets.
            self._buf.extend(REQUIRE_SUCCESS_BLOCK)
        return self

    def call_contract_abi(
        self,
        to: Address,
        abi_sig: str,
        *args: object,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit an external call encoded from a human-readable ABI signature and args.

        This is a higher-level companion to :meth:`call_contract` that builds the
        calldata automatically from a **human-readable ABI function signature** and
        the Python argument values, using
        :class:`eth_contract.contract.ContractFunction` internally.

        The ``function`` keyword in *abi_sig* is optional — both bare
        ``"transfer(address,uint256)"`` and fully qualified
        ``"function transfer(address to, uint256 amount) external"`` forms are
        accepted.  Parameter names are also optional.

        All Solidity primitive types as well as nested tuples and arrays are
        supported (anything that :func:`eth_abi.encode` can handle).

        When one or more positional arguments are :class:`Patch` instances the
        method automatically:

        1. Collects all :class:`Patch` objects from the argument tree (including
           those nested inside ``tuple`` or ``list`` arguments at any depth).
        2. Encodes the full ABI calldata using
           :func:`eth_abi.encode_with_hooks`, which calls each :class:`Patch`
           hook with an :class:`~eth_abi.hooks.EncodingContext` carrying the
           exact byte offset and size of that value in the encoded output.
        3. Delegates to :meth:`call_with_patches` with the discovered offsets
           and sizes.

        :class:`Patch` may appear as a leaf element at any nesting depth —
        directly as a function argument, inside a ``tuple`` (struct) argument, or
        inside a ``list`` (array) argument, including arrays of tuples and
        multi-dimensional arrays.  The underlying ABI encoder determines whether
        a given type is patchable (numeric types always work; types like
        ``address`` and ``bool`` will raise an encoding error since they do not
        accept ``0`` as a placeholder).

        Args:
            to: Target contract address as :class:`~hexbytes.HexBytes` (``Address``).
            abi_sig: Human-readable function signature, e.g.
                ``"transfer(address,uint256)"`` or
                ``"function exactInputSingle((address,address,uint24,...) params)"``.
            *args: Positional arguments matching the signature's input parameters.
                Plain values (``int``, ``str`` address, ``tuple``, …) are encoded
                statically.  :class:`Patch` instances are used as callable hooks;
                their calldata offsets and sizes are resolved automatically.
            value: ETH value to forward with the call (wei), default 0.
            gas: Gas limit for the sub-call (0 = forward all remaining gas).
            require_success: If ``True`` (default), revert if the sub-call fails.

        Returns:
            ``self`` for chaining.

        Example (static args)::

            # ERC-20 transfer — no need to pre-build calldata
            bytecode = (
                Program()
                .call_contract_abi(TOKEN, "transfer(address,uint256)", RECIPIENT, 10**18)
                .pop()
                .build()
            )

        Example with :class:`Patch`::

            from pydefi.vm import Program, Patch

            # Patch uint256 amountIn from register 1 and uint256 minOut from register 2
            bytecode = (
                Program()
                .push_u256(2)
                .push_u256(1)
                .call_contract_abi(
                    ROUTER,
                    "function swap(uint256 amountIn, uint256 minOut)",
                    Patch(),
                    Patch(),
                )
                .pop()
                .build()
            )
        """

        normalised = abi_sig if abi_sig.lstrip().startswith("function ") else "function " + abi_sig
        fn = ContractFunction.from_abi(normalised)

        # Collect all Patch objects from the arg tree and detect whether any
        # patching is needed.
        param_types: list[str] = fn.input_types

        if len(args) != len(param_types):
            raise ValueError(
                f"call_contract_abi: expected {len(param_types)} argument(s) "
                f"for signature {abi_sig!r}, got {len(args)}."
            )

        patch_list: list[Patch] = []
        _collect_patches(args, patch_list)

        # Fast path: no Patch objects anywhere in the argument tree.
        if not patch_list:
            return self.call_contract(
                to, HexBytes(fn(*args).data), value=value, gas=gas, require_success=require_success
            )

        # Slow path: Patch objects are callable hooks; encode_with_hooks calls
        # each one with the EncodingContext so each Patch stores its offset/size.
        encoded_args = encode_with_hooks(param_types, args)
        calldata = bytes(fn.selector) + encoded_args

        patches: list[PatchSpec] = [
            (p.offset, p.size) for p in patch_list if p.offset is not None and p.size is not None
        ]
        # `patches` preserves `patch_list` order. `patch_bytes_from_stack`
        # consumes source values from TOS in that same order, so the source
        # value for `patches[0]` / the first collected Patch must already be
        # at TOS when `call_with_patches()` runs. Therefore, when multiple
        # Patch() arguments are used, callers must push their source values
        # onto the stack in reverse `patch_list` order so the first patch's
        # source ends up on top.
        return self.call_with_patches(to, calldata, patches, value=value, gas=gas, require_success=require_success)

    def call_with_patches(
        self,
        to: Address,
        calldata: bytes,
        patches: list[PatchSpec],
        *,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit a patched external call where patch values come from the stack.

        This is the **calldata surgery** helper.  Push each patch value onto the
        stack *before* calling this method (last-patch value deepest, first-patch
        value at TOS), then pass ``(offset, size)`` pairs as *patches*.

        Each entry in *patches* is a 2-tuple ``(offset, size)``:

        - *offset* — byte offset in the calldata template to overwrite.
        - *size* — number of bytes to overwrite: 32 for a ``uint256``/``int256``
          word, or 20 for an ``address``.

        Patch values are consumed from the stack top-to-bottom in *patches* list
        order.  All values are consumed before the external call is issued, so no
        post-call stack cleanup is needed.

        Example::

            from pydefi.vm.program import ret_u256

            # Push the amount value (from last returndata) first, then call.
            program = (
                Program()
                .call_contract(QUOTER, quote_calldata)
                .pop()
                .ret_u256(0)                    # push amount → TOS
                .call_with_patches(
                    ROUTER,
                    swap_template,               # swap(0, …) — placeholder at offset 36
                    patches=[(36, 32)],          # fill 32-byte slot from TOS
                )
                .pop()
                .build()
            )

        Args:
            to: Target contract address.
            calldata: Mutable calldata template bytes.
            patches: List of ``(offset, size)`` patch descriptors.  The first
                entry is matched to the current TOS, the second to the item
                below that, etc.
            value: ETH value to forward (wei), default 0.
            gas: Sub-call gas limit (0 = forward all remaining gas).
            require_success: Revert if the sub-call fails (default ``True``).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If any patch *size* is not in the range ``(0, 32]``.
        """
        if not patches:
            return self.call_contract(to, calldata, value=value, gas=gas, require_success=require_success)

        # Non-empty patches: values come from the EVM stack at runtime — manual path.
        # Push calldata buffer; patch values remain below it on the stack.
        # Stack: [argsOffset(TOS), argsLen(2nd), val1(3rd), …, valN(2+N)]
        self.push_bytes(calldata)  # CODECOPY via Program.push_bytes

        # Apply all patches, consuming each stack value directly (no DUP needed).
        self.patch_bytes_from_stack(patches)

        # Stack is now: [argsOffset(TOS), argsLen(2nd), …]
        # Insert retOffset=0 and retLen=0 *below* argsLen using a compact 7-byte
        # SWAP/PUSH1 sequence instead of two 33-byte PUSH32 instructions:
        #
        #   SWAP1          → [argsLen, argsOffset, …]
        #   PUSH1 0x00     → [0(retOffset), argsLen, argsOffset, …]
        #   SWAP1          → [argsLen, 0(retOffset), argsOffset, …]
        #   PUSH1 0x00     → [0(retLen), argsLen, 0(retOffset), argsOffset, …]
        #   SWAP3          → [argsOffset, argsLen, 0(retOffset), 0(retLen), …]
        self._emit(bytes([0x90, 0x60, 0x00, 0x90, 0x60, 0x00, 0x92]))

        # Stack: [argsOffset(TOS), argsLen, retOffset=0, retLen=0, …]
        self._emit(push_u256(value))
        self._emit(push_addr(to))
        self._emit(gas_opcode() if gas == 0 else push_u256(gas))
        self._emit(call(require_success))
        return self

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def extend(self, other: "Program") -> "Program":
        """Append *other*'s instructions to this program **in-place**.

        All byte offsets in *other*'s labels and fixup table are adjusted by
        the current length of ``self`` so that label references remain correct
        after merging.  Data sections from *other* are appended and their
        fixup indices shifted accordingly.

        Raises :exc:`ValueError` if *other* defines a label that already exists
        in ``self``.

        Returns:
            ``self`` for chaining.
        """
        # Snapshot other so extend() never mutates its argument.
        src = Program._snapshot(other)
        # Pre-validate label collisions before mutating internal state to
        # avoid leaving this Program instance in a partially-updated state.
        for name in src._labels:
            if name in self._labels:
                raise ValueError(f"Program: duplicate label {name!r} during extend")
        buf_offset = len(self._buf)
        ds_offset = len(self._data_sections)
        self._buf.extend(src._buf)
        for name, pos in src._labels.items():
            self._labels[name] = pos + buf_offset
        for fixup_off, name in src._fixups:
            self._fixups.append((fixup_off + buf_offset, name))
        self._data_sections.extend(src._data_sections)
        for fixup_pos, ds_idx, fixup_size in src._data_fixups:
            self._data_fixups.append((fixup_pos + buf_offset, ds_idx + ds_offset, fixup_size))
        return self

    def __add__(self, other: "Program") -> "Program":
        """Return a new :class:`Program` that concatenates *self* and *other*.

        Neither ``self`` nor ``other`` is modified.

        Raises :exc:`ValueError` on duplicate label names.
        """
        return Program._snapshot(self).extend(other)

    def __iadd__(self, other: "Program") -> "Program":
        """Extend this program in-place (``self += other``)."""
        return self.extend(other)

    @classmethod
    def compose(cls, programs: list["Program"]) -> "Program":
        """Compose a sequence of programs into a single :class:`Program`.

        Equivalent to reducing the list with ``+``, but more efficient for
        large numbers of sub-programs.

        Example::

            parts = [approve_prog, wrap_prog, swap_prog, unwrap_prog]
            bytecode = Program.compose(parts).build()

        Raises :exc:`ValueError` on duplicate label names across sub-programs.
        """
        result = cls()
        for prog in programs:
            result.extend(prog)
        return result

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> bytes:
        """Resolve label fixups, patch data-section offsets, and return the final bytecode.

        The returned bytes are structured as::

            <code section>  <data section 0>  <data section 1>  …

        Each data section is a raw byte slice (zero-padded to a 32-byte
        boundary) referenced by a ``CODECOPY`` instruction in the code section.
        The absolute code offset for each ``CODECOPY`` is computed here once the
        total code length is known and patched into the 4-byte placeholder
        emitted by :meth:`push_bytes`.

        When the program contains data sections, the returned object is a
        :class:`_BuiltBytecode` (a :class:`bytes` subclass) that **refuses
        concatenation with** ``+`` — splicing would shift CODECOPY source
        offsets and silently corrupt the program.  Compose at the
        :class:`Program` level instead (:meth:`extend`, ``prog_a + prog_b``,
        :meth:`compose`) before calling :meth:`build`::

            combined = (prog_a + prog_b).build()         # ✓ correct
            combined = prog_a.build() + prog_b.build()   # ✗ raises TypeError

        For programs without data sections, the result is plain :class:`bytes`
        and concatenation is unrestricted.

        Raises :exc:`ValueError` if any label referenced in a jump has not
        been defined, or if a label's target offset does not fit in 16 bits.
        """
        buf = bytearray(self._buf)
        for fixup_offset, name in self._fixups:
            if name not in self._labels:
                raise ValueError(f"Program: undefined label {name!r}")
            target = self._labels[name]
            if not 0 <= target <= 0xFFFF:
                raise ValueError(f"Program: label {name!r} target offset {target} out of range for 16-bit jump")
            struct.pack_into(">H", buf, fixup_offset, target)
        # Patch CODECOPY source offsets: each data section starts immediately
        # after the code section, with sections laid out consecutively.
        # A STOP (0x00) byte is inserted before data so execution never falls
        # into raw calldata bytes if the program has no explicit terminator.
        if not self._data_fixups:
            return bytes(buf)
        buf.append(0x00)  # STOP
        code_end = len(buf)
        section_starts: list[int] = []
        pos = code_end
        for ds in self._data_sections:
            section_starts.append(pos)
            pos += len(ds)
        for fixup_pos, ds_idx, fixup_size in self._data_fixups:
            target = section_starts[ds_idx]
            if target >> (fixup_size * 8) != 0:
                raise ValueError(
                    f"Program: data-section offset {target} does not fit in {fixup_size}-byte fixup at {fixup_pos}"
                )
            buf[fixup_pos : fixup_pos + fixup_size] = target.to_bytes(fixup_size, "big")
        for ds in self._data_sections:
            buf.extend(ds)
        # Wrap in _BuiltBytecode so any `+` concatenation raises and forces the
        # user to compose at the Program level (which correctly recomputes the
        # baked-in CODECOPY data-section offsets).
        return _BuiltBytecode(bytes(buf))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        """Allow ``bytes(program)`` as an alias for ``program.build()``."""
        return self.build()

    def __len__(self) -> int:
        """Return the current (unresolved) byte length of the program."""
        return len(self._buf)

    def __repr__(self) -> str:
        return f"Program(len={len(self._buf)}, labels={list(self._labels)!r})"


# ---------------------------------------------------------------------------
# Venom compilation helpers (private)
# ---------------------------------------------------------------------------


#: Name used for the dummy data-section IRLabel that holds the CODECOPY source
#: placeholder.  Each compile uses a fresh IRLabel object so multiple fragments
#: in the same process never alias.
_FRAGMENT_PLACEHOLDER_LABEL_NAME: str = "pydefi_call_data_placeholder"


def _asm_item_size(item: object) -> int:
    """Return the byte size that *item* contributes when assembled to EVM bytecode.

    Mirrors :func:`vyper.evm.assembler.core._assembly_to_evm` for the asm item
    types Venom emits in our IR.  Used by :func:`_locate_push_ofst` to compute
    byte positions without re-assembling.
    """
    if item == "DEBUG" or isinstance(item, (CONST, DataHeader)):
        return 0
    if isinstance(item, PUSHLABEL):
        return 1 + SYMBOL_SIZE
    if isinstance(item, Label):
        return 1  # JUMPDEST
    if isinstance(item, PUSH_OFST):
        if isinstance(item.label, Label):
            return 1 + SYMBOL_SIZE
        # CONSTREF PUSH_OFST has variable length (depends on the const value);
        # we never emit it, but raise loudly if Venom ever does.
        raise NotImplementedError(f"_asm_item_size: CONSTREF PUSH_OFST not supported: {item!r}")
    if isinstance(item, int):
        return 1
    if isinstance(item, str):
        up = item.upper()
        if up.startswith("PUSH") and up != "PUSH":
            return 1  # opcode byte; immediates are separate int items in the asm list
        if up.startswith("DUP") or up.startswith("SWAP"):
            return 1
        if up in OPCODES:
            return 1
        raise ValueError(f"_asm_item_size: unrecognised asm string item: {item!r}")
    if isinstance(item, DATA_ITEM):
        if isinstance(item.data, bytes):
            return len(item.data)
        if isinstance(item.data, Label):
            return SYMBOL_SIZE
        raise ValueError(f"_asm_item_size: unrecognised DATA_ITEM payload: {item.data!r}")
    raise ValueError(f"_asm_item_size: unknown asm item type: {type(item).__name__}")


def _locate_push_ofst(asm: list, target_label: IRLabel) -> int:
    """Return the byte offset of the ``PUSH_OFST(target_label, 0)`` immediate.

    Scans *asm* once, tracking the byte position that each item will occupy in
    the final bytecode.  Returns the position of the immediate (i.e. the byte
    just after the ``PUSH<SYMBOL_SIZE>`` opcode), which is where
    :meth:`Program.build` will patch the resolved data-section offset.

    Matches by label *name* — Venom asm wraps :class:`IRLabel` names in the
    assembler-level :class:`vyper.evm.assembler.instructions.Label` type, so
    object identity / ``==`` between the two doesn't hold; the underlying
    string name is the stable identifier.

    Raises:
        RuntimeError: If zero or more than one matching PUSH_OFST is found.
    """
    target_name = target_label.value
    pos = 0
    found = -1
    for item in asm:
        if (
            isinstance(item, PUSH_OFST)
            and isinstance(item.label, Label)
            and item.label.label == target_name
        ):
            if found != -1:
                raise RuntimeError(
                    f"_locate_push_ofst: multiple PUSH_OFST({target_name!r}) found in asm "
                    f"(first at byte {found}, second at byte {pos + 1})"
                )
            found = pos + 1  # +1 to skip the PUSH<SYMBOL_SIZE> opcode byte
        pos += _asm_item_size(item)
    if found < 0:
        raise RuntimeError(f"_locate_push_ofst: PUSH_OFST({target_name!r}) not found in asm")
    return found


def _venom_alloc_copy_fragment(
    builder: VenomBuilder,
    blen_padded: int,
    placeholder_label: IRLabel,
) -> Any:
    """Emit fp_init + CODECOPY-from-*placeholder_label* + advance FP into *builder*.

    *placeholder_label* must already be registered as a data section in the
    enclosing IRContext so Venom can resolve ``builder.offset(0, label)``.
    """
    fp = builder.mload(0x40)  # type: ignore[arg-type]
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)
    data_src = builder.offset(0, placeholder_label)
    builder.codecopy(base_fp, data_src, blen_padded)
    builder.mstore(0x40, builder.add(base_fp, blen_padded))  # type: ignore[arg-type]
    return base_fp


_OP_CALL: int = 0xF1
_OP_REVERT: int = 0xFD


def _compile_fragment_ctx(
    ctx: IRContext,
    placeholder_label: IRLabel,
    truncate_opcode: int,
    truncate_mode: str,
    *,
    fragment_name: str,
) -> tuple[bytes, int]:
    """Compile *ctx* with Venom and extract a composable fragment.

    Runs the Venom pipeline, locates the ``PUSH_OFST(placeholder_label, 0)``
    immediate via the asm list, truncates the bytecode at *truncate_opcode*
    using *truncate_mode*, and zeros the SYMBOL_SIZE-byte placeholder so the
    caller's :class:`Program._data_fixups` can patch it at build() time.

    Shared helper for :func:`compile_venom_call_contract_fragment` and
    :func:`compile_venom_push_bytes_fragment` — they only differ in IR
    construction and the truncation anchor.

    Venom optimization is enabled (``OptimizationLevel.GAS``).  The fragment
    extraction is layout-agnostic — the placeholder is found by label name
    via the asm walk in :func:`_locate_push_ofst`, and truncation anchors on
    the unique CALL/REVERT opcode rather than fixed byte patterns — so any
    instruction reordering the optimizer performs is safe.
    """
    flags = VenomOptimizationFlags(  # type: ignore[call-arg]
        level=OptimizationLevel.GAS,  # type: ignore[attr-defined]
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)  # type: ignore[misc]
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.GAS)  # type: ignore[misc,attr-defined]
    bytecode, _ = generate_bytecode(asm)  # type: ignore[misc]

    fixup_pos = _locate_push_ofst(asm, placeholder_label)
    code_frag = _truncate_at_opcode(bytecode, truncate_opcode, mode=truncate_mode)
    if fixup_pos + SYMBOL_SIZE > len(code_frag):
        raise RuntimeError(
            f"{fragment_name}: PUSH_OFST at byte {fixup_pos} falls past the truncated code length ({len(code_frag)})"
        )
    code_frag[fixup_pos : fixup_pos + SYMBOL_SIZE] = b"\x00" * SYMBOL_SIZE
    return bytes(code_frag), fixup_pos


def _truncate_at_opcode(code_bytes: bytes, opcode: int, *, mode: str) -> bytearray:
    """Truncate *code_bytes* at an opcode boundary.

    Walks opcodes (correctly skipping ``PUSH<n>`` immediates) so an ``opcode``
    byte that appears inside a PUSH literal is not mistaken for an instruction.

    Args:
        opcode: The opcode byte to anchor on (e.g. 0xF1 for CALL, 0xFD for REVERT).
        mode: ``"after_last"`` keeps everything up to and including the last
            occurrence of *opcode*; ``"before_first"`` keeps everything up to
            but **excluding** the first occurrence.

    Raises:
        RuntimeError: If no instance of *opcode* is found.
        ValueError: If *mode* is unrecognised.
    """
    if mode not in ("after_last", "before_first"):
        raise ValueError(f"_truncate_at_opcode: bad mode {mode!r}")
    n = len(code_bytes)
    last = -1
    first = -1
    i = 0
    while i < n:
        op = code_bytes[i]
        if op == opcode:
            if first == -1:
                first = i
            last = i
            i += 1
        elif 0x60 <= op <= 0x7F:  # PUSH1..PUSH32
            i += 1 + (op - 0x5F)
        else:
            i += 1
    anchor = last if mode == "after_last" else first
    if anchor < 0:
        raise RuntimeError(f"_truncate_at_opcode: opcode 0x{opcode:02x} not found in compiled fragment")
    return bytearray(code_bytes[: anchor + 1] if mode == "after_last" else code_bytes[:anchor])


def compile_venom_call_contract_fragment(
    to: Address,
    calldata: bytes,
    *,
    value: int = 0,
    gas: int = 0,
) -> tuple[bytes, int]:
    """Compile a composable Venom call fragment for use in Program.call_contract.

    Produces a code fragment ending at ``CALL`` (success on the EVM stack) with
    the CODECOPY source operand emitted as a Venom-resolved label reference
    (``PUSH<SYMBOL_SIZE>``).  The placeholder bytes are zeroed in the returned
    fragment so :class:`Program._data_fixups` can patch the actual data-section
    offset at build() time.

    Implementation:

    1. Register a dummy data section in the IRContext under a fresh
       :class:`IRLabel`; its content is irrelevant — only its label is used as
       a resolution target by ``builder.offset(0, label)``.
    2. Emit the fp_init / CODECOPY / CALL sequence; Venom resolves the
       PUSH_OFST(label, 0) operand at compile time.
    3. After compilation, locate the resolved PUSH_OFST in the asm by walking
       the asm list (tracking byte positions exactly as
       :func:`vyper.evm.assembler.core._assembly_to_evm` does) and zero the
       SYMBOL_SIZE-byte immediate.
    4. Truncate everything after the last CALL — the
       ``mstore(0, success); return_(0, 32)`` epilogue and Venom's fallback
       revert block are discarded along with the dummy data section.

    ``require_success`` checking is intentionally omitted; Program.call_contract
    appends a PC-relative revert block when require_success=True so the
    fragment itself stays compose-friendly.

    Returns:
        (code_frag, fixup_pos): code_frag ends at CALL with success on the
        EVM stack; fixup_pos is the byte index of the SYMBOL_SIZE-byte
        placeholder for the data-section offset.
    """
    if len(to) != 20:
        raise ValueError(f"bad address length: {to!r}")
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if gas < 0:
        raise ValueError(f"gas must be non-negative, got {gas}")
    blen = len(calldata)
    blen_padded = (blen + 31) & ~31

    ctx = IRContext()  # type: ignore[operator]
    placeholder_label = IRLabel(_FRAGMENT_PLACEHOLDER_LABEL_NAME, is_symbol=True)  # type: ignore[operator]
    ctx.append_data_section(placeholder_label)
    # Dummy payload — content doesn't matter, only the label position does.
    # Use a single byte so Venom emits a small data section that gets truncated
    # off along with the post-CALL epilogue.
    ctx.append_data_item(b"\x00")

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)  # type: ignore[operator]
    base_fp = _venom_alloc_copy_fragment(builder, blen_padded, placeholder_label)
    gas_operand = builder.gas() if gas == 0 else gas
    to_int = int.from_bytes(bytes(to), "big")
    success = builder.call(gas_operand, to_int, value, base_fp, blen, 0, 0)
    # Force a basic-block terminator so Venom doesn't insert one we'd have to guess at.
    builder.mstore(0, success)  # type: ignore[arg-type]
    builder.return_(0, 32)  # type: ignore[arg-type]  # truncated by _extract_fragment

    return _compile_fragment_ctx(
        ctx,
        placeholder_label,
        _OP_CALL,
        "after_last",
        fragment_name="compile_venom_call_contract_fragment",
    )


#: Name used for the dummy data-section IRLabel that holds the push_bytes
#: CODECOPY source placeholder.  Distinct from _FRAGMENT_PLACEHOLDER_LABEL_NAME
#: so the two fragments are unambiguous if they ever co-exist in one IRContext.
_PUSH_BYTES_PLACEHOLDER_LABEL_NAME: str = "pydefi_push_bytes_placeholder"


def compile_venom_push_bytes_fragment(blen: int) -> tuple[bytes, int]:
    """Compile a composable Venom push_bytes fragment for use in Program.push_bytes.

    Produces a code fragment that, when executed, copies ``blen_padded`` bytes
    from the program's data section into free memory and leaves
    ``[argsOffset(TOS), argsLen]`` on the EVM stack — same stack effect as the
    hand-rolled version.  The CODECOPY src operand is emitted via a Venom
    ``PUSH_OFST(label, 0)``; the placeholder bytes are zeroed in the returned
    fragment so :class:`Program._data_fixups` can patch the actual data-section
    offset at build() time.

    Truncation strategy: Venom requires a basic-block terminator, so we emit
    ``revert(base_fp, blen)`` whose lowering is::

        PUSH blen          # second arg (length)
        PUSH base_fp       # first arg (offset, TOS)
        REVERT

    The fragment is then truncated **before** the first ``REVERT`` opcode,
    which discards the REVERT instruction itself but preserves the two PUSHes
    on the stack — yielding ``[base_fp(=argsOffset, TOS), blen(=argsLen)]``,
    exactly the contract the caller expects.  Venom's fallback revert block
    appears later in the output and is also discarded by the truncation.

    Args:
        blen: Logical length of the data (not zero-padded).

    Returns:
        (code_frag, fixup_pos): code_frag ends just before the REVERT opcode
        with [argsOffset, argsLen] on the EVM stack; fixup_pos is the byte
        index of the SYMBOL_SIZE-byte placeholder for the data-section offset.
    """
    if blen < 0:
        raise ValueError(f"compile_venom_push_bytes_fragment: blen must be non-negative, got {blen}")
    if blen > 0xFFFF:
        raise ValueError(f"compile_venom_push_bytes_fragment: blen too large ({blen} bytes, max 65535)")
    blen_padded = (blen + 31) & ~31

    ctx = IRContext()  # type: ignore[operator]
    placeholder_label = IRLabel(_PUSH_BYTES_PLACEHOLDER_LABEL_NAME, is_symbol=True)  # type: ignore[operator]
    ctx.append_data_section(placeholder_label)
    ctx.append_data_item(b"\x00")  # dummy; truncated off

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)  # type: ignore[operator]
    base_fp = _venom_alloc_copy_fragment(builder, blen_padded, placeholder_label)
    # revert(offset, length) lowers to PUSH length; PUSH offset; REVERT.
    # Truncating before REVERT leaves the two PUSHes' values on the stack.
    builder.revert(base_fp, blen)  # type: ignore[arg-type]

    return _compile_fragment_ctx(
        ctx,
        placeholder_label,
        _OP_REVERT,
        "before_first",
        fragment_name="compile_venom_push_bytes_fragment",
    )


