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
from typing import TYPE_CHECKING, cast

from eth_abi.abi import encode_with_hooks

if TYPE_CHECKING:
    from eth_abi.hooks import EncodingContext

from eth_contract.contract import ContractFunction
from hexbytes import HexBytes

from pydefi.types import Address
from pydefi.vm.abi import emit_abi_encode, emit_abi_encode_packed
from pydefi.vm.program import (
    _DUP3,
    _PUSH4,
    OP_CODECOPY,
    OP_JUMPDEST,
    _push_imm,
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
    fp_init,
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
    push_bytes,
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

try:
    from vyper.compiler.phases import generate_bytecode
    from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
    from vyper.venom import generate_assembly_experimental, run_passes_on
    from vyper.venom.basicblock import IRLabel
    from vyper.venom.builder import VenomBuilder
    from vyper.venom.context import IRContext

    _VENOM_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised by availability checks
    generate_bytecode = None
    OptimizationLevel = None
    VenomOptimizationFlags = None
    generate_assembly_experimental = None
    run_passes_on = None
    VenomBuilder = None
    IRContext = None
    IRLabel = None
    _VENOM_IMPORT_ERROR = str(exc)

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
        self._data_fixups: list[tuple[int, int]] = []  # (u32 offset in _buf, data_section_index)
        self._venom_enabled = venom_is_available()
        self._venom_plan: dict[str, object] | None = None
        self._venom_prefix_stack: list[tuple[str, int]] = []

    @classmethod
    def create(cls) -> "Program":
        """Create a program builder via factory semantics.

        The VM builder uses a single Program type; Venom planning is enabled
        automatically when local Vyper Venom APIs are importable.
        """
        return create_program()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, data: bytes) -> "Program":
        if self._venom_enabled and self._venom_prefix_stack:
            self._materialize_prefix_stack_to_manual()
        if self._venom_enabled and self._venom_plan is not None:
            self._materialize_plan_to_manual()
        self._buf.extend(data)
        return self

    def _is_pristine(self) -> bool:
        return (
            len(self._buf) == 0
            and not self._labels
            and not self._fixups
            and not self._data_sections
            and not self._data_fixups
        )

    def _materialize_prefix_stack_to_manual(self) -> None:
        if not self._venom_prefix_stack:
            return
        for kind, value in self._venom_prefix_stack:
            if kind == "u256":
                self._buf.extend(push_u256(value))
            elif kind == "addr":
                self._buf.extend(push_addr(HexBytes(value.to_bytes(20, "big"))))
            else:
                raise ValueError(f"unknown venom prefix stack kind: {kind!r}")
        self._venom_prefix_stack.clear()

    def _consume_prefix_patch_values(self, patch_count: int) -> list[int] | None:
        if patch_count < 0:
            return None
        if patch_count == 0:
            if self._venom_prefix_stack:
                return None
            return []
        if len(self._venom_prefix_stack) != patch_count:
            return None
        vals = [v for _k, v in self._venom_prefix_stack]
        self._venom_prefix_stack.clear()
        vals.reverse()  # TOS-first patch order
        return vals

    @property
    def has_pending_venom_plan(self) -> bool:
        """Return True if this program is pending Venom-plan compilation."""
        return self._venom_plan is not None

    @property
    def pending_venom_plan_kind(self) -> str | None:
        """Return the pending plan kind, or None when no Venom plan is active."""
        if self._venom_plan is None:
            return None
        kind = self._venom_plan.get("kind")
        return str(kind) if kind is not None else None

    def _materialize_plan_to_manual(self) -> None:
        self._materialize_prefix_stack_to_manual()
        if self._venom_plan is None:
            return
        plan = self._venom_plan
        self._venom_plan = None
        kind = plan.get("kind")
        if kind == "call_contract":
            to = cast(Address, plan["to"])
            calldata = cast(bytes, plan["calldata"])
            value = cast(int, plan["value"])
            gas = cast(int, plan["gas"])
            require_success = cast(bool, plan["require_success"])
            self._buf.extend(push_u256(0))
            self._buf.extend(push_u256(0))
            self._buf.extend(push_bytes(calldata))
            self._buf.extend(push_u256(value))
            self._buf.extend(push_addr(to))
            self._buf.extend(gas_opcode() if gas == 0 else push_u256(gas))
            self._buf.extend(call(require_success))
            if plan.get("drop_result"):
                self._buf.extend(pop())
            return
        if kind == "call_with_patches":
            to = cast(Address, plan["to"])
            calldata = cast(bytes, plan["calldata"])
            patches = cast(list[tuple[int, int]], plan["patches"])
            value = cast(int, plan["value"])
            gas = cast(int, plan["gas"])
            require_success = cast(bool, plan["require_success"])
            patch_values = plan.get("patch_values")
            if isinstance(patch_values, list):
                for pv in reversed(patch_values):
                    self._buf.extend(push_u256(int(pv)))
            self._buf.extend(push_bytes(calldata))
            for offset, size in patches:
                if not (0 < size <= 32):
                    raise ValueError(f"call_with_patches: patch size {size!r} not supported; expected 0 < size <= 32")
                self._buf.extend(bytes([0x90]))  # SWAP1
                self._buf.extend(bytes([0x91]))  # SWAP2
                self._buf.extend(patch_value(offset, size))
            self._buf.extend(bytes([0x90, 0x60, 0x00, 0x90, 0x60, 0x00, 0x92]))
            self._buf.extend(push_u256(value))
            self._buf.extend(push_addr(to))
            self._buf.extend(gas_opcode() if gas == 0 else push_u256(gas))
            self._buf.extend(call(require_success))
            if plan.get("drop_result"):
                self._buf.extend(pop())
            return
        raise ValueError(f"unknown venom plan kind: {kind!r}")

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def label(self, name: str) -> "Program":
        """Mark the current program position with *name* and emit a JUMPDEST.

        Use the same name as the target in :meth:`jump` or :meth:`jumpi`
        to create a labelled branch without computing byte offsets.

        Raises :exc:`ValueError` if the label has already been defined.
        """
        if self._venom_enabled and self._venom_prefix_stack:
            self._materialize_prefix_stack_to_manual()
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
        if self._venom_enabled and self._venom_plan is not None:
            self._materialize_plan_to_manual()
        if self._venom_enabled and self._is_pristine():
            self._venom_prefix_stack.append(("u256", n))
            return self
        return self._emit(push_u256(n))

    def push_addr(self, a: Address) -> "Program":
        """Emit PUSH_ADDR."""
        if len(a) != 20:
            raise ValueError(f"push_addr: bad address length: {a!r}")
        if self._venom_enabled and self._venom_plan is not None:
            self._materialize_plan_to_manual()
        if self._venom_enabled and self._is_pristine():
            self._venom_prefix_stack.append(("addr", int.from_bytes(bytes(a), "big")))
            return self
        return self._emit(push_addr(a))

    def push_bytes(self, data: bytes) -> "Program":
        """Copy *data* into free memory via CODECOPY and leave ``[argsOffset(TOS), argsLen(2nd)]``.

        Unlike the standalone :func:`~pydefi.vm.program.push_bytes` which embeds
        data inline as ``PUSH32``/``MSTORE`` chains, this version appends *data*
        as a data section after the code and emits a single ``CODECOPY`` sequence.
        The code overhead is ~32 bytes regardless of data size, making it O(1)
        in bytecode growth for the code section.

        CODECOPY sequence emitted (~ 32 bytes fixed overhead):

        .. code-block:: text

            fp_init()                    # → [base_fp]
            PUSH<n> blen_padded          # [blen_padded, base_fp]
            PUSH4 <code_offset>          # [code_offset, blen_padded, base_fp]  ← fixup at build()
            DUP3                         # [base_fp, code_offset, blen_padded, base_fp]
            CODECOPY                     # mem[base_fp..+blen_padded] = code[code_offset..]; [base_fp]
            DUP1                         # [base_fp, base_fp]
            PUSH<n> blen_padded          # [blen_padded, base_fp, base_fp]
            ADD                          # [new_fp, base_fp]
            PUSH1 0x40                   # [0x40, new_fp, base_fp]
            MSTORE                       # mem[0x40] = new_fp; [base_fp]
            PUSH<n> blen                 # [blen, base_fp]
            SWAP1                        # [base_fp=argsOffset, blen=argsLen]

        The *data* in the appended section is zero-padded to a 32-byte boundary
        so that the copied region is fully initialised.
        """
        if len(data) > 0xFFFF:
            raise ValueError(f"push_bytes: data too large ({len(data)} bytes, max 65535)")
        # Flush any pending Venom state so the data section + CODECOPY sequence
        # is emitted in the correct order (after any preceding pushes).
        if self._venom_enabled and (self._venom_prefix_stack or self._venom_plan is not None):
            self._materialize_plan_to_manual()
        blen = len(data)
        blen_padded = (blen + 31) & ~31

        # Register data section (padded to 32-byte boundary).
        ds_index = len(self._data_sections)
        self._data_sections.append(data.ljust(blen_padded, b"\x00") if blen_padded > blen else data)

        # Emit CODECOPY sequence.
        # Step 1: compute base_fp (free-memory pointer, defaulting to 0x280).
        self._buf.extend(fp_init())

        # Step 2: push CODECOPY arguments — size, then src offset (fixup), then dest (DUP3).
        imm_padded = _push_imm(blen_padded)
        self._buf.extend(imm_padded)  # PUSH<n> blen_padded  (size)
        self._buf.append(_PUSH4)  # PUSH4 opcode
        fixup_pos = len(self._buf)  # position of the 4-byte placeholder
        self._buf.extend(b"\x00\x00\x00\x00")  # placeholder for code_offset
        self._buf.append(_DUP3)  # DUP3 → base_fp as destOffset
        self._buf.append(OP_CODECOPY)  # CODECOPY(dest=base_fp, src=code_offset, size=blen_padded)

        self._data_fixups.append((fixup_pos, ds_index))

        # Step 3: advance free-memory pointer by blen_padded.
        self._buf.append(0x80)  # DUP1
        self._buf.extend(imm_padded)  # PUSH<n> blen_padded
        self._buf.append(0x01)  # ADD
        self._buf.extend(b"\x60\x40")  # PUSH1 0x40
        self._buf.append(0x52)  # MSTORE — mem[0x40] = new_fp; [base_fp]

        # Step 4: leave [argsOffset(TOS), argsLen(below)].
        self._buf.extend(_push_imm(blen))  # PUSH<n> blen
        self._buf.append(0x90)  # SWAP1 → [base_fp=argsOffset, blen=argsLen]

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
        if (
            self._venom_enabled
            and self._venom_plan is not None
            and self._venom_plan.get("kind")
            in {
                "call_contract",
                "call_with_patches",
            }
        ):
            self._venom_plan["drop_result"] = True
            return self
        if self._venom_enabled and self._venom_prefix_stack and self._is_pristine():
            self._venom_prefix_stack.pop()
            return self
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
        if self._venom_enabled and self._venom_prefix_stack:
            self._materialize_prefix_stack_to_manual()
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
        if self._venom_enabled and self._venom_prefix_stack:
            self._materialize_prefix_stack_to_manual()
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
        if self._venom_enabled and self._venom_plan is None and self._is_pristine() and not self._venom_prefix_stack:
            self._venom_plan = {
                "kind": "call_contract",
                "to": to,
                "calldata": bytes(calldata),
                "value": value,
                "gas": gas,
                "require_success": require_success,
                "drop_result": False,
            }
            return self

        if self._venom_enabled and self._venom_plan is not None:
            self._materialize_plan_to_manual()
        return (
            self._emit(push_u256(0))  # retSize
            ._emit(push_u256(0))  # retOffset
            ._emit(push_bytes(calldata))  # argsOffset (TOS), argsLen
            ._emit(push_u256(value))
            ._emit(push_addr(to))
            ._emit(gas_opcode() if gas == 0 else push_u256(gas))
            ._emit(call(require_success))
        )

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
        if self._venom_enabled and self._venom_plan is None and self._is_pristine():
            patch_values = self._consume_prefix_patch_values(len(patches))
            if patch_values is not None:
                for _offset, size in patches:
                    if not (0 < size <= 32):
                        raise ValueError(
                            f"call_with_patches: patch size {size!r} not supported; expected 0 < size <= 32"
                        )
                if any(v < 0 for v in patch_values):
                    raise ValueError("call_with_patches: patch values must be non-negative")
                self._venom_plan = {
                    "kind": "call_with_patches",
                    "to": to,
                    "calldata": bytes(calldata),
                    "patches": list(patches),
                    "patch_values": list(patch_values),
                    "value": value,
                    "gas": gas,
                    "require_success": require_success,
                    "drop_result": False,
                }
                return self

        if self._venom_enabled and self._venom_prefix_stack:
            self._materialize_prefix_stack_to_manual()
        if self._venom_enabled and self._venom_plan is not None:
            self._materialize_plan_to_manual()

        # Push calldata buffer; patch values remain below it on the stack.
        # Stack: [argsOffset(TOS), argsLen(2nd), val1(3rd), …, valN(2+N)]
        self._emit(push_bytes(calldata))  # PUSH32/MSTORE chain (works in interpreter calldata context)

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
        # Materialize any pending Venom state (prefix-stack pushes and/or a
        # pending call plan) to _buf before merging.  self must flush first so
        # bytes stay in order; other is flushed so its _buf is complete before copy.
        if self._venom_enabled and (self._venom_prefix_stack or self._venom_plan is not None):
            self._materialize_plan_to_manual()
        if other._venom_enabled and (other._venom_prefix_stack or other._venom_plan is not None):
            other._materialize_plan_to_manual()
        # Pre-validate label collisions before mutating internal state to
        # avoid leaving this Program instance in a partially-updated state.
        for name in other._labels:
            if name in self._labels:
                raise ValueError(f"Program: duplicate label {name!r} during extend")
        buf_offset = len(self._buf)
        ds_offset = len(self._data_sections)
        self._buf.extend(other._buf)
        for name, pos in other._labels.items():
            self._labels[name] = pos + buf_offset
        for fixup_off, name in other._fixups:
            self._fixups.append((fixup_off + buf_offset, name))
        self._data_sections.extend(other._data_sections)
        for fixup_pos, ds_idx in other._data_fixups:
            self._data_fixups.append((fixup_pos + buf_offset, ds_idx + ds_offset))
        return self

    def __add__(self, other: "Program") -> "Program":
        """Return a new :class:`Program` that concatenates *self* and *other*.

        Neither ``self`` nor ``other`` is modified.

        Raises :exc:`ValueError` on duplicate label names.
        """
        # Materialize self's pending Venom state before copying.
        # Materialization is idempotent: it moves speculative state (prefix
        # stack / plan) into _buf; the resulting bytecode is identical.
        if self._venom_enabled and (self._venom_prefix_stack or self._venom_plan is not None):
            self._materialize_plan_to_manual()
        result = Program()
        result._buf.extend(self._buf)
        result._labels.update(self._labels)
        result._fixups.extend(self._fixups)
        result._data_sections.extend(self._data_sections)
        result._data_fixups.extend(self._data_fixups)
        result.extend(other)
        return result

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

        Raises :exc:`ValueError` if any label referenced in a jump has not
        been defined, or if a label's target offset does not fit in 16 bits.
        """
        if self._venom_enabled and self._venom_prefix_stack:
            self._materialize_prefix_stack_to_manual()

        if self._venom_enabled and self._venom_plan is not None:
            plan = self._venom_plan
            if plan.get("kind") == "call_contract":
                to = cast(Address, plan["to"])
                calldata = cast(bytes, plan["calldata"])
                value = cast(int, plan["value"])
                gas = cast(int, plan["gas"])
                require_success = cast(bool, plan["require_success"])
                return compile_venom_call_contract_probe(
                    to,
                    calldata,
                    value=value,
                    gas=gas,
                    require_success=require_success,
                    return_success=not bool(plan.get("drop_result")),
                )
            if plan.get("kind") == "call_with_patches":
                to = cast(Address, plan["to"])
                calldata = cast(bytes, plan["calldata"])
                patches = cast(list[tuple[int, int]], plan["patches"])
                patch_values = cast(list[int], plan["patch_values"])
                value = cast(int, plan["value"])
                gas = cast(int, plan["gas"])
                require_success = cast(bool, plan["require_success"])
                return compile_venom_call_with_patches_probe(
                    to,
                    calldata,
                    patches=patches,
                    patch_values=patch_values,
                    value=value,
                    gas=gas,
                    require_success=require_success,
                    return_success=not bool(plan.get("drop_result")),
                )

            self._materialize_plan_to_manual()

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
        if self._data_fixups:
            code_end = len(buf)
            section_starts: list[int] = []
            pos = code_end
            for ds in self._data_sections:
                section_starts.append(pos)
                pos += len(ds)
            for fixup_pos, ds_idx in self._data_fixups:
                struct.pack_into(">I", buf, fixup_pos, section_starts[ds_idx])
            for ds in self._data_sections:
                buf.extend(ds)
        return bytes(buf)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        """Allow ``bytes(program)`` as an alias for ``program.build()``."""
        return self.build()

    def __len__(self) -> int:
        """Return the current (unresolved) byte length of the program."""
        if not self._venom_prefix_stack:
            return len(self._buf)
        # Prefix-stack entries are pending push_u256/push_addr bytes not yet in _buf.
        prefix_len = sum(33 if kind == "u256" else 21 for kind, _ in self._venom_prefix_stack)
        return len(self._buf) + prefix_len

    def __repr__(self) -> str:
        return f"Program(len={len(self._buf)}, labels={list(self._labels)!r})"


def venom_is_available() -> bool:
    """Return True when Vyper Venom APIs are importable in this environment."""
    return _VENOM_IMPORT_ERROR is None


def venom_import_error() -> str | None:
    """Return import error details when Venom APIs are unavailable."""
    return _VENOM_IMPORT_ERROR


def compile_venom_smoke_bytecode() -> bytes:
    """Compile a minimal Venom IR program to bytecode.

    This function validates local integration wiring with the Vyper Venom APIs
    using the documented pipeline:

    IRContext -> run_passes_on -> generate_assembly_experimental -> generate_bytecode.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")

    assert IRContext is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    ctx = IRContext()
    fn = ctx.create_function("main")
    ctx.entry_function = fn

    builder = VenomBuilder(ctx, fn)
    builder.return_(0, 0)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_push_bytes_probe(data: bytes) -> bytes:
    """Compile a Venom program that copies *data* from a data section into memory.

    The generated runtime mirrors ``fp_init`` + ``push_bytes`` semantics:
    - free-memory pointer defaults to ``0x280`` when ``mem[0x40] == 0``,
    - payload is copied with ``dloadbytes`` (lowered to ``CODECOPY``),
    - ``mem[0x40]`` advances by padded length.

    The program returns the first 32-byte word at the copied destination,
    which allows execution-time verification that CODECOPY placed data
    correctly in memory.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")
    if len(data) > 0xFFFF:
        raise ValueError(f"push_bytes: data too large ({len(data)} bytes, max 65535)")

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    blen = len(data)
    blen_padded = (blen + 31) & ~31
    payload = data.ljust(blen_padded, b"\x00") if blen_padded > blen else data

    ctx = IRContext()
    data_label = IRLabel("pydefi_payload", is_symbol=True)
    ctx.append_data_section(data_label)
    ctx.append_data_item(payload)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    # fp_init(): fp | (0x280 * iszero(fp))
    fp = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)

    # Copy payload from readonly data segment into memory[base_fp..].
    # Use offset(0, data_label) + codecopy instead of dloadbytes to avoid the
    # LowerDloadPass code_end bug: dloadbytes emits CODECOPY with src=code_end,
    # but code_end in the assembler equals total bytecode length (code+data),
    # not the actual data section start address.
    data_src = builder.offset(0, data_label)
    builder.codecopy(base_fp, data_src, blen_padded)

    # Advance free-memory pointer.
    builder.mstore(0x40, builder.add(base_fp, blen_padded))

    # Return first copied word for easy runtime verification.
    builder.mstore(0, builder.mload(base_fp))
    builder.return_(0, 32)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_label_jump_probe(*, take_then_branch: bool) -> bytes:
    """Compile a minimal Venom conditional branch program.

    This probe validates that labels and conditional jumps are represented via
    Venom basic blocks (no manual byte-offset fixups).
    The compiled runtime returns ``1`` from the ``then`` branch and ``2`` from
    the ``else`` branch.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")

    assert IRContext is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    ctx = IRContext()
    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    then_bb = builder.create_block("then")
    else_bb = builder.create_block("else")
    builder.append_block(then_bb)
    builder.append_block(else_bb)

    cond = builder.eq(1, 1 if take_then_branch else 0)
    builder.jnz(cond, then_bb.label, else_bb.label)

    builder.set_block(then_bb)
    builder.mstore(0, 1)
    builder.return_(0, 32)

    builder.set_block(else_bb)
    builder.mstore(0, 2)
    builder.return_(0, 32)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_two_data_sections_probe(data_a: bytes, data_b: bytes) -> bytes:
    """Compile a runtime with two readonly Venom data sections.

    This probe validates that multiple data sections can be attached to the
    context and emitted in a single compilation pipeline.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    ctx = IRContext()
    ctx.append_data_section(IRLabel("pydefi_data_a", is_symbol=True))
    ctx.append_data_item(data_a)
    ctx.append_data_section(IRLabel("pydefi_data_b", is_symbol=True))
    ctx.append_data_item(data_b)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)
    builder.return_(0, 0)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_memory_progression_probe(data_a: bytes, data_b: bytes) -> bytes:
    """Compile a runtime that performs two data-section allocations and returns mem[0x40].

    This validates free-memory-pointer progression over consecutive allocations
    with fp_init-compatible semantics:

    - first allocation starts at 0x280 when mem[0x40] is zero,
    - second allocation starts at first allocation end,
    - final mem[0x40] equals 0x280 + pad32(len(a)) + pad32(len(b)).
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    len_a_padded = (len(data_a) + 31) & ~31
    len_b_padded = (len(data_b) + 31) & ~31
    payload_a = data_a.ljust(len_a_padded, b"\x00") if len_a_padded > len(data_a) else data_a
    payload_b = data_b.ljust(len_b_padded, b"\x00") if len_b_padded > len(data_b) else data_b

    ctx = IRContext()
    mem_a_label = IRLabel("pydefi_mem_a", is_symbol=True)
    ctx.append_data_section(mem_a_label)
    ctx.append_data_item(payload_a)
    mem_b_label = IRLabel("pydefi_mem_b", is_symbol=True)
    ctx.append_data_section(mem_b_label)
    ctx.append_data_item(payload_b)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    # fp_init(): fp | (0x280 * iszero(fp))
    fp0 = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp0), 0x280)
    base_fp = builder.or_(fp0, default_fp)

    # Allocation 1: copy data_a via its label address.
    data_src_a = builder.offset(0, mem_a_label)
    builder.codecopy(base_fp, data_src_a, len_a_padded)
    fp1 = builder.add(base_fp, len_a_padded)
    builder.mstore(0x40, fp1)

    # Allocation 2: copy data_b via its own label address.
    data_src_b = builder.offset(0, mem_b_label)
    builder.codecopy(fp1, data_src_b, len_b_padded)
    fp2 = builder.add(fp1, len_b_padded)
    builder.mstore(0x40, fp2)

    # Return final mem[0x40]
    builder.mstore(0, builder.mload(0x40))
    builder.return_(0, 32)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_call_contract_probe(
    to: Address,
    calldata: bytes,
    *,
    value: int = 0,
    gas: int = 0,
    require_success: bool = True,
    return_success: bool = True,
) -> bytes:
    """Compile a Venom runtime that executes CALL with static calldata.

    The calldata is stored in a Venom readonly data section and copied into
    memory via ``dloadbytes`` before CALL. The runtime returns the CALL success
    flag as a 32-byte word.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")
    if len(to) != 20:
        raise ValueError(f"call_contract_probe: bad address length: {to!r}")
    if value < 0:
        raise ValueError(f"call_contract_probe: value must be non-negative, got {value}")
    if gas < 0:
        raise ValueError(f"call_contract_probe: gas must be non-negative, got {gas}")
    if len(calldata) > 0xFFFF:
        raise ValueError(f"call_contract_probe: calldata too large ({len(calldata)} bytes, max 65535)")

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    blen = len(calldata)
    blen_padded = (blen + 31) & ~31
    payload = calldata.ljust(blen_padded, b"\x00") if blen_padded > blen else calldata

    ctx = IRContext()
    call_payload_label = IRLabel("pydefi_call_payload", is_symbol=True)
    ctx.append_data_section(call_payload_label)
    ctx.append_data_item(payload)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    # fp_init(): fp | (0x280 * iszero(fp))
    fp = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)

    # Copy calldata template from readonly data section via offset+codecopy.
    data_src = builder.offset(0, call_payload_label)
    builder.codecopy(base_fp, data_src, blen_padded)
    builder.mstore(0x40, builder.add(base_fp, blen_padded))

    gas_operand = builder.gas() if gas == 0 else gas
    to_int = int.from_bytes(bytes(to), "big")
    success = builder.call(gas_operand, to_int, value, base_fp, blen, 0, 0)
    if require_success:
        builder.assert_(success)

    if return_success:
        builder.mstore(0, success)
        builder.return_(0, 32)
    else:
        builder.return_(0, 0)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_call_with_patch_probe(
    to: Address,
    calldata: bytes,
    *,
    patch_value: int,
    patch_offset: int,
    patch_size: int,
    value: int = 0,
    gas: int = 0,
    require_success: bool = True,
) -> bytes:
    """Compile a Venom runtime that patches calldata in memory before CALL.

    This mirrors the core patching layout used by ``call_with_patches``:
    write ``patch_value`` with ``MSTORE`` at ``argsOffset + (offset + size - 32)``
    then execute CALL and return the success flag as a 32-byte word.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")
    if len(to) != 20:
        raise ValueError(f"call_with_patch_probe: bad address length: {to!r}")
    if patch_value < 0:
        raise ValueError(f"call_with_patch_probe: patch_value must be non-negative, got {patch_value}")
    if not (0 < patch_size <= 32):
        raise ValueError(f"call_with_patch_probe: patch_size must be in (0, 32], got {patch_size}")
    mstore_off = patch_offset + patch_size - 32
    if mstore_off < 0:
        raise ValueError(
            f"call_with_patch_probe: offset {patch_offset} is too small for size {patch_size}; "
            f"MSTORE target {mstore_off} would be negative"
        )
    if value < 0:
        raise ValueError(f"call_with_patch_probe: value must be non-negative, got {value}")
    if gas < 0:
        raise ValueError(f"call_with_patch_probe: gas must be non-negative, got {gas}")
    if len(calldata) > 0xFFFF:
        raise ValueError(f"call_with_patch_probe: calldata too large ({len(calldata)} bytes, max 65535)")

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    blen = len(calldata)
    blen_padded = (blen + 31) & ~31
    payload = calldata.ljust(blen_padded, b"\x00") if blen_padded > blen else calldata

    ctx = IRContext()
    patch_payload_label = IRLabel("pydefi_patch_payload", is_symbol=True)
    ctx.append_data_section(patch_payload_label)
    ctx.append_data_item(payload)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    # fp_init(): fp | (0x280 * iszero(fp))
    fp = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)

    # Copy calldata template via offset+codecopy.
    data_src = builder.offset(0, patch_payload_label)
    builder.codecopy(base_fp, data_src, blen_padded)
    patch_ptr = builder.add(base_fp, mstore_off)
    builder.mstore(patch_ptr, patch_value)
    builder.mstore(0x40, builder.add(base_fp, blen_padded))

    gas_operand = builder.gas() if gas == 0 else gas
    to_int = int.from_bytes(bytes(to), "big")
    success = builder.call(gas_operand, to_int, value, base_fp, blen, 0, 0)
    if require_success:
        builder.assert_(success)

    builder.mstore(0, success)
    builder.return_(0, 32)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_call_with_patches_probe(
    to: Address,
    calldata: bytes,
    *,
    patches: list[tuple[int, int]],
    patch_values: list[int],
    value: int = 0,
    gas: int = 0,
    require_success: bool = True,
    return_success: bool = True,
) -> bytes:
    """Compile a Venom runtime that applies multiple calldata patches before CALL.

    Each patch/value pair is applied in order using the same MSTORE target rule
    as ``patch_value`` in the manual builder:

    ``mstore_ptr = argsOffset + (offset + size - 32)``
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")
    if len(to) != 20:
        raise ValueError(f"call_with_patches_probe: bad address length: {to!r}")
    if len(patches) != len(patch_values):
        raise ValueError(
            f"call_with_patches_probe: patches/value count mismatch: {len(patches)} patch(es), "
            f"{len(patch_values)} value(s)"
        )
    if any(v < 0 for v in patch_values):
        raise ValueError("call_with_patches_probe: patch values must be non-negative")
    if value < 0:
        raise ValueError(f"call_with_patches_probe: value must be non-negative, got {value}")
    if gas < 0:
        raise ValueError(f"call_with_patches_probe: gas must be non-negative, got {gas}")
    if len(calldata) > 0xFFFF:
        raise ValueError(f"call_with_patches_probe: calldata too large ({len(calldata)} bytes, max 65535)")

    normalized_offsets: list[int] = []
    for offset, size in patches:
        if not (0 < size <= 32):
            raise ValueError(f"call_with_patches_probe: patch size {size!r} not supported; expected 0 < size <= 32")
        mstore_off = offset + size - 32
        if mstore_off < 0:
            raise ValueError(
                f"call_with_patches_probe: offset {offset} is too small for size {size}; "
                f"MSTORE target {mstore_off} would be negative"
            )
        normalized_offsets.append(mstore_off)

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    blen = len(calldata)
    blen_padded = (blen + 31) & ~31
    payload = calldata.ljust(blen_padded, b"\x00") if blen_padded > blen else calldata

    ctx = IRContext()
    patches_payload_label = IRLabel("pydefi_patches_payload", is_symbol=True)
    ctx.append_data_section(patches_payload_label)
    ctx.append_data_item(payload)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    # fp_init(): fp | (0x280 * iszero(fp))
    fp = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)

    # Copy calldata template via offset+codecopy and apply patches in order.
    data_src = builder.offset(0, patches_payload_label)
    builder.codecopy(base_fp, data_src, blen_padded)
    for mstore_off, pv in zip(normalized_offsets, patch_values):
        patch_ptr = builder.add(base_fp, mstore_off)
        builder.mstore(patch_ptr, pv)

    builder.mstore(0x40, builder.add(base_fp, blen_padded))

    gas_operand = builder.gas() if gas == 0 else gas
    to_int = int.from_bytes(bytes(to), "big")
    success = builder.call(gas_operand, to_int, value, base_fp, blen, 0, 0)
    if require_success:
        builder.assert_(success)

    if return_success:
        builder.mstore(0, success)
        builder.return_(0, 32)
    else:
        builder.return_(0, 0)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def compile_venom_patch_preview_probe(
    calldata: bytes,
    *,
    patches: list[tuple[int, int]],
    patch_values: list[int],
    read_offset: int = 0,
) -> bytes:
    """Compile a Venom runtime that patches calldata and returns one memory word.

    This helper validates patching semantics independent of CALL behavior.
    It copies calldata from a data section to memory, applies all patches in
    order, then returns ``mload(argsOffset + read_offset)`` as a 32-byte word.
    """
    if not venom_is_available():
        raise ImportError(f"Vyper Venom APIs are unavailable: {venom_import_error()}")
    if len(patches) != len(patch_values):
        raise ValueError(
            f"patch_preview_probe: patches/value count mismatch: {len(patches)} patch(es), {len(patch_values)} value(s)"
        )
    if any(v < 0 for v in patch_values):
        raise ValueError("patch_preview_probe: patch values must be non-negative")
    if read_offset < 0:
        raise ValueError(f"patch_preview_probe: read_offset must be non-negative, got {read_offset}")
    if len(calldata) > 0xFFFF:
        raise ValueError(f"patch_preview_probe: calldata too large ({len(calldata)} bytes, max 65535)")

    normalized_offsets: list[int] = []
    for offset, size in patches:
        if not (0 < size <= 32):
            raise ValueError(f"patch_preview_probe: patch size {size!r} not supported; expected 0 < size <= 32")
        mstore_off = offset + size - 32
        if mstore_off < 0:
            raise ValueError(
                f"patch_preview_probe: offset {offset} is too small for size {size}; "
                f"MSTORE target {mstore_off} would be negative"
            )
        normalized_offsets.append(mstore_off)

    assert IRContext is not None
    assert IRLabel is not None
    assert VenomBuilder is not None
    assert VenomOptimizationFlags is not None
    assert OptimizationLevel is not None
    assert run_passes_on is not None
    assert generate_assembly_experimental is not None
    assert generate_bytecode is not None

    blen = len(calldata)
    blen_padded = (blen + 31) & ~31
    payload = calldata.ljust(blen_padded, b"\x00") if blen_padded > blen else calldata

    ctx = IRContext()
    patch_preview_label = IRLabel("pydefi_patch_preview", is_symbol=True)
    ctx.append_data_section(patch_preview_label)
    ctx.append_data_item(payload)

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)

    # fp_init(): fp | (0x280 * iszero(fp))
    fp = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)

    # Use offset + codecopy instead of dloadbytes (same fix as push_bytes probe).
    data_src = builder.offset(0, patch_preview_label)
    builder.codecopy(base_fp, data_src, blen_padded)
    for mstore_off, pv in zip(normalized_offsets, patch_values):
        patch_ptr = builder.add(base_fp, mstore_off)
        builder.mstore(patch_ptr, pv)

    # Return selected 32-byte word from patched calldata memory.
    word_ptr = builder.add(base_fp, read_offset)
    builder.mstore(0, builder.mload(word_ptr))
    builder.return_(0, 32)

    flags = VenomOptimizationFlags(
        level=OptimizationLevel.NONE,
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)
    bytecode, _ = generate_bytecode(asm)
    return bytecode


def create_program() -> Program:
    """Create a VM program builder instance.

    The VM builder surface now exposes a single Program type. Venom planning
    is used opportunistically when available; the same Program instance
    materializes to manual bytecode when Venom is unavailable.
    """
    return Program()
