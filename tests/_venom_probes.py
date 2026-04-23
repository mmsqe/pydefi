"""Test-only Venom probe helpers.

Standalone Venom-compiled programs used by tests for parity checks against
:class:`pydefi.vm.builder.Program`'s production fragment path.  These were
previously colocated in :mod:`pydefi.vm.builder` but are not used by any
production code, so they live here to keep the production surface lean.
"""

from typing import Any

from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.basicblock import IRLabel
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

from pydefi.types import Address


def _pad32(data: bytes) -> bytes:
    """Return *data* zero-padded to the next 32-byte boundary."""
    return data.ljust((len(data) + 31) & ~31, b"\x00")


def _compile_venom_ctx(ctx: IRContext) -> bytes:
    """Run the standard Venom pipeline on *ctx* and return the compiled bytecode.

    Used by the probe helpers below.  Production code (the call/push_bytes
    fragments in :mod:`pydefi.vm.builder`) compiles inline so it can also
    inspect the asm list for label resolution.
    """
    flags = VenomOptimizationFlags(  # type: ignore[call-arg]
        level=OptimizationLevel.NONE,  # type: ignore[attr-defined]
        disable_mem2var=True,
        disable_load_elimination=True,
        disable_dead_store_elimination=True,
    )
    run_passes_on(ctx, flags, disable_mem_checks=True)  # type: ignore[misc]
    asm = generate_assembly_experimental(ctx, optimize=OptimizationLevel.NONE)  # type: ignore[misc,attr-defined]
    bytecode, _ = generate_bytecode(asm)  # type: ignore[misc]
    return bytecode


def _venom_alloc_and_copy(builder: VenomBuilder, label: IRLabel, blen_padded: int) -> Any:
    """Emit fp_init + CODECOPY from *label* into memory[base_fp..+blen_padded], advance FP.

    Returns the ``base_fp`` SSA value for use in subsequent builder instructions.
    """
    fp = builder.mload(0x40)
    default_fp = builder.mul(builder.iszero(fp), 0x280)
    base_fp = builder.or_(fp, default_fp)
    data_src = builder.offset(0, label)
    builder.codecopy(base_fp, data_src, blen_padded)
    builder.mstore(0x40, builder.add(base_fp, blen_padded))
    return base_fp


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
    if len(to) != 20:
        raise ValueError(f"call_contract_probe: bad address length: {to!r}")
    if value < 0:
        raise ValueError(f"call_contract_probe: value must be non-negative, got {value}")
    if gas < 0:
        raise ValueError(f"call_contract_probe: gas must be non-negative, got {gas}")
    if len(calldata) > 0xFFFF:
        raise ValueError(f"call_contract_probe: calldata too large ({len(calldata)} bytes, max 65535)")

    blen = len(calldata)
    blen_padded = (blen + 31) & ~31

    ctx = IRContext()  # type: ignore[operator]
    call_payload_label = IRLabel("pydefi_call_payload", is_symbol=True)  # type: ignore[operator]
    ctx.append_data_section(call_payload_label)
    ctx.append_data_item(_pad32(calldata))

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)  # type: ignore[operator]

    base_fp = _venom_alloc_and_copy(builder, call_payload_label, blen_padded)

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

    return _compile_venom_ctx(ctx)


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

    blen = len(calldata)
    blen_padded = (blen + 31) & ~31

    ctx = IRContext()  # type: ignore[operator]
    patches_payload_label = IRLabel("pydefi_patches_payload", is_symbol=True)  # type: ignore[operator]
    ctx.append_data_section(patches_payload_label)
    ctx.append_data_item(_pad32(calldata))

    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)  # type: ignore[operator]

    base_fp = _venom_alloc_and_copy(builder, patches_payload_label, blen_padded)

    for mstore_off, pv in zip(normalized_offsets, patch_values):
        builder.mstore(builder.add(base_fp, mstore_off), pv)

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

    return _compile_venom_ctx(ctx)
