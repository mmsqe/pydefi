"""Tests for the ProgramContext high-level Venom IR builder.

Verifies:
1. Basic construction and inherited methods work
2. ABI encode — simple scalar types, with and without method_id
3. ABI decode — decode a Bytes buffer back to Vyper memory layout
4. Compilation produces runnable bytecode (via mini_evm)
5. ABI encode + decode round-trip
6. Complex types: dynamic bytes, string, nested tuples, arrays
"""

from __future__ import annotations

import pytest
from vyper.compiler.settings import Settings, anchor_settings
from vyper.semantics.types.bytestrings import BytesT, StringT
from vyper.semantics.types.primitives import AddressT, BytesM_T
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.semantics.types.subscriptable import DArrayT, SArrayT, TupleT
from vyper.venom.basicblock import IRLiteral, IROperand, IRVariable
from vyper.venom.context import IRContext

from pydefi.vm.context import ProgramContext
from pydefi.vm.stdlib import build_stdlib
from tests.conftest import mini_evm

# mini_evm uses the Shanghai fork.  The default Venom EVM version is Prague
# (which emits Cancun opcodes like MCOPY), so we must pin the EVM version to
# "shanghai" during both IR construction (where version_check gates mcopy vs
# identity-precompile) and compilation.
_SHANGHAI_SETTINGS = Settings(evm_version="shanghai")


@pytest.fixture(autouse=True)
def _pin_shanghai_evm():
    """Force vyper EVM version to shanghai for every test in this module."""
    with anchor_settings(_SHANGHAI_SETTINGS):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_return_bytes_buffer(ctx: ProgramContext, buf: IROperand) -> None:
    """Return the Bytes buffer at *buf* via ``RETURN(memory[buf..buf+32+len])``.

    The buffer layout is: [length word (at buf+0)] [data (at buf+32)...].
    We read the length word, add 32 for the total return size, and RETURN
    from buf with that size.
    """
    b = ctx.builder
    assert isinstance(buf, IRVariable)
    length = b.mload(buf)
    size = b.add(length, IRLiteral(32))
    b.return_(buf, size)


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------


def test_construct():
    ctx = ProgramContext()
    assert ctx.ir_ctx.entry_function is not None
    assert str(ctx.ir_ctx.entry_function.name) == "main"
    assert ctx.builder is not None


def test_inherited_new_variable():
    ctx = ProgramContext()
    v = ctx.new_variable("x", UINT256_T)
    assert v.name == "x"
    assert str(v.typ) == "uint256"
    assert not v.value.is_stack_value


def test_inherited_lookup():
    ctx = ProgramContext()
    ctx.new_variable("y", UINT256_T)
    v = ctx.lookup("y")
    assert v.name == "y"


def test_basic_compile():
    ctx = ProgramContext()
    x = ctx.builder.literal(42)
    ctx.builder.mstore(ctx.builder.alloca(32), x)
    ctx.builder.stop()
    bytecode = ctx.compile()
    assert isinstance(bytecode, bytes)
    assert len(bytecode) > 0


def test_stdlib_then_program_compiles():
    ir_ctx = IRContext()
    build_stdlib(ir_ctx)
    ctx = ProgramContext(ir_ctx, "main")
    ctx.builder.stop()
    assert next(iter(ir_ctx.functions)) == ir_ctx.entry_function.name
    bytecode = ctx.compile()
    assert len(bytecode) > 0


# ---------------------------------------------------------------------------
# 2. ABI encode — simple scalars
# ---------------------------------------------------------------------------


def test_abi_encode_single_uint256():
    """abi_encode([42], [uint256]) should produce 64 bytes: length=32 + value."""
    ctx = ProgramContext()
    val = IRLiteral(42)
    buf = ctx.abi_encode([val], [UINT256_T])
    assert not buf.is_stack_value

    _build_return_bytes_buffer(ctx, buf.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    # Bytes layout: [32-byte length][32-byte ABI-encoded value]
    length = int.from_bytes(result.output[:32], "big")
    assert length == 32
    encoded_value = int.from_bytes(result.output[32:64], "big")
    assert encoded_value == 42


def test_abi_encode_multiple_scalars():
    """abi_encode([uint256, uint256]) should encode as a tuple."""
    ctx = ProgramContext()
    buf = ctx.abi_encode([IRLiteral(10), IRLiteral(20)], [UINT256_T, UINT256_T])

    _build_return_bytes_buffer(ctx, buf.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    # Tuple of two uint256 = 64 bytes of data
    length = int.from_bytes(result.output[:32], "big")
    assert length == 64
    assert int.from_bytes(result.output[32:64], "big") == 10
    assert int.from_bytes(result.output[64:96], "big") == 20


def test_abi_encode_with_method_id():
    """abi_encode with method_id should prepend 4 bytes."""
    ctx = ProgramContext()
    method_id = bytes.fromhex("a9059cbb")
    buf = ctx.abi_encode(
        [IRLiteral(1), IRLiteral(2)],
        [AddressT(), UINT256_T],
        method_id=method_id,
    )

    _build_return_bytes_buffer(ctx, buf.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    length = int.from_bytes(result.output[:32], "big")
    # Length = 4 (method_id) + 2 * 32 (address + uint256) = 68
    assert length == 68
    data = result.output[32:]
    # First 4 bytes = method_id (at the start of the data area)
    assert data[:4] == method_id


def test_abi_encode_address():
    """abi_encode an address — should be left-padded to 32 bytes."""
    ctx = ProgramContext()
    addr = IRLiteral(0xABCDEF0000000000000000000000000000001234)
    buf = ctx.abi_encode([addr], [AddressT()])

    _build_return_bytes_buffer(ctx, buf.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    encoded = result.output[32:64]
    # Address value is right-aligned (uint160) in the 32-byte slot
    assert encoded[12:] == addr.value.to_bytes(20, "big")


def test_abi_encode_bytes32():
    """abi_encode a bytes32 value."""
    ctx = ProgramContext()
    val = IRLiteral(int.from_bytes(b"hello" + b"\x00" * 27, "big"))
    buf = ctx.abi_encode([val], [BytesM_T(32)])

    _build_return_bytes_buffer(ctx, buf.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    length = int.from_bytes(result.output[:32], "big")
    assert length == 32
    assert result.output[32:37] == b"hello"


# ---------------------------------------------------------------------------
# 3. ABI decode (static data via embed_and_load)
# ---------------------------------------------------------------------------

# embed_and_load embeds raw byte data in a bytecode data-section and copies
# it into a memory buffer at runtime via codecopy (volatile).  This prevents
# the optimiser from eliminating the alloca-backed buffer, which would
# otherwise happen when the input data is a compile-time constant.


def test_abi_decode_uint256():
    """Decode a Bytes buffer containing an ABI-encoded uint256."""
    ctx = ProgramContext()

    buf = ctx.embed_and_load((32).to_bytes(32, "big") + (99).to_bytes(32, "big"))
    decoded = ctx.abi_decode(buf, UINT256_T, unwrap_tuple=False)
    loaded = ctx.builder.mload(decoded.operand)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 99


def test_abi_decode_reverts_on_short_data():
    """abi_decode with too-short data should revert."""
    ctx = ProgramContext()

    buf = ctx.embed_and_load((16).to_bytes(32, "big") + (42).to_bytes(32, "big"))
    ctx.abi_decode(buf, UINT256_T, unwrap_tuple=False)
    ctx.builder.stop()

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    # uint256 requires exactly 32 bytes of data
    assert result.is_error


# ---------------------------------------------------------------------------
# 4. Round-trip: encode then decode (static data only)
# ---------------------------------------------------------------------------


def test_abi_roundtrip_uint256():
    """Encode, then decode in separate programs — decode(encode(x)) == x."""
    ctx_enc = ProgramContext()
    buf = ctx_enc.abi_encode([IRLiteral(12345)], [UINT256_T])
    _build_return_bytes_buffer(ctx_enc, buf.operand)
    bytecode_enc = ctx_enc.compile()
    result_enc = mini_evm(bytecode_enc)
    assert not result_enc.is_error
    encoded = result_enc.output  # Bytes buffer: [length][data]

    ctx_dec = ProgramContext()
    buf_dec = ctx_dec.embed_and_load(encoded)
    decoded = ctx_dec.abi_decode(buf_dec, UINT256_T)
    loaded_val = ctx_dec.builder.mload(decoded.operand)
    out = ctx_dec.allocate_buffer(32)
    ctx_dec.builder.mstore(out.base_ptr().operand, loaded_val)
    ctx_dec.builder.return_(out.base_ptr().operand, IRLiteral(32))
    bytecode_dec = ctx_dec.compile()

    result_dec = mini_evm(bytecode_dec)
    assert not result_dec.is_error
    assert int.from_bytes(result_dec.output, "big") == 12345


def test_abi_roundtrip_two_uint256():
    """Round-trip a (uint256, uint256) tuple through encode / decode."""
    tuple_t = TupleT((UINT256_T, UINT256_T))

    ctx_enc = ProgramContext()
    buf = ctx_enc.abi_encode([IRLiteral(7), IRLiteral(42)], [UINT256_T, UINT256_T])
    _build_return_bytes_buffer(ctx_enc, buf.operand)
    bytecode_enc = ctx_enc.compile()
    result_enc = mini_evm(bytecode_enc)
    assert not result_enc.is_error
    encoded = result_enc.output

    ctx_dec = ProgramContext()
    buf_dec = ctx_dec.embed_and_load(encoded)
    decoded = ctx_dec.abi_decode(buf_dec, tuple_t)
    elem1_ptr = ctx_dec.builder.add(decoded.operand, IRLiteral(32))
    loaded = ctx_dec.builder.mload(elem1_ptr)
    out = ctx_dec.allocate_buffer(32)
    ctx_dec.builder.mstore(out.base_ptr().operand, loaded)
    ctx_dec.builder.return_(out.base_ptr().operand, IRLiteral(32))
    bytecode_dec = ctx_dec.compile()

    result_dec = mini_evm(bytecode_dec)
    assert not result_dec.is_error
    assert int.from_bytes(result_dec.output, "big") == 42


# ---------------------------------------------------------------------------
# 5. Complex types: dynamic bytes / string
# ---------------------------------------------------------------------------


def test_abi_encode_dynamic_bytes():
    """Encode a Bytes[32] value (dynamic bytestring) and check the ABI output."""
    ctx = ProgramContext()

    # Bytes buffer: [length word][data...]
    buf = ctx.embed_and_load((4).to_bytes(32, "big") + b"dead\x00" * 8)

    encoded = ctx.abi_encode([buf], [BytesT(32)], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    # Bytes encoding: [32-byte length][data padded to 32]
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == 64  # 32 + ceil32(4) = 32 + 32
    assert int.from_bytes(result.output[32:64], "big") == 4
    assert result.output[64:68] == b"dead"


def test_abi_encode_string():
    """Encode a String[32] value and check the ABI output."""
    ctx = ProgramContext()

    buf = ctx.embed_and_load((5).to_bytes(32, "big") + b"hello\x00" * 8)

    encoded = ctx.abi_encode([buf], [StringT(32)], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == 64  # 32 + ceil32(5)
    assert int.from_bytes(result.output[32:64], "big") == 5
    assert result.output[64:69] == b"hello"


def test_abi_decode_dynamic_bytes():
    """Decode ABI-encoded Bytes[32] data."""
    ctx = ProgramContext()

    # ABI encoding of Bytes[32]("dead") inside a tuple: [offset=32][len=4][data=padded]
    raw_abi = (32).to_bytes(32, "big") + (4).to_bytes(32, "big") + b"dead\x00" * 8
    # Bytes wrapper: [total_len=96][abi_data]
    buf = ctx.embed_and_load((96).to_bytes(32, "big") + raw_abi)
    decoded = ctx.abi_decode(buf, BytesT(32))

    _build_return_bytes_buffer(ctx, decoded.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    dec_len = int.from_bytes(result.output[:32], "big")
    assert dec_len == 4
    assert result.output[32:36] == b"dead"


# ---------------------------------------------------------------------------
# 6. Complex types: tuples
# ---------------------------------------------------------------------------


def test_abi_encode_nested_tuple():
    """Encode a nested tuple: (uint256, (address, uint256))."""
    ctx = ProgramContext()

    inner = TupleT((AddressT(), UINT256_T))
    outer = TupleT((UINT256_T, inner))

    addr_val = 0xCB00CB00CB00CB00CB00CB00CB00CB00CB00CB00
    raw = (10).to_bytes(32, "big") + addr_val.to_bytes(32, "big") + (42).to_bytes(32, "big")
    buf = ctx.embed_and_load(raw)

    encoded = ctx.abi_encode([buf], [outer], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == 96  # 3 static words = 96 bytes
    assert int.from_bytes(result.output[32:64], "big") == 10
    assert result.output[76:96] == b"\xcb\x00" * 10  # address at offset 44
    assert int.from_bytes(result.output[96:128], "big") == 42


def test_abi_decode_nested_tuple():
    """Decode a nested tuple from ABI-encoded input."""
    ctx = ProgramContext()

    inner = TupleT((AddressT(), UINT256_T))
    outer = TupleT((UINT256_T, inner))

    addr_val = 0xCB00CB00CB00CB00CB00CB00CB00CB00CB00CB00
    raw = (10).to_bytes(32, "big") + addr_val.to_bytes(32, "big") + (42).to_bytes(32, "big")
    buf = ctx.embed_and_load((96).to_bytes(32, "big") + raw)
    decoded = ctx.abi_decode(buf, outer)

    # Return the inner uint256 (offset 32 + 32 = 64)
    inner_ptr = ctx.builder.add(decoded.operand, IRLiteral(64))
    loaded = ctx.builder.mload(inner_ptr)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 42


# ---------------------------------------------------------------------------
# 7. Complex types: arrays
# ---------------------------------------------------------------------------


def test_abi_encode_static_array():
    """Encode a static array uint256[3]."""
    ctx = ProgramContext()

    arr_type = SArrayT(UINT256_T, 3)
    buf = ctx.embed_and_load((10).to_bytes(32, "big") + (20).to_bytes(32, "big") + (30).to_bytes(32, "big"))

    encoded = ctx.abi_encode([buf], [arr_type], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == 96  # 3 * 32 = 96
    assert int.from_bytes(result.output[32:64], "big") == 10
    assert int.from_bytes(result.output[64:96], "big") == 20
    assert int.from_bytes(result.output[96:128], "big") == 30


def test_abi_decode_static_array():
    """Decode a static array uint256[3] from ABI-encoded input."""
    ctx = ProgramContext()

    arr_type = SArrayT(UINT256_T, 3)
    raw = (10).to_bytes(32, "big") + (20).to_bytes(32, "big") + (30).to_bytes(32, "big")
    buf = ctx.embed_and_load((96).to_bytes(32, "big") + raw)
    decoded = ctx.abi_decode(buf, arr_type)

    # Return third element
    elem2_ptr = ctx.builder.add(decoded.operand, IRLiteral(64))
    loaded = ctx.builder.mload(elem2_ptr)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 30


def test_abi_encode_dynamic_array():
    """Encode a DynArray[uint256, 3]."""
    ctx = ProgramContext()

    arr_type = DArrayT(UINT256_T, 3)
    raw = (2).to_bytes(32, "big") + (100).to_bytes(32, "big") + (200).to_bytes(32, "big") + (0).to_bytes(32, "big")
    buf = ctx.embed_and_load(raw)

    encoded = ctx.abi_encode([buf], [arr_type], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == 96  # 32 + 2*32
    count = int.from_bytes(result.output[32:64], "big")
    assert count == 2
    assert int.from_bytes(result.output[64:96], "big") == 100
    assert int.from_bytes(result.output[96:128], "big") == 200


def test_abi_decode_dynamic_array():
    """Decode a DynArray[uint256, 3] from ABI-encoded input."""
    ctx = ProgramContext()

    arr_type = DArrayT(UINT256_T, 3)
    raw = (2).to_bytes(32, "big") + (10).to_bytes(32, "big") + (20).to_bytes(32, "big")
    buf = ctx.embed_and_load((96).to_bytes(32, "big") + raw)
    decoded = ctx.abi_decode(buf, arr_type, unwrap_tuple=False)

    # Return second element (offset 32 + 32 = 64)
    elem1_ptr = ctx.builder.add(decoded.operand, IRLiteral(64))
    loaded = ctx.builder.mload(elem1_ptr)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 20
