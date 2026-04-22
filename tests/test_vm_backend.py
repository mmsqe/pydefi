import pytest
from hexbytes import HexBytes

from pydefi.types import RouteDAG, Token
from pydefi.vm import build_execution_program_for_dag, build_quote_program_for_dag
from pydefi.vm.builder import (
    IRContext,
    IRLabel,
    Patch,
    Program,
    VenomBuilder,
    _compile_venom_ctx,
    _pad32,
    _venom_alloc_and_copy,
    compile_venom_call_contract_probe,
    compile_venom_call_with_patches_probe,
    venom_is_available,
)
from pydefi.vm.program import add, pop
from tests.conftest import RETURN_TOP, mini_evm

# ---------------------------------------------------------------------------
# Basic factory / execution smoke tests
# ---------------------------------------------------------------------------


def test_backend_parity_addition_program():
    bytecode = Program().push_u256(3).push_u256(5)._emit(add()).build()
    result = mini_evm(bytecode + RETURN_TOP)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 8


def test_backend_parity_push_bytes_length_semantics():
    payload = b"abcdef"
    bytecode = Program().push_bytes(payload)._emit(pop())._emit(RETURN_TOP).build()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == len(payload)


# ---------------------------------------------------------------------------
# Venom compilation pipeline probes
# (skipped when Vyper is not installed)
# ---------------------------------------------------------------------------


def _push_bytes_probe(data: bytes) -> bytes:
    """Local test helper: compile a Venom program that copies *data* into memory
    and returns the first 32-byte word, for CODECOPY correctness verification."""
    blen_padded = (len(data) + 31) & ~31
    ctx = IRContext()  # type: ignore[operator]
    label = IRLabel("pydefi_payload", is_symbol=True)  # type: ignore[operator]
    ctx.append_data_section(label)
    ctx.append_data_item(_pad32(data))
    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder = VenomBuilder(ctx, fn)  # type: ignore[operator]
    # Use offset+codecopy (not dloadbytes) to avoid the LowerDloadPass code_end bug.
    base_fp = _venom_alloc_and_copy(builder, label, blen_padded)
    builder.mstore(0, builder.mload(base_fp))
    builder.return_(0, 32)
    return _compile_venom_ctx(ctx)


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_push_bytes_probe_copies_data_into_memory():
    payload = b"\xde\xad\xbe\xef"
    result = mini_evm(_push_bytes_probe(payload))
    assert not result.is_error, f"venom push-bytes probe reverted: {result.output.hex()}"
    assert int.from_bytes(result.output, "big") == int.from_bytes(payload.ljust(32, b"\x00"), "big")


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_push_bytes_probe_embeds_data_section():
    payload = b"hello venom"
    bytecode = _push_bytes_probe(payload)
    assert bytecode.endswith(payload.ljust(((len(payload) + 31) & ~31), b"\x00"))


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_call_contract_probe_returns_success_flag():
    target = HexBytes("0x" + "99" * 20)
    bytecode = compile_venom_call_contract_probe(target, b"\x12\x34\x56\x78")
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom call_contract probe reverted: {result.output.hex()}"
    assert int.from_bytes(result.output, "big") == 1


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_call_contract_probe_rejects_bad_address_length():
    with pytest.raises(ValueError, match="bad address length"):
        compile_venom_call_contract_probe(HexBytes("0x1234"), b"")


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_call_with_patches_probe_returns_success_flag():
    target = HexBytes("0x" + "ac" * 20)
    template = bytes.fromhex("12345678") + b"\x00" * 64
    bytecode = compile_venom_call_with_patches_probe(
        target,
        template,
        patches=[(4, 32), (36, 20)],
        patch_values=[111, int.from_bytes(HexBytes("0x" + "11" * 20), "big")],
    )
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom call_with_patches probe reverted: {result.output.hex()}"
    assert int.from_bytes(result.output, "big") == 1


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_call_with_patches_probe_matches_manual_success_semantics():
    target = HexBytes("0x" + "ad" * 20)
    template = bytes.fromhex("12345678") + b"\x00" * 64
    patch_values = [111, int.from_bytes(HexBytes("0x" + "22" * 20), "big")]

    manual = (
        Program()
        .push_u256(patch_values[1])
        .push_u256(patch_values[0])
        .call_with_patches(target, template, [(4, 32), (36, 20)])
        ._emit(RETURN_TOP)
        .build()
    )
    manual_result = mini_evm(manual)
    assert not manual_result.is_error

    venom = compile_venom_call_with_patches_probe(
        target,
        template,
        patches=[(4, 32), (36, 20)],
        patch_values=patch_values,
    )
    venom_result = mini_evm(venom)
    assert not venom_result.is_error

    assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")


# ---------------------------------------------------------------------------
# DAG program builder
# ---------------------------------------------------------------------------


def _mk_test_dag() -> RouteDAG:
    token_in = Token(chain_id=1, address=HexBytes("0x" + "11" * 20), symbol="TKA", decimals=18)
    token_out = Token(chain_id=1, address=HexBytes("0x" + "22" * 20), symbol="TKB", decimals=18)

    class _DummyPool:
        def __init__(self) -> None:
            self.protocol = "uniswap_v2"
            self.pool_address = HexBytes("0x" + "33" * 20)
            self.token_in = token_in
            self.token_out = token_out
            self.fee_bps = 30

        def zero_for_one(self, _token_out: HexBytes) -> bool:
            return True

    return RouteDAG().from_token(token_in).swap(token_out, _DummyPool())


def test_build_execution_program_for_dag_uses_backend_selection():
    program = build_execution_program_for_dag(
        _mk_test_dag(),
        amount_in=123,
        vm_address="0x" + "44" * 20,
        recipient="0x" + "55" * 20,
    )
    assert type(program) is Program


def test_build_quote_program_for_dag_uses_backend_selection():
    program = build_quote_program_for_dag(
        _mk_test_dag(),
        amount_in=123,
    )
    assert type(program) is Program


# ---------------------------------------------------------------------------
# Program EVM behavior: call_contract and call_with_patches
# ---------------------------------------------------------------------------


def test_call_contract_then_pop_evm_behavior():
    target = HexBytes("0x" + "7b" * 20)
    calldata = b"\x12\x34\x56\x78"
    result = mini_evm(Program().call_contract(target, calldata).pop().build())
    assert not result.is_error
    assert result.output == b""


def test_call_with_patches_empty_then_pop_evm_behavior():
    target = HexBytes("0x" + "8c" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64
    result = mini_evm(Program().call_with_patches(target, template, []).pop().build())
    assert not result.is_error
    assert result.output == b""


def test_call_with_patches_stack_literal_evm_behavior():
    target = HexBytes("0x" + "8b" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64
    result = mini_evm(Program().push_u256(123).call_with_patches(target, template, [(4, 32)])._emit(RETURN_TOP).build())
    assert not result.is_error


def test_call_contract_abi_no_patch_evm_behavior():
    target = HexBytes("0x" + "9a" * 20)
    result = mini_evm(Program().call_contract_abi(target, "function ping() external").build() + RETURN_TOP)
    assert not result.is_error


def test_call_contract_abi_patch_evm_behavior():
    target = HexBytes("0x" + "9b" * 20)
    result = mini_evm(
        Program().push_u256(123).call_contract_abi(target, "function set(uint256 value) external", Patch()).build()
        + RETURN_TOP
    )
    assert not result.is_error


# ---------------------------------------------------------------------------
# Venom plan state introspection
# ---------------------------------------------------------------------------


def test_venom_program_plan_state_for_call_contract():
    program = Program.create()
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    assert program.has_pending_venom_plan is False
    assert program.pending_venom_plan_kind is None

    program.call_contract(HexBytes("0x" + "9d" * 20), b"\x12\x34")
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_contract"


def test_venom_program_plan_clears_when_mutated_after_call_contract():
    program = Program.create()
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    program.call_contract(HexBytes("0x" + "9e" * 20), b"\x12\x34")
    assert program.has_pending_venom_plan is True

    # Trailing POP is now represented in the pending Venom plan.
    program.pop()
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_contract"

    # Any extra emission after planned call+pop forces materialization.
    program.push_u256(1)
    assert program.has_pending_venom_plan is False
    assert program.pending_venom_plan_kind is None


def test_venom_program_plan_state_for_empty_call_with_patches():
    program = Program.create()
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    program.call_with_patches(HexBytes("0x" + "9f" * 20), b"\xaa\xbb", [])
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_with_patches"


def test_venom_program_plan_for_stack_literal_patch_case():
    program = Program.create()
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    program.push_u256(123).call_with_patches(HexBytes("0x" + "90" * 20), b"\xaa\xbb", [(0, 1)])
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_with_patches"


def test_venom_program_no_plan_for_mismatched_stack_literal_patch_case():
    program = Program.create()
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    # 2 stack values pushed but only 1 patch → mismatch, Venom plan not set.
    # Use a valid patch descriptor (offset=28, size=4 → mstore_off=0) so the
    # manual fallback path doesn't raise on the negative-mstore-target check.
    calldata = b"\xaa\xbb\xcc\xdd" + b"\x00" * 28  # 32-byte calldata
    program.push_u256(1).push_u256(2).call_with_patches(HexBytes("0x" + "91" * 20), calldata, [(28, 4)])
    assert program.has_pending_venom_plan is False
    assert program.pending_venom_plan_kind is None
