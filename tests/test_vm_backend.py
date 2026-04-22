import pytest
from hexbytes import HexBytes

from pydefi.types import RouteDAG, Token
from pydefi.vm import build_execution_program_for_dag, build_quote_program_for_dag
from pydefi.vm.builder import (
    Patch,
    Program,
    compile_venom_call_contract_probe,
    compile_venom_call_with_patch_probe,
    compile_venom_call_with_patches_probe,
    compile_venom_label_jump_probe,
    compile_venom_memory_progression_probe,
    compile_venom_patch_preview_probe,
    compile_venom_push_bytes_probe,
    compile_venom_smoke_bytecode,
    compile_venom_two_data_sections_probe,
    create_program,
    venom_is_available,
)
from pydefi.vm.program import add, pop, push_u256
from tests.conftest import RETURN_TOP, mini_evm


def test_create_program_defaults_to_venom_backend_when_available():
    program = create_program()
    assert type(program) is Program


def test_program_create_defaults_to_venom_backend_when_available():
    program = Program.create()
    assert type(program) is Program


def test_create_program_unknown_backend_raises():
    with pytest.raises(TypeError, match="backend"):
        create_program(backend="does-not-exist")


def test_program_create_unknown_backend_raises():
    with pytest.raises(TypeError, match="backend"):
        Program.create(backend="does-not-exist")


def test_create_program_venom_falls_back_when_unavailable_only_if_not_required():
    program = create_program(require_venom=False)
    assert isinstance(program, Program)


def test_create_program_manual_backend_is_rejected():
    with pytest.raises(TypeError, match="backend"):
        create_program(backend="manual")


def test_program_create_manual_backend_is_rejected():
    with pytest.raises(TypeError, match="backend"):
        Program.create(backend="manual")


def test_create_program_default_selection_uses_venom_when_available():
    program = create_program()
    assert type(program) is Program


def test_program_create_default_selection_uses_venom_when_available():
    program = Program.create()
    assert type(program) is Program


def test_backend_parity_addition_program():
    program = Program.create(require_venom=False)
    bytecode = program.push_u256(3).push_u256(5)._emit(add()).build()
    result = mini_evm(bytecode + RETURN_TOP)
    assert not result.is_error, "venom backend add program reverted"
    assert int.from_bytes(result.output, "big") == 8


def test_backend_parity_push_bytes_length_semantics():
    payload = b"abcdef"
    program = Program.create(require_venom=False)
    bytecode = program.push_bytes(payload)._emit(pop())._emit(RETURN_TOP).build()
    result = mini_evm(bytecode)
    assert not result.is_error, "venom backend push_bytes length probe reverted"
    assert int.from_bytes(result.output, "big") == len(payload)


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_create_program_venom_returns_venom_program_when_available():
    program = create_program(require_venom=True)
    assert type(program) is Program


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_smoke_bytecode():
    bytecode = compile_venom_smoke_bytecode()
    assert isinstance(bytecode, bytes)
    assert len(bytecode) > 0


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_push_bytes_probe_copies_data_into_memory():
    payload = b"\xde\xad\xbe\xef"
    bytecode = compile_venom_push_bytes_probe(payload)
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom push-bytes probe reverted: {result.output.hex()}"
    expected = int.from_bytes(payload.ljust(32, b"\x00"), "big")
    assert int.from_bytes(result.output, "big") == expected


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_push_bytes_probe_embeds_data_section():
    payload = b"hello venom"
    bytecode = compile_venom_push_bytes_probe(payload)
    assert bytecode.endswith(payload.ljust(((len(payload) + 31) & ~31), b"\x00"))


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_label_jump_probe_then_branch():
    bytecode = compile_venom_label_jump_probe(take_then_branch=True)
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom label/jump probe (then) reverted: {result.output.hex()}"
    assert int.from_bytes(result.output, "big") == 1


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_label_jump_probe_else_branch():
    bytecode = compile_venom_label_jump_probe(take_then_branch=False)
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom label/jump probe (else) reverted: {result.output.hex()}"
    assert int.from_bytes(result.output, "big") == 2


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_two_data_sections_probe_embeds_both_sections():
    data_a = b"section-a"
    data_b = b"section-b"
    bytecode = compile_venom_two_data_sections_probe(data_a, data_b)
    assert data_a in bytecode
    assert data_b in bytecode


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_memory_progression_probe_returns_expected_free_ptr():
    data_a = b"abcd"  # padded 32
    data_b = b"1234567890"  # padded 32
    bytecode = compile_venom_memory_progression_probe(data_a, data_b)
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom memory progression probe reverted: {result.output.hex()}"
    expected = 0x280 + 32 + 32
    assert int.from_bytes(result.output, "big") == expected


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_call_contract_probe_returns_success_flag():
    # CALL to an EOA/non-contract with value=0 should succeed and return 1.
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
def test_compile_venom_call_with_patch_probe_returns_success_flag():
    target = HexBytes("0x" + "aa" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64
    bytecode = compile_venom_call_with_patch_probe(
        target,
        template,
        patch_value=123,
        patch_offset=4,
        patch_size=32,
    )
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom call_with_patch probe reverted: {result.output.hex()}"
    assert int.from_bytes(result.output, "big") == 1


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_call_with_patch_probe_matches_manual_success_semantics():
    target = HexBytes("0x" + "ab" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64

    manual = Program().push_u256(123).call_with_patches(target, template, [(4, 32)])._emit(RETURN_TOP).build()
    manual_result = mini_evm(manual)
    assert not manual_result.is_error

    venom = compile_venom_call_with_patch_probe(
        target,
        template,
        patch_value=123,
        patch_offset=4,
        patch_size=32,
    )
    venom_result = mini_evm(venom)
    assert not venom_result.is_error

    assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")


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


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_patch_preview_probe_single_patch_matches_expected_word():
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64
    patch_val = 123
    bytecode = compile_venom_patch_preview_probe(
        template,
        patches=[(4, 32)],
        patch_values=[patch_val],
        read_offset=0,
    )
    result = mini_evm(bytecode)
    assert not result.is_error, f"venom patch preview probe reverted: {result.output.hex()}"

    expected_word = int.from_bytes(bytes.fromhex("a9059cbb") + b"\x00" * 28, "big")
    assert int.from_bytes(result.output, "big") == expected_word


@pytest.mark.skipif(not venom_is_available(), reason="Vyper Venom APIs are unavailable in this environment")
def test_compile_venom_patch_preview_probe_matches_manual_for_patched_word():
    template = bytes.fromhex("12345678") + b"\x00" * 64
    patch_values = [111, int.from_bytes(HexBytes("0x" + "22" * 20), "big")]

    # Manual path: patch bytes then read first patched argument word at offset 4.
    manual = (
        Program()
        .push_u256(patch_values[1])
        .push_u256(patch_values[0])
        .push_bytes(template)
        .patch_bytes_from_stack([(4, 32), (36, 20)])
        ._emit(pop())  # drop argsLen
        ._emit(push_u256(4))
        ._emit(add())
        ._emit(bytes([0x51]))  # MLOAD at argsOffset+4
        ._emit(RETURN_TOP)
        .build()
    )
    manual_result = mini_evm(manual)
    assert not manual_result.is_error

    venom = compile_venom_patch_preview_probe(
        template,
        patches=[(4, 32), (36, 20)],
        patch_values=patch_values,
        read_offset=4,
    )
    venom_result = mini_evm(venom)
    assert not venom_result.is_error

    assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")


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


def test_backend_parity_call_contract_bytecode():
    target = HexBytes("0x" + "77" * 20)
    calldata = b"\x12\x34\x56\x78"

    manual = Program().call_contract(target, calldata).build()
    venom = Program.create(require_venom=False).call_contract(target, calldata).build()
    if venom_is_available():
        manual_result = mini_evm(manual + RETURN_TOP)
        venom_result = mini_evm(venom + RETURN_TOP)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")
    else:
        assert venom == manual


def test_backend_parity_call_with_patches_bytecode():
    target = HexBytes("0x" + "88" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64
    patches = [(4, 32)]

    manual = Program().call_with_patches(target, template, patches).build()
    venom = Program.create(require_venom=False).call_with_patches(target, template, patches).build()
    assert venom == manual


def test_venom_program_call_with_patches_empty_uses_behavior_parity():
    target = HexBytes("0x" + "8a" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64

    manual = Program().call_with_patches(target, template, []).build()
    venom = Program.create(require_venom=False).call_with_patches(target, template, []).build()

    if venom_is_available():
        manual_result = mini_evm(manual + RETURN_TOP)
        venom_result = mini_evm(venom + RETURN_TOP)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")
    else:
        assert venom == manual


def test_venom_program_call_with_patches_stack_literal_case_behavior_parity():
    target = HexBytes("0x" + "8b" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64

    venom = Program.create(require_venom=False).push_u256(123).call_with_patches(target, template, [(4, 32)]).build()
    manual = Program().push_u256(123).call_with_patches(target, template, [(4, 32)]).build()
    if venom_is_available():
        manual_result = mini_evm(manual + RETURN_TOP)
        venom_result = mini_evm(venom + RETURN_TOP)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")
    else:
        assert venom == manual


def test_venom_program_call_with_patches_non_literal_stack_case_falls_back_to_manual():
    target = HexBytes("0x" + "8e" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64

    venom = (
        Program.create(require_venom=False)
        .call_contract(target, b"\x12\x34")
        .ret_u256(0)
        .call_with_patches(target, template, [(4, 32)])
        .build()
    )
    manual = (
        Program().call_contract(target, b"\x12\x34").ret_u256(0).call_with_patches(target, template, [(4, 32)]).build()
    )
    assert venom == manual


def test_venom_program_call_with_patches_empty_then_pop_behavior_parity():
    target = HexBytes("0x" + "8c" * 20)
    template = bytes.fromhex("a9059cbb") + b"\x00" * 64

    manual = Program().call_with_patches(target, template, []).pop().build()
    venom = Program.create(require_venom=False).call_with_patches(target, template, []).pop().build()

    if venom_is_available():
        manual_result = mini_evm(manual)
        venom_result = mini_evm(venom)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert venom_result.output == manual_result.output
    else:
        assert venom == manual


def test_venom_program_call_then_pop_then_mutate_falls_back_to_manual():
    target = HexBytes("0x" + "8d" * 20)
    calldata = b"\x12\x34"

    venom = Program.create(require_venom=False).call_contract(target, calldata).pop().push_u256(1).build()
    manual = Program().call_contract(target, calldata).pop().push_u256(1).build()
    assert venom == manual


def test_venom_program_call_contract_falls_back_to_manual_when_mutated():
    target = HexBytes("0x" + "7a" * 20)
    calldata = b"\x12\x34\x56\x78"

    venom = Program.create(require_venom=False).call_contract(target, calldata).pop().build()
    manual = Program().call_contract(target, calldata).pop().build()
    assert venom == manual


def test_venom_program_call_contract_then_pop_behavior_parity():
    target = HexBytes("0x" + "7b" * 20)
    calldata = b"\x12\x34\x56\x78"

    manual = Program().call_contract(target, calldata).pop().build()
    venom = Program.create(require_venom=False).call_contract(target, calldata).pop().build()

    if venom_is_available():
        manual_result = mini_evm(manual)
        venom_result = mini_evm(venom)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert venom_result.output == manual_result.output
    else:
        assert venom == manual


def test_venom_program_call_contract_abi_no_patch_behavior_parity():
    target = HexBytes("0x" + "9a" * 20)

    manual = Program().call_contract_abi(target, "function ping() external").build()
    venom = Program.create(require_venom=False).call_contract_abi(target, "function ping() external").build()

    if venom_is_available():
        manual_result = mini_evm(manual + RETURN_TOP)
        venom_result = mini_evm(venom + RETURN_TOP)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")
    else:
        assert venom == manual


def test_venom_program_call_contract_abi_patch_case_behavior_parity():
    target = HexBytes("0x" + "9b" * 20)

    venom = (
        Program.create(require_venom=False)
        .push_u256(123)
        .call_contract_abi(target, "function set(uint256 value) external", Patch())
        .build()
    )
    manual = Program().push_u256(123).call_contract_abi(target, "function set(uint256 value) external", Patch()).build()
    if venom_is_available():
        manual_result = mini_evm(manual + RETURN_TOP)
        venom_result = mini_evm(venom + RETURN_TOP)
        assert not manual_result.is_error
        assert not venom_result.is_error
        assert int.from_bytes(venom_result.output, "big") == int.from_bytes(manual_result.output, "big")
    else:
        assert venom == manual


def test_venom_program_call_contract_abi_no_patch_falls_back_when_mutated():
    target = HexBytes("0x" + "9c" * 20)

    venom = Program.create(require_venom=False).call_contract_abi(target, "function ping() external").pop().build()
    manual = Program().call_contract_abi(target, "function ping() external").pop().build()
    assert venom == manual


def test_venom_program_plan_state_for_call_contract():
    program = Program.create(require_venom=False)
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    assert program.has_pending_venom_plan is False
    assert program.pending_venom_plan_kind is None

    program.call_contract(HexBytes("0x" + "9d" * 20), b"\x12\x34")
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_contract"


def test_venom_program_plan_clears_when_mutated_after_call_contract():
    program = Program.create(require_venom=False)
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
    program = Program.create(require_venom=False)
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    program.call_with_patches(HexBytes("0x" + "9f" * 20), b"\xaa\xbb", [])
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_with_patches"


def test_venom_program_plan_for_stack_literal_patch_case():
    program = Program.create(require_venom=False)
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    program.push_u256(123).call_with_patches(HexBytes("0x" + "90" * 20), b"\xaa\xbb", [(0, 1)])
    assert program.has_pending_venom_plan is True
    assert program.pending_venom_plan_kind == "call_with_patches"


def test_venom_program_no_plan_for_mismatched_stack_literal_patch_case():
    program = Program.create(require_venom=False)
    if not venom_is_available():
        pytest.skip("Venom backend unavailable")

    program.push_u256(1).push_u256(2).call_with_patches(HexBytes("0x" + "91" * 20), b"\xaa\xbb", [(0, 1)])
    assert program.has_pending_venom_plan is False
    assert program.pending_venom_plan_kind is None
