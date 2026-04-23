from typing import Any

import pytest
from hexbytes import HexBytes

from pydefi.types import BasePool, RouteDAG, SwapProtocol, Token
from pydefi.vm import build_execution_program_for_dag, build_quote_program_for_dag
from pydefi.vm.builder import (
    _FRAGMENT_DATA_SRC_PLACEHOLDER,
    IRContext,
    IRLabel,
    Patch,
    Program,
    VenomBuilder,
    _compile_venom_ctx,
    _pad32,
    _venom_alloc_and_copy,
    compile_venom_call_contract_fragment,
    compile_venom_call_contract_probe,
    compile_venom_call_with_patches_probe,
)
from pydefi.vm.program import add, pop
from tests.conftest import RETURN_TOP, mini_evm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _push_bytes_probe(data: bytes) -> bytes:
    blen_padded = (len(data) + 31) & ~31
    ctx: Any = IRContext()
    label: Any = IRLabel("pydefi_payload", is_symbol=True)
    ctx.append_data_section(label)
    ctx.append_data_item(_pad32(data))
    fn = ctx.create_function("main")
    ctx.entry_function = fn
    builder: Any = VenomBuilder(ctx, fn)
    # Use offset+codecopy (not dloadbytes) to avoid the LowerDloadPass code_end bug.
    base_fp = _venom_alloc_and_copy(builder, label, blen_padded)
    builder.mstore(0, builder.mload(base_fp))
    builder.return_(0, 32)
    return _compile_venom_ctx(ctx)


def _mk_test_dag() -> RouteDAG:
    token_in = Token(chain_id=1, address=HexBytes("0x" + "11" * 20), symbol="TKA", decimals=18)
    token_out = Token(chain_id=1, address=HexBytes("0x" + "22" * 20), symbol="TKB", decimals=18)

    class _DummyPool(BasePool):
        protocol = SwapProtocol.UNISWAP_V2
        pool_address = HexBytes("0x" + "33" * 20)
        fee_bps = 30

        def __init__(self) -> None:
            self.token_in = token_in
            self.token_out = token_out

        def zero_for_one(self, _token_out: HexBytes) -> bool:
            return True

    return RouteDAG().from_token(token_in).swap(token_out, _DummyPool())


# ---------------------------------------------------------------------------
# Program basics
# ---------------------------------------------------------------------------


class TestProgramBasics:
    def test_addition(self):
        bytecode = Program().push_u256(3).push_u256(5)._emit(add()).build()
        result = mini_evm(bytecode + RETURN_TOP)
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 8

    def test_push_bytes_length(self):
        payload = b"abcdef"
        bytecode = Program().push_bytes(payload)._emit(pop())._emit(RETURN_TOP).build()
        result = mini_evm(bytecode)
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == len(payload)


# ---------------------------------------------------------------------------
# Venom data section (push_bytes probe)
# ---------------------------------------------------------------------------


class TestPushBytesProbe:
    def test_copies_data_into_memory(self):
        payload = b"\xde\xad\xbe\xef"
        result = mini_evm(_push_bytes_probe(payload))
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == int.from_bytes(payload.ljust(32, b"\x00"), "big")

    def test_embeds_data_section(self):
        payload = b"hello venom"
        bytecode = _push_bytes_probe(payload)
        assert bytecode.endswith(payload.ljust(((len(payload) + 31) & ~31), b"\x00"))


# ---------------------------------------------------------------------------
# compile_venom_call_contract_probe
# ---------------------------------------------------------------------------


class TestCallContractProbe:
    TARGET = HexBytes("0x" + "99" * 20)

    def test_returns_success_flag(self):
        result = mini_evm(compile_venom_call_contract_probe(self.TARGET, b"\x12\x34\x56\x78"))
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 1

    def test_rejects_bad_address_length(self):
        with pytest.raises(ValueError, match="bad address length"):
            compile_venom_call_contract_probe(HexBytes("0x1234"), b"")


# ---------------------------------------------------------------------------
# compile_venom_call_with_patches_probe
# ---------------------------------------------------------------------------


class TestCallWithPatchesProbe:
    TARGET = HexBytes("0x" + "ac" * 20)
    TEMPLATE = bytes.fromhex("12345678") + b"\x00" * 64
    PATCHES = [(4, 32), (36, 20)]

    def test_returns_success_flag(self):
        result = mini_evm(
            compile_venom_call_with_patches_probe(
                self.TARGET,
                self.TEMPLATE,
                patches=self.PATCHES,
                patch_values=[111, int.from_bytes(HexBytes("0x" + "11" * 20), "big")],
            )
        )
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 1

    def test_matches_manual_program(self):
        target = HexBytes("0x" + "ad" * 20)
        patch_values = [111, int.from_bytes(HexBytes("0x" + "22" * 20), "big")]
        manual = (
            Program()
            .push_u256(patch_values[1])
            .push_u256(patch_values[0])
            .call_with_patches(target, self.TEMPLATE, self.PATCHES)
            ._emit(RETURN_TOP)
            .build()
        )
        venom = compile_venom_call_with_patches_probe(
            target, self.TEMPLATE, patches=self.PATCHES, patch_values=patch_values
        )
        assert not mini_evm(manual).is_error
        assert int.from_bytes(mini_evm(venom).output, "big") == int.from_bytes(mini_evm(manual).output, "big")


# ---------------------------------------------------------------------------
# DAG builder
# ---------------------------------------------------------------------------


class TestDagBuilder:
    def test_execution_program_type(self):
        program = build_execution_program_for_dag(
            _mk_test_dag(), amount_in=123, vm_address="0x" + "44" * 20, recipient="0x" + "55" * 20
        )
        assert type(program) is Program

    def test_quote_program_type(self):
        assert type(build_quote_program_for_dag(_mk_test_dag(), amount_in=123)) is Program


# ---------------------------------------------------------------------------
# Program.call_contract
# ---------------------------------------------------------------------------


class TestCallContract:
    TARGET = HexBytes("0x" + "7b" * 20)
    CALLDATA = b"\x12\x34\x56\x78"

    def test_rejects_bad_address_length(self):
        with pytest.raises(ValueError, match="bad address length"):
            Program().call_contract(HexBytes("0x1234"), b"")

    def test_rejects_negative_value(self):
        with pytest.raises(ValueError, match="value must be non-negative"):
            Program().call_contract(self.TARGET, b"", value=-1)

    def test_rejects_negative_gas(self):
        with pytest.raises(ValueError, match="gas must be non-negative"):
            Program().call_contract(self.TARGET, b"", gas=-5)

    def test_evm_call_then_pop(self):
        result = mini_evm(Program().call_contract(self.TARGET, self.CALLDATA).pop().build())
        assert not result.is_error
        assert result.output == b""

    def test_fragment_ends_with_call_opcode(self):
        code_frag, fixup_pos = compile_venom_call_contract_fragment(self.TARGET, self.CALLDATA)
        assert code_frag[-1] == 0xF1, f"expected CALL (0xF1), got 0x{code_frag[-1]:02x}"
        assert code_frag[fixup_pos : fixup_pos + 4] == b"\x00\x00\x00\x00"
        assert _FRAGMENT_DATA_SRC_PLACEHOLDER.to_bytes(4, "big") not in bytes(code_frag)

    def test_require_success_pc_relative_after_compose(self):
        """require_success JUMPI stays correct when call_contract is shifted by a preamble."""
        preamble = Program().push_u256(42)
        call_prog = Program().call_contract(HexBytes("0x" + "9f" * 20), self.CALLDATA).pop()
        result = mini_evm((preamble + call_prog)._emit(RETURN_TOP).build())
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 42

    def test_two_calls_composed(self):
        t1, t2 = HexBytes("0x" + "a1" * 20), HexBytes("0x" + "a2" * 20)
        result = mini_evm(
            Program().call_contract(t1, b"\x11\x22\x33\x44").pop().call_contract(t2, b"\x55\x66\x77\x88").pop().build()
        )
        assert not result.is_error
        assert result.output == b""


# ---------------------------------------------------------------------------
# Program.call_with_patches
# ---------------------------------------------------------------------------


class TestCallWithPatches:
    TEMPLATE = bytes.fromhex("a9059cbb") + b"\x00" * 64

    def test_empty_patches(self):
        result = mini_evm(Program().call_with_patches(HexBytes("0x" + "8c" * 20), self.TEMPLATE, []).pop().build())
        assert not result.is_error
        assert result.output == b""

    def test_single_u256_patch(self):
        result = mini_evm(
            Program()
            .push_u256(123)
            .call_with_patches(HexBytes("0x" + "8b" * 20), self.TEMPLATE, [(4, 32)])
            ._emit(RETURN_TOP)
            .build()
        )
        assert not result.is_error

    def test_single_u256_returns_success(self):
        result = mini_evm(
            Program()
            .push_u256(10**18)
            .call_with_patches(HexBytes("0x" + "c4" * 20), self.TEMPLATE, [(4, 32)])
            ._emit(RETURN_TOP)
            .build()
        )
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 1

    def test_two_patches_returns_success(self):
        addr_val = int.from_bytes(HexBytes("0x" + "ab" * 20), "big")
        result = mini_evm(
            Program()
            .push_u256(addr_val)
            .push_u256(999)
            .call_with_patches(HexBytes("0x" + "c5" * 20), self.TEMPLATE, [(4, 32), (36, 20)])
            ._emit(RETURN_TOP)
            .build()
        )
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 1


# ---------------------------------------------------------------------------
# Program.call_contract_abi
# ---------------------------------------------------------------------------


class TestCallContractAbi:
    def test_no_patch(self):
        # call_contract_abi already terminates after CALL; the program halts at the
        # STOP that build() inserts before the data section, so no extra terminator
        # is needed (and Program.build() refuses external `+` concat for programs
        # with data sections).
        result = mini_evm(Program().call_contract_abi(HexBytes("0x" + "9a" * 20), "function ping() external").build())
        assert not result.is_error

    def test_with_patch(self):
        result = mini_evm(
            Program()
            .push_u256(123)
            .call_contract_abi(HexBytes("0x" + "9b" * 20), "function set(uint256 value) external", Patch())
            ._emit(RETURN_TOP)
            .build()
        )
        assert not result.is_error


# ---------------------------------------------------------------------------
# Program.build() — concat protection for data-section programs
# ---------------------------------------------------------------------------


class TestBuildConcatProtection:
    """Programs containing data sections (push_bytes / call_contract / …) must
    refuse external `+` concatenation — splicing would invalidate the
    CODECOPY data-section offsets baked in by build()."""

    def test_no_data_returns_plain_bytes(self):
        # Programs without data sections behave as plain bytes — concat is fine.
        code = Program().push_u256(7).build()
        assert type(code) is bytes
        _ = code + RETURN_TOP  # must not raise

    def test_push_bytes_built_refuses_append(self):
        code = Program().push_bytes(b"abc")._emit(pop()).build()
        with pytest.raises(TypeError, match="data sections"):
            _ = code + RETURN_TOP

    def test_push_bytes_built_refuses_prepend(self):
        code = Program().push_bytes(b"abc")._emit(pop()).build()
        with pytest.raises(TypeError, match="data sections"):
            _ = b"\x60\x00" + code

    def test_call_contract_built_refuses_concat(self):
        code = Program().call_contract(HexBytes("0x" + "aa" * 20), b"\x12\x34").pop().build()
        with pytest.raises(TypeError, match="data sections"):
            _ = code + RETURN_TOP

    def test_two_built_programs_refuse_concat(self):
        a = Program().push_bytes(b"aa").pop().build()
        b = Program().push_bytes(b"bb").pop().build()
        with pytest.raises(TypeError, match="data sections"):
            _ = a + b
