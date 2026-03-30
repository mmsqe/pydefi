from __future__ import annotations

import pytest

from pydefi.vm.builder import DeFiBuilder
from pydefi.vm.planner import apply_action_graph


class _DummyVM:
    register_count = 16


def _addr(byte_hex: str) -> str:
    return "0x" + byte_hex * 20


def _u256(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _new_builder() -> DeFiBuilder:
    return DeFiBuilder(_DummyVM()).from_token(_addr("11"), amount_in=5).to(_addr("22"))


def _assert_call_with_surgery(bp: object, *, expected_target_hex: str, expected_patch_offset: int) -> None:
    assert [cmd.opcode for cmd in bp.commands] == ["CALLDATA_BUILD", "CALLDATA_SURGERY", "CALL"]

    surgery = bp.commands[1]
    assert surgery.data[0] == 1
    assert int.from_bytes(surgery.data[1:3], "big") == expected_patch_offset
    assert bp.commands[-1].data[:20].hex() == expected_target_hex * 20


@pytest.mark.parametrize(
    ("action", "expected_target_hex", "expected_patch_offset"),
    [
        pytest.param(
            {
                "kind": "swap",
                "target": _addr("33"),
                "token_out": _addr("44"),
                "calldata": bytes.fromhex("deadbeef") + _u256(5),
                "amount_placeholder": 5,
                "auto_amount_from_prev_call": False,
            },
            "33",
            4,
            id="swap-infers-patch-offset-from-placeholder",
        ),
        pytest.param(
            {
                "type": "deposit",
                "protocol": _addr("77"),
                "calldata": bytes.fromhex("cafebabe") + _u256(5),
                "patch_offset": "4",
                "auto_amount_from_prev_call": False,
            },
            "77",
            4,
            id="deposit-normalizes-kind-alias-and-string-patch-offset",
        ),
    ],
)
def test_apply_action_graph_call_actions_patch_amount(
    action: dict[str, object],
    expected_target_hex: str,
    expected_patch_offset: int,
) -> None:
    builder = _new_builder()

    apply_action_graph(builder, actions=[action])

    bp = builder.build()
    _assert_call_with_surgery(
        bp,
        expected_target_hex=expected_target_hex,
        expected_patch_offset=expected_patch_offset,
    )


@pytest.mark.parametrize(
    ("action", "expected_opcode", "expected_data_as_int"),
    [
        pytest.param(
            {
                "kind": "extract",
                "extraction_mode": "ret_u256",
                "extraction_offset": "32",
            },
            "RET_U256",
            32,
            id="extract-ret-u256",
        ),
        pytest.param(
            {
                "kind": "extract",
                "extraction_mode": "ret_last32",
            },
            "RET_LAST32",
            None,
            id="extract-ret-last32",
        ),
    ],
)
def test_apply_action_graph_supports_extract_actions(
    action: dict[str, object],
    expected_opcode: str,
    expected_data_as_int: int | None,
) -> None:
    builder = _new_builder()

    apply_action_graph(builder, actions=[action])

    bp = builder.build()
    assert len(bp.commands) == 1
    assert bp.commands[0].opcode == expected_opcode
    if expected_data_as_int is not None:
        assert int.from_bytes(bp.commands[0].data, "big") == expected_data_as_int


def test_apply_action_graph_enforces_bridge_amount_sanity() -> None:
    builder = _new_builder()

    selector = bytes.fromhex("12345678")
    bridge_calldata = selector + (b"\x00" * 64)

    with pytest.raises(ValueError, match="minAmountLD"):
        apply_action_graph(
            builder,
            actions=[
                {
                    "kind": "bridge",
                    "target": _addr("55"),
                    "dst_chain": 30110,
                    "dst_token": _addr("66"),
                    "calldata": bridge_calldata,
                }
            ],
            enforce_chain_token_consistency=False,
            bridge_amount_decoders={selector: lambda _calldata: (100, 101)},
        )
