"""Benchmark: PUSH32/MSTORE calldata encoding vs Venom CODECOPY data-section path.

Compares gas and bytecode size for ``call_contract`` across a range of calldata
sizes (4 – 2048 bytes).  When Venom is unavailable both paths fall back to the
same PUSH32/MSTORE chain so the comparison degenerates to equal cost.

Run with:
    python3 tests/bench_call_contract.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexbytes import HexBytes

from pydefi.vm import Program
from pydefi.vm.program import call, gas_opcode, push_addr, push_bytes, push_u256
from tests.conftest import mini_evm

_TARGET: HexBytes = HexBytes("0x" + "de" * 20)

# Single POP byte — discard the CALL success flag so both paths terminate cleanly.
_POP = bytes([0x50])


def _old(calldata: bytes) -> bytes:
    """Inline PUSH32/MSTORE calldata chain (old manual builder path).

    Uses the functional ``push_bytes`` helper which emits one PUSH32 opcode per
    32-byte chunk.  Terminates with POP+STOP so gas is comparable with ``_new``.
    """
    return (
        push_u256(0)  # retSize
        + push_u256(0)  # retOffset
        + push_bytes(calldata)  # PUSH32/MSTORE chain → [argsOffset, argsLen]
        + push_u256(0)  # value
        + push_addr(_TARGET)
        + gas_opcode()
        + call(require_success=False)
        + _POP  # discard success flag → implicit STOP
    )


def _new(calldata: bytes) -> bytes:
    """CODECOPY data-section path (Venom builder, matches Solidity pattern).

    When Venom is available, ``Program().call_contract()`` compiles via the
    Venom IR pipeline and stores calldata in a readonly data section, loading
    it at runtime with a single CODECOPY instruction before the CALL.
    When Venom is unavailable the program falls back to the same PUSH32/MSTORE
    path as ``_old`` so the comparison is still valid (both have identical cost).
    """
    # .pop() after call_contract: when a Venom plan is pending it sets
    # drop_result=True (compile_venom_call_contract_probe return_success=False →
    # return(0,0)), matching the STOP semantics of _old().  When there is no
    # Venom plan it emits a literal POP, identical to _old().
    return Program().call_contract(_TARGET, calldata, require_success=False).pop().build()


SIZES = [4, 32, 64, 128, 256, 512, 1024, 2048]


def run() -> None:
    print(
        f"{'size':>6}  {'old_gas':>8}  {'new_gas':>8}  {'saved':>8}  {'pct':>6}  {'old_bytes':>10}  {'new_bytes':>10}"
    )
    print("-" * 80)
    for size in SIZES:
        data = bytes(i % 256 for i in range(size))

        old_code = _old(data)
        new_code = _new(data)

        old_r = mini_evm(old_code)
        new_r = mini_evm(new_code)

        assert not old_r.is_error, f"old path failed at size={size}: {old_r.output.hex()}"
        assert not new_r.is_error, f"new path failed at size={size}: {new_r.output.hex()}"

        saved = old_r.gas_used - new_r.gas_used
        pct = saved / old_r.gas_used * 100 if old_r.gas_used else 0.0
        print(
            f"{size:>6}  {old_r.gas_used:>8}  {new_r.gas_used:>8}"
            f"  {saved:>+8}  {pct:>5.1f}%"
            f"  {len(old_code):>10}  {len(new_code):>10}"
        )


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


def test_bench_call_contract_correctness():
    """Both encodings execute without error."""
    data = bytes(range(68))  # 4-byte selector + 64-byte args
    old_r = mini_evm(_old(data))
    new_r = mini_evm(_new(data))
    assert not old_r.is_error
    assert not new_r.is_error


def test_bench_call_contract_new_cheaper_for_large_calldata():
    """CODECOPY path uses less gas than PUSH32/MSTORE for calldata ≥ 64 bytes."""
    data = bytes(range(128))
    old_r = mini_evm(_old(data))
    new_r = mini_evm(_new(data))
    assert new_r.gas_used < old_r.gas_used, (
        f"expected new ({new_r.gas_used}) < old ({old_r.gas_used}) for 128-byte calldata"
    )


def test_bench_call_contract_print_table(capsys):
    """Print the full comparison table when run with -s."""
    run()
    captured = capsys.readouterr()
    # Verify the table was produced for every size.
    for size in SIZES:
        assert str(size) in captured.out


if __name__ == "__main__":
    run()
