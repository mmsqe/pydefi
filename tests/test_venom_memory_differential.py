"""Comprehensive memory-operation differential tests: Lean Venom vs a real EVM.

The Venom **memory model** is the linchpin of the formalization, so it is
validated *behaviorally* here before (and alongside) the formal proofs
(`EvmYul/Venom/MemInvariant.lean`): real Vyper contracts that compile —
``vyper --experimental-codegen`` — to a wide range of memory ops are driven
through both

  * the Lean ``venom_run`` oracle (the formal ``Mem`` semantics), and
  * titanoboa (a production EVM),

and their observable results are asserted equal. Memory ops exercised:
``mstore`` / ``mload`` (word round-trips, fixed & 2-D arrays, mutation, memory
expansion), ``mcopy`` (``slice`` / nested ``slice`` / dynamic arrays / structs),
``calldatacopy`` (argument staging, loops), and ``sha3`` over a memory span
(``keccak256(concat(...))``, ``abi_encode``, variable-length slices, nested
mapping slots). A match across this range is strong evidence the ``Mem`` model
the proofs reason over is the real memory semantics.

Each contract is a single external function (linear ``jnz`` dispatch). Skipped
unless ``venom_run`` is built (``VENOM_RUN_BIN`` or an EVMYulLean sibling).
"""

from __future__ import annotations

import pytest

from pydefi.venom.erc20_abi import selector, word
from pydefi.venom.lean_oracle import LeanVenom, compile_runtime_venom, find_bin

pytestmark = pytest.mark.skipif(
    find_bin() is None, reason="venom_run oracle not built (set VENOM_RUN_BIN)"
)


def _check_int(venom_path, sig, args, expected):
    """Drive both engines on `sig(args)` and assert they return the same int."""
    import boa

    c = boa.loads(_SRC[sig])
    lean = LeanVenom(venom_path)
    r = lean.call(selector(sig) + args)
    assert r.status == "return", f"venom_run: {r.status}"
    boa_val = getattr(c, sig.split("(")[0])(*expected.args)
    assert r.as_int == int(boa_val) == expected.value
    return r


class Exp:
    def __init__(self, args, value):
        self.args = args
        self.value = value


# --- contracts: one external function each, exercising a memory pattern -------

_SRC = {
    # mload/mstore: a fixed memory array, written then indexed.
    "pick(uint256)": """
@external
def pick(i: uint256) -> uint256:
    a: uint256[4] = [11, 22, 33, 44]
    return a[i % 4]
""",
    # mstore/mload + mutation: write an element, read it back.
    "mut(uint256,uint256)": """
@external
def mut(i: uint256, v: uint256) -> uint256:
    a: uint256[3] = [10, 20, 30]
    a[i % 3] = v
    return a[i % 3]
""",
    # memory expansion: a 16-word array (forces growth).
    "big(uint256)": """
@external
def big(i: uint256) -> uint256:
    a: uint256[16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    return a[i % 16]
""",
    # 2-D fixed array: nested mload/mstore addressing.
    "grid(uint256,uint256)": """
@external
def grid(i: uint256, j: uint256) -> uint256:
    a: uint256[2][2] = [[1, 2], [3, 4]]
    return a[i % 2][j % 2]
""",
    # mcopy: dynamic array, summed in a loop.
    "dsum(uint256)": """
@external
def dsum(i: uint256) -> uint256:
    a: DynArray[uint256, 8] = [1, 2, 3, 4, 5, 6, 7, 8]
    s: uint256 = 0
    for k: uint256 in range(8):
        s += a[k]
    return s + i * 0
""",
    # calldatacopy + loop: build a memory array from a loop, index it.
    "sq(uint256)": """
@external
def sq(n: uint256) -> uint256:
    a: uint256[5] = [0, 0, 0, 0, 0]
    for i: uint256 in range(5):
        a[i] = i * i
    return a[n % 5]
""",
    # struct staged in memory (mcopy + calldatacopy).
    "padd(uint256,uint256)": """
struct P:
    x: uint256
    y: uint256
@external
def padd(a: uint256, b: uint256) -> uint256:
    p: P = P(x=a, y=b)
    return p.x + p.y
""",
    # mcopy via slice: 32-byte window out of an in-memory 64-byte buffer.
    "win(uint256,uint256)": """
@external
def win(x: uint256, off: uint256) -> uint256:
    b: Bytes[64] = concat(convert(x, bytes32), convert(x, bytes32))
    return convert(slice(b, off, 32), uint256)
""",
    # nested slice: slice of a slice.
    "win2(uint256)": """
@external
def win2(x: uint256) -> uint256:
    b: Bytes[128] = concat(convert(x, bytes32), convert(x + 1, bytes32),
                           convert(x + 2, bytes32), convert(x + 3, bytes32))
    c2: Bytes[64] = slice(b, 32, 64)
    return convert(slice(c2, 32, 32), uint256)
""",
    # dynamic array append (length-prefixed memory growth).
    "app(uint256)": """
@external
def app(x: uint256) -> uint256:
    a: DynArray[uint256, 4] = []
    a.append(x)
    a.append(x + 1)
    return a[1]
""",
    # single-byte slice out of a word (byte-granular memory read).
    "byt(uint256,uint256)": """
@external
def byt(x: uint256, i: uint256) -> uint256:
    b: bytes32 = convert(x, bytes32)
    return convert(slice(b, i % 32, 1), uint256)
""",
    # empty-bytes concat edge case (zero-length copy).
    "emp(uint256)": """
@external
def emp(x: uint256) -> uint256:
    b: Bytes[64] = concat(b"", convert(x, bytes32))
    return convert(slice(b, 0, 32), uint256)
""",
}

# Out of scope (documented boundary): `returndatacopy` requires message calls
# (`raw_call`/external calls), which `venom_run` models as unsupported — such
# programs step to `stuck` rather than run, so they are not differential-tested.

# keccak-over-memory contracts return bytes32 — compared separately.
_SRC_HASH = {
    "hash2(uint256,uint256)": """
@external
def hash2(a: uint256, b: uint256) -> bytes32:
    return keccak256(concat(convert(a, bytes32), convert(b, bytes32)))
""",
    "habi(uint256,uint256)": """
@external
def habi(a: uint256, b: uint256) -> bytes32:
    return keccak256(abi_encode(a, b))
""",
    "hdyn(uint256,uint256)": """
@external
def hdyn(x: uint256, n: uint256) -> bytes32:
    b: Bytes[96] = concat(convert(x, bytes32), convert(x, bytes32), convert(x, bytes32))
    return keccak256(slice(b, 0, 32 + (n % 3) * 32))
""",
}


# --- int-returning memory ops -------------------------------------------------


@pytest.mark.parametrize(
    "sig,args,exp",
    [
        ("pick(uint256)", word(0), Exp((0,), 11)),
        ("pick(uint256)", word(3), Exp((3,), 44)),
        ("pick(uint256)", word(9), Exp((9,), 22)),
        ("mut(uint256,uint256)", word(2) + word(99), Exp((2, 99), 99)),
        ("mut(uint256,uint256)", word(0) + word(7), Exp((0, 7), 7)),
        ("big(uint256)", word(15), Exp((15,), 15)),
        ("big(uint256)", word(0), Exp((0,), 0)),
        ("grid(uint256,uint256)", word(1) + word(0), Exp((1, 0), 3)),
        ("grid(uint256,uint256)", word(0) + word(1), Exp((0, 1), 2)),
        ("dsum(uint256)", word(0), Exp((0,), 36)),
        ("sq(uint256)", word(3), Exp((3,), 9)),
        ("sq(uint256)", word(4), Exp((4,), 16)),
        ("padd(uint256,uint256)", word(7) + word(11), Exp((7, 11), 18)),
        ("win(uint256,uint256)", word(0x1122334455667788) + word(0), Exp((0x1122334455667788, 0), 0x1122334455667788)),
        ("win2(uint256)", word(100), Exp((100,), 102)),
        ("app(uint256)", word(5), Exp((5,), 6)),
        ("app(uint256)", word(2**200), Exp((2**200,), 2**200 + 1)),
        ("byt(uint256,uint256)", word(0x1122334455667788) + word(31), Exp((0x1122334455667788, 31), 0x88)),
        ("byt(uint256,uint256)", word(0x1122334455667788) + word(24), Exp((0x1122334455667788, 24), 0x11)),
        ("emp(uint256)", word(0xDEAD), Exp((0xDEAD,), 0xDEAD)),
    ],
)
def test_memory_op_int_differential(tmp_path, sig, args, exp):
    venom_path = compile_runtime_venom(_SRC[sig], str(tmp_path / "m.venom"))
    _check_int(venom_path, sig, args, exp)


# --- keccak-over-memory ops (bytes32) -----------------------------------------


@pytest.mark.parametrize(
    "sig,a,b",
    [
        ("hash2(uint256,uint256)", 0x1234, 0xABCD),
        ("hash2(uint256,uint256)", 0, 2**256 - 1),
        ("habi(uint256,uint256)", 3, 5),
        ("habi(uint256,uint256)", 2**255, 1),
        ("hdyn(uint256,uint256)", 0xBEEF, 2),
        ("hdyn(uint256,uint256)", 0xBEEF, 0),
    ],
)
def test_keccak_over_memory_differential(tmp_path, sig, a, b):
    import boa

    venom_path = compile_runtime_venom(_SRC_HASH[sig], str(tmp_path / "h.venom"))
    c = boa.loads(_SRC_HASH[sig])
    lean = LeanVenom(venom_path)
    r = lean.call(selector(sig) + word(a) + word(b))
    assert r.status == "return"
    assert r.ret == getattr(c, sig.split("(")[0])(a, b)  # 32-byte digests agree
