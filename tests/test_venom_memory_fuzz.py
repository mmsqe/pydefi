"""Large-scale property-based differential fuzz of the Venom **Memory Invariant
Layer** — Lean ``venom_run`` oracle vs a production EVM (titanoboa).

The formal memory model (``EvmYul/Venom/MemInvariant.lean``) proves the algebraic
laws the optimization proofs reason over:

  * read-after-write           (``loadWord_storeWord_self'``)
  * frame / no-aliasing        (``loadWord_storeWord_disjoint``)
  * mcopy correctness          (``readBytes_writeBytes_self``)
  * byte store (mstore8)       (``getByte_storeByte_self`` / ``_ne``)

This file *behaviourally* validates those laws at scale: each law is exercised by
a real Vyper contract (``vyper --experimental-codegen``) driven on **many random
inputs** — edge values (``0``, ``2**256-1``, powers of two, single bytes,
byte-pattern words) plus uniform 256-bit randoms — through both engines, asserting
the observable result is identical. A match across thousands of cases is strong
evidence the ``Mem`` model the proofs reason over *is* the real memory semantics.

Each contract is pure (no storage) and isolates one memory access pattern: word
round-trips at varied addresses, disjoint writes, overwrite, 2-D / dynamic arrays,
byte-granular reads, sub-word windows crossing word boundaries (mcopy), memory
expansion, and keccak over a memory span.

``VENOM_FUZZ_N`` sets the per-pattern case count (default 30 for CI). Run the
campaign standalone for a big sweep + summary table::

    VENOM_RUN_BIN=.../venom_run python tests/test_venom_memory_fuzz.py 150

Skipped unless ``venom_run`` is built (``VENOM_RUN_BIN`` or an EVMYulLean sibling).
"""

from __future__ import annotations

import os
import random

import pytest

from pydefi.venom.erc20_abi import selector, word
from pydefi.venom.lean_oracle import LeanVenom, compile_runtime_venom, find_bin

pytestmark = pytest.mark.skipif(find_bin() is None, reason="venom_run oracle not built (set VENOM_RUN_BIN)")


# --- value generator: edge-biased over the full uint256 range -----------------

_EDGE = [
    0,
    1,
    2,
    3,
    7,
    255,
    256,
    257,
    0xFF,
    0xFF00,
    0xFFFF,
    2**8,
    2**16,
    2**32,
    2**64,
    2**128,
    2**160,
    2**192,
    2**248,
    2**255,
    2**256 - 1,
    2**256 - 256,
    2**256 - 2**128,
    (0xFF << 248),
    (1 << 255) + 1,
    0x0101010101010101010101010101010101010101010101010101010101010101,
    0xDEADBEEFCAFEBABE1122334455667788990011223344556677889900AABBCCDD,
]


def _gen_u256(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.35:
        return rng.choice(_EDGE)
    if r < 0.55:
        return 1 << rng.randint(0, 255)  # power of two (word/byte alignment)
    if r < 0.70:
        return rng.randint(0, 511)  # small (indices, low bytes)
    if r < 0.82:
        return (1 << rng.randint(1, 256)) - 1  # low-bit masks
    return rng.getrandbits(256)  # uniform 256-bit


# --- the memory-law contracts (one pure pattern each) -------------------------

# (sig, return-kind, vyper source). All args are uint256; the contract reduces
# index args mod the array size, so every random input is in range.
PATTERNS: list[tuple[str, str, str]] = [
    # read-after-write: store a word at a chosen address, read it back.
    (
        "raw_rw(uint256,uint256)",
        "int",
        """
@external
def raw_rw(idx: uint256, val: uint256) -> uint256:
    a: uint256[32] = empty(uint256[32])
    a[idx % 32] = val
    return a[idx % 32]
""",
    ),
    # no-aliasing / overwrite: two writes, read the first (independent unless i==j).
    (
        "noalias(uint256,uint256,uint256,uint256)",
        "int",
        """
@external
def noalias(i: uint256, j: uint256, vi: uint256, vj: uint256) -> uint256:
    a: uint256[16] = empty(uint256[16])
    a[i % 16] = vi
    a[j % 16] = vj
    return a[i % 16]
""",
    ),
    # frame: bump neighbours, the middle word must be untouched.
    (
        "frame3(uint256,uint256,uint256)",
        "int",
        """
@external
def frame3(v0: uint256, v1: uint256, v2: uint256) -> uint256:
    a: uint256[3] = [v0, v1, v2]
    a[0] = unsafe_add(a[0], 1)
    a[2] = unsafe_add(a[2], 1)
    return a[1]
""",
    ),
    # byte-granular read (mstore8 / getByte): the i-th byte of a word.
    (
        "byte_rw(uint256,uint256)",
        "int",
        """
@external
def byte_rw(x: uint256, i: uint256) -> uint256:
    b: bytes32 = convert(x, bytes32)
    return convert(slice(b, i % 32, 1), uint256)
""",
    ),
    # 32-byte window at a variable offset out of a 64-byte buffer (mcopy crossing
    # a word boundary — the byte-level frame law).
    (
        "win32(uint256,uint256,uint256)",
        "int",
        """
@external
def win32(x: uint256, y: uint256, off: uint256) -> uint256:
    b: Bytes[64] = concat(convert(x, bytes32), convert(y, bytes32))
    return convert(slice(b, off % 33, 32), uint256)
""",
    ),
    # memory expansion: a chosen address in a large array forces growth.
    (
        "expand(uint256,uint256)",
        "int",
        """
@external
def expand(idx: uint256, val: uint256) -> uint256:
    a: uint256[48] = empty(uint256[48])
    a[idx % 48] = val
    return a[idx % 48]
""",
    ),
    # 2-D fixed array: nested mload/mstore addressing.
    (
        "twod(uint256,uint256,uint256)",
        "int",
        """
@external
def twod(i: uint256, j: uint256, v: uint256) -> uint256:
    a: uint256[4][4] = empty(uint256[4][4])
    a[i % 4][j % 4] = v
    return a[i % 4][j % 4]
""",
    ),
    # dynamic array append + index (length-prefixed memory growth).
    (
        "dynarr(uint256,uint256,uint256,uint256)",
        "int",
        """
@external
def dynarr(a0: uint256, a1: uint256, a2: uint256, k: uint256) -> uint256:
    arr: DynArray[uint256, 8] = []
    arr.append(a0)
    arr.append(a1)
    arr.append(a2)
    return arr[k % 3]
""",
    ),
    # mcopy: copy a dynamic array, index the copy.
    (
        "mcopy_arr(uint256,uint256,uint256,uint256,uint256)",
        "int",
        """
@external
def mcopy_arr(a0: uint256, a1: uint256, a2: uint256, a3: uint256, k: uint256) -> uint256:
    src: DynArray[uint256, 4] = [a0, a1, a2, a3]
    dst: DynArray[uint256, 4] = src
    return dst[k % 4]
""",
    ),
    # many writes then one read (multiple storeWords, frame across all of them).
    (
        "rwseq(uint256,uint256)",
        "int",
        """
@external
def rwseq(s: uint256, k: uint256) -> uint256:
    a: uint256[8] = empty(uint256[8])
    for i: uint256 in range(8):
        a[i] = unsafe_add(s, i)
    return a[k % 8]
""",
    ),
    # swap two elements (read+write interplay at two addresses).
    (
        "swap2(uint256,uint256,uint256,uint256)",
        "int",
        """
@external
def swap2(i: uint256, j: uint256, v0: uint256, v1: uint256) -> uint256:
    a: uint256[8] = empty(uint256[8])
    a[i % 8] = v0
    a[j % 8] = v1
    t: uint256 = a[i % 8]
    a[i % 8] = a[j % 8]
    a[j % 8] = t
    return a[i % 8]
""",
    ),
    # abi_encode then keccak (mcopy + calldatacopy staging) — bytes32 digest.
    (
        "abienc(uint256,uint256)",
        "bytes",
        """
@external
def abienc(a: uint256, b: uint256) -> bytes32:
    return keccak256(abi_encode(a, b))
""",
    ),
    # keccak over a fixed memory span (sha3) — bytes32 digest.
    (
        "kec2(uint256,uint256)",
        "bytes",
        """
@external
def kec2(a: uint256, b: uint256) -> bytes32:
    return keccak256(concat(convert(a, bytes32), convert(b, bytes32)))
""",
    ),
    # keccak over a variable-length memory span.
    (
        "kecdyn(uint256,uint256)",
        "bytes",
        """
@external
def kecdyn(x: uint256, n: uint256) -> bytes32:
    b: Bytes[96] = concat(convert(x, bytes32), convert(x, bytes32), convert(x, bytes32))
    return keccak256(slice(b, 0, 32 + (n % 3) * 32))
""",
    ),
]


def _nargs(sig: str) -> int:
    return sig.count("uint256")


def _drive(sig: str, ret_kind: str, src: str, n: int, seed: int, tmp_path) -> list:
    """Drive `n` random inputs through both engines; return the list of
    (args, lean, evm) mismatches (empty when the law holds)."""
    import boa

    rng = random.Random(seed)
    venom_path = compile_runtime_venom(src, str(tmp_path / "fuzz.venom"))
    c = boa.loads(src)
    lean = LeanVenom(venom_path)
    fn = sig.split("(")[0]
    k = _nargs(sig)
    mismatches = []
    for _ in range(n):
        args = tuple(_gen_u256(rng) for _ in range(k))
        r = lean.call(selector(sig) + b"".join(word(a) for a in args))
        if r.status != "return":
            mismatches.append((args, f"status={r.status}", "return"))
            continue
        evm = getattr(c, fn)(*args)
        if ret_kind == "int":
            lean_val = r.as_int
            if lean_val != int(evm):
                mismatches.append((args, lean_val, int(evm)))
        else:
            if r.ret != evm:
                mismatches.append((args, r.ret.hex(), evm.hex()))
    return mismatches


@pytest.mark.parametrize("sig,ret_kind,src", PATTERNS, ids=[p[0].split("(")[0] for p in PATTERNS])
def test_memory_law_fuzz(tmp_path, sig, ret_kind, src):
    n = int(os.environ.get("VENOM_FUZZ_N", "30"))
    seed = 0x5EED ^ hash(sig) & 0xFFFFFFFF
    mismatches = _drive(sig, ret_kind, src, n, seed, tmp_path)
    assert not mismatches, f"{sig}: {len(mismatches)}/{n} mismatches; first={mismatches[0]}"


# --- standalone campaign runner (prints a summary table) ----------------------

if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    if find_bin() is None:
        print("venom_run not built (set VENOM_RUN_BIN)")
        sys.exit(1)

    total = 0
    total_bad = 0
    print(f"\n  Memory Invariant Layer — differential fuzz ({N} cases/pattern)\n")
    print(f"  {'pattern':<12} {'cases':>6} {'pass':>6} {'fail':>5}")
    print("  " + "-" * 33)
    with tempfile.TemporaryDirectory() as d:
        for sig, ret_kind, src in PATTERNS:
            seed = 0x5EED ^ hash(sig) & 0xFFFFFFFF
            bad = _drive(sig, ret_kind, src, N, seed, Path(d))
            total += N
            total_bad += len(bad)
            name = sig.split("(")[0]
            print(f"  {name:<12} {N:>6} {N - len(bad):>6} {len(bad):>5}" + ("" if not bad else f"   !! {bad[0]}"))
    print("  " + "-" * 33)
    print(f"  {'TOTAL':<12} {total:>6} {total - total_bad:>6} {total_bad:>5}\n")
    sys.exit(1 if total_bad else 0)
