"""Bridge to the EVMYulLean ``venom_run`` reference oracle.

Runs message calls against a ``.venom`` program with the *formal Lean Venom
semantics*, as a subprocess speaking JSON, with persistent storage threaded
across calls. Used to differentially test the formal semantics against a
production EVM (titanoboa / anvil) on real ``vyper --experimental-codegen``
output — see ``tests/test_venom_lean_differential.py``.

The oracle binary is the EVMYulLean ``venom_run`` executable
(``lake exe venom_run``). Point ``VENOM_RUN_BIN`` at it, or it is auto-located
in a sibling checkout. :func:`find_bin` returns ``None`` when unavailable so
callers can skip.

The Lean side computes ``keccak256``/``sha3`` with the real FFI hash, so
storage slots and return values are concrete and directly comparable to an EVM
(verified: the HashMap balance slot equals ``keccak(bytes32(slot) ++
bytes32(addr))`` bit-for-bit).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def find_bin() -> str | None:
    """Locate the ``venom_run`` oracle binary, or ``None``."""
    env = os.environ.get("VENOM_RUN_BIN")
    if env and Path(env).exists():
        return env
    for base in (
        Path.home() / "Documents" / "crypto" / "EVMYulLean",
        Path.home() / "src" / "EVMYulLean",
    ):
        cand = base / ".lake" / "build" / "bin" / "venom_run"
        if cand.exists():
            return str(cand)
    return None


def compile_runtime_venom(src: str, out_path: str | Path) -> str:
    """Compile Vyper `src` to its runtime ``.venom`` text (Venom backend) and
    write it to `out_path`. Returns the path."""
    from vyper.compiler.phases import CompilerData
    from vyper.compiler.settings import OptimizationLevel, Settings, anchor_settings

    cd = CompilerData(src, settings=Settings(experimental_codegen=True, optimize=OptimizationLevel.GAS))
    with anchor_settings(cd.settings):
        txt = str(cd.venom_runtime)
    Path(out_path).write_text(txt)
    return str(out_path)


@dataclass
class LeanResult:
    status: str  # return | revert | stop | invalid | selfdestruct | stuck | outoffuel
    ret: bytes
    storage: list  # list of [key_hex, val_hex] pairs (the full resulting state)

    @property
    def as_int(self) -> int:
        """The return data decoded as a big-endian uint256."""
        return int.from_bytes(self.ret, "big")


class LeanVenom:
    """A ``.venom`` runtime executed by the Lean oracle, with storage that
    persists across :meth:`call` (so a transaction *sequence* can be driven)."""

    def __init__(self, venom_path: str | Path, bin_path: str | None = None):
        self.venom_path = str(venom_path)
        self.bin = bin_path or find_bin()
        if self.bin is None:
            raise FileNotFoundError("venom_run oracle not found (set VENOM_RUN_BIN)")
        self.storage: list = []

    def call(self, calldata: bytes, caller: int = 0, callvalue: int = 0) -> LeanResult:
        spec = {
            "calldata": "0x" + calldata.hex(),
            "caller": hex(caller),
            "callvalue": str(callvalue),
            "storage": self.storage,
        }
        out = subprocess.run(
            [self.bin, self.venom_path],
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            check=True,
        )
        r = json.loads(out.stdout)
        self.storage = r["storage"]  # thread state forward
        rhex = r.get("return", "0x")
        ret = bytes.fromhex(rhex[2:] if rhex.startswith("0x") else rhex)
        return LeanResult(r["status"], ret, r["storage"])
