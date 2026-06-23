"""Corpus sweep: apply the Venom balance-slot peephole across several Vyper
contracts and check, for each, behavioural parity (original vs patched) and
the per-operation gas delta.

For every contract we compile with ``--experimental-codegen``, detect the
``HashMap[address, …]`` balance slot from the storage layout, run the patcher
at that slot, deploy the original and the patched runtime in titanoboa, drive
an identical op sequence on both, and assert their outputs match — then report
gas. This checks the patcher *generalises* beyond the single vendored Snekmate
ERC-20 it was tuned on: the corpus places the balance map at slot 0, at a
higher slot (scalars before it), and after another address-keyed map (slot
discrimination), plus the real Snekmate artifact when present.

Note: pydefi's own contracts are Solidity, so the corpus is drawn from
inline Vyper variants (+ the vendored Snekmate artifact). The peephole only
applies to single-level ``HashMap[address, uintN]`` slots; nested maps
(``allowance[a][b]``) are intentionally left unpatched.

Run with ``-s`` to see the gas table. Skipped if boa/vyper are unavailable.
"""

from __future__ import annotations

import pytest

boa = pytest.importorskip("boa")
vyper = pytest.importorskip("vyper")

from vyper.compiler.settings import OptimizationLevel, Settings  # noqa: E402

from pydefi.venom import balance_patch as P  # noqa: E402
from pydefi.venom.erc20_abi import (  # noqa: E402
    APPROVE,
    BALANCE_OF,
    MINT,
    TRANSFER,
    TRANSFER_FROM,
    addr20,
    arg_addr,
    word,
)

ONE = 10**18
A, B, C = 0xAA, 0xBB, 0xCC

# --- corpus ----------------------------------------------------------------

_ERC20_BODY = """
balanceOf: public(HashMap[address, uint256])
allowance: public(HashMap[address, HashMap[address, uint256]])

@external
def mint(to: address, amount: uint256):
    self.balanceOf[to] += amount

@external
def transfer(to: address, amount: uint256) -> bool:
    self.balanceOf[msg.sender] -= amount
    self.balanceOf[to] += amount
    return True

@external
def approve(spender: address, amount: uint256) -> bool:
    self.allowance[msg.sender][spender] = amount
    return True

@external
def transferFrom(owner: address, to: address, amount: uint256) -> bool:
    self.allowance[owner][msg.sender] -= amount
    self.balanceOf[owner] -= amount
    self.balanceOf[to] += amount
    return True
"""

# (name, preamble that shifts the balance slot)
INLINE_CORPUS = [
    ("erc20 @ slot0", ""),
    ("erc20 @ slot3 (scalars first)", "owner: public(address)\ntotalSupply: public(uint256)\npaused: public(bool)\n"),
    ("erc20 @ after addr-map", "minters: public(HashMap[address, bool])\n"),
]


def _compile_venom(src: str):
    out = vyper.compile_code(
        src,
        output_formats=["bytecode", "bytecode_runtime", "layout"],
        settings=Settings(experimental_codegen=True, optimize=OptimizationLevel.GAS),
    )
    creation = bytes.fromhex(out["bytecode"][2:])
    runtime = bytes.fromhex(out["bytecode_runtime"][2:])
    slot = int(out["layout"]["storage_layout"]["balanceOf"]["slot"])
    return creation, runtime, slot


def _op_sequence():
    """A uniform op sequence over the shared ERC-20 ABI.

    Each item: (label, calldata, sender). Run identically on orig and opt;
    state evolves the same way on both, so the per-op gas delta is exact."""
    return [
        ("mint", MINT + arg_addr(A) + word(1000 * ONE), None),
        ("mint (warm)", MINT + arg_addr(A) + word(ONE), None),
        ("approve", APPROVE + arg_addr(B) + word(5 * ONE), addr20(A)),
        ("transfer", TRANSFER + arg_addr(B) + word(3 * ONE), addr20(A)),
        ("transferFrom", TRANSFER_FROM + arg_addr(A) + arg_addr(C) + word(2 * ONE), addr20(B)),
        ("balanceOf", BALANCE_OF + arg_addr(A), None),
    ]


def _run_pair(creation: bytes, patched_creation: bytes, ops):
    orig, _ = boa.env.deploy_code(bytecode=creation)
    opt, _ = boa.env.deploy_code(bytecode=patched_creation)
    rows = []
    for label, data, sender in ops:
        ro = boa.env.raw_call(orig, data=data, sender=sender)
        rp = boa.env.raw_call(opt, data=data, sender=sender)
        assert ro.output == rp.output, f"output mismatch on {label}: {ro.output!r} vs {rp.output!r}"
        rows.append((label, ro.get_gas_used(), rp.get_gas_used()))
    return rows


def _print_table(name: str, slot: int, sites: int, length_ok: bool, rows):
    print(f"\n{name}: balanceOf @ slot {slot}, {sites} sites patched, length preserved={length_ok}")
    print(f"  {'op':<16}{'orig':>8}{'patched':>9}{'Δ gas':>8}")
    tot_o = tot_p = 0
    for label, o, p in rows:
        print(f"  {label:<16}{o:>8}{p:>9}{p - o:>+8}")
        tot_o += o
        tot_p += p
    print(f"  {'total':<16}{tot_o:>8}{tot_p:>9}{tot_p - tot_o:>+8}")


@pytest.mark.parametrize("name,preamble", INLINE_CORPUS, ids=[c[0] for c in INLINE_CORPUS])
def test_peephole_parity_and_gas_inline(name, preamble):
    creation, runtime, slot = _compile_venom(preamble + _ERC20_BODY)
    sites = P.count_sites(runtime, slot)
    assert sites > 0, f"{name}: patcher found no balance sites at slot {slot}"
    patched_creation = P.patch_creation(creation, runtime, slot)
    assert len(patched_creation) == len(creation), "patch must be length-preserving"

    rows = _run_pair(creation, patched_creation, _op_sequence())
    _print_table(name, slot, sites, len(patched_creation) == len(creation), rows)

    # every balance-touching op should cost no more after patching
    assert sum(p for _, _, p in rows) <= sum(o for _, o, _ in rows), "patched total gas exceeded original"


def test_peephole_parity_and_gas_snekmate():
    """The real vendored Snekmate ERC-20 (full contract, dense dispatch)."""
    if not P.DEFAULT_ARTIFACT.exists():
        pytest.skip("vendored Snekmate Venom artifact not present")
    creation = P.creation_from_artifact(P.DEFAULT_ARTIFACT)
    runtime = P.runtime_from_artifact(P.DEFAULT_ARTIFACT)
    slot = P.BALANCE_SLOT
    sites = P.count_sites(runtime, slot)
    assert sites > 0
    patched = P.patch(runtime)
    assert len(patched) == len(runtime)

    orig, _ = boa.env.deploy_code(bytecode=creation)
    opt, _ = boa.env.deploy_code(bytecode=creation)
    boa.env.set_code(opt, patched)
    dep = bytes(boa.env.eoa.canonical_address)

    rows = []
    # Snekmate mint requires the deployer (minter); call from the EOA.
    for label, data in [
        ("mint", MINT + arg_addr(dep) + word(1000 * ONE)),
        ("transfer", TRANSFER + arg_addr(B) + word(ONE)),
        ("balanceOf", BALANCE_OF + arg_addr(dep)),
    ]:
        ro = boa.env.raw_call(orig, data=data)
        rp = boa.env.raw_call(opt, data=data)
        assert ro.output == rp.output, f"Snekmate parity on {label}"
        rows.append((label, ro.get_gas_used(), rp.get_gas_used()))
    _print_table("snekmate erc20 (artifact)", slot, sites, len(patched) == len(runtime), rows)
    assert sum(p for _, _, p in rows) <= sum(o for _, o, _ in rows)
