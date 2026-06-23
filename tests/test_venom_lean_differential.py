"""Whole-transaction differential tests: Lean Venom semantics vs a real EVM.

Compiles a Vyper contract to its runtime ``.venom`` with
``--experimental-codegen``, then drives the *same* transaction sequence through

  * the formal Lean Venom interpreter (the EVMYulLean ``venom_run`` oracle,
    via :class:`pydefi.venom.lean_oracle.LeanVenom`, threading storage), and
  * titanoboa (a production EVM),

asserting their observable results agree. This validates the formal Venom IR
semantics — including the balance-slot ``keccak256(slot ++ addr)`` derivation
the ``~addr`` peephole rewrites — against a real EVM on real compiler output,
end to end.

Coverage exercised here: selector dispatch, calldata decoding, single and
*nested* HashMap keccak slots, ``msg.sender`` (`caller`), event ``log`` (a
no-op in the model — log data is not compared), safe-math reverts with state
rollback, and ``bool`` return values.

Limitation: Vyper switches from linear (`jnz`) selector dispatch to a dense
jump table (`codecopy`/`offset`/`djmp`) at 4+ external selectors, and that
table encodes label offsets only resolved at assembly time (absent from the
`.venom` text). So these contracts are kept at <=3 selectors; larger ones step
to ``stuck`` rather than run.

Skipped unless the ``venom_run`` oracle is built (set ``VENOM_RUN_BIN`` or have
an EVMYulLean sibling checkout with ``lake exe venom_run``).
"""

from __future__ import annotations

import pytest

from pydefi.venom.erc20_abi import addr_str, selector, word
from pydefi.venom.lean_oracle import LeanVenom, compile_runtime_venom, find_bin

pytestmark = pytest.mark.skipif(
    find_bin() is None, reason="venom_run oracle not built (set VENOM_RUN_BIN)"
)

# --- helpers ---------------------------------------------------------------

A = 0x00000000000000000000000000000000000000AA
B = 0x00000000000000000000000000000000000000BB


# --- contracts (<=3 selectors so dispatch stays linear) --------------------

SRC_HASHMAP = """
balances: public(HashMap[address, uint256])

@external
def setbal(a: address, v: uint256):
    self.balances[a] = v

@external
def getbal(a: address) -> uint256:
    return self.balances[a]
"""

# mint / transfer / balanceOf(getter): events, msg.sender, safe-math revert.
SRC_BAL = """
balanceOf: public(HashMap[address, uint256])

event Transfer:
    sender: indexed(address)
    receiver: indexed(address)
    amount: uint256

@external
def mint(to: address, amount: uint256):
    self.balanceOf[to] += amount
    log Transfer(sender=empty(address), receiver=to, amount=amount)

@external
def transfer(to: address, amount: uint256) -> bool:
    self.balanceOf[msg.sender] -= amount
    self.balanceOf[to] += amount
    log Transfer(sender=msg.sender, receiver=to, amount=amount)
    return True
"""

# approve / allowance(getter): nested HashMap => double keccak slot.
SRC_ALLOW = """
allowance: public(HashMap[address, HashMap[address, uint256]])

event Approval:
    owner: indexed(address)
    spender: indexed(address)
    amount: uint256

@external
def approve(spender: address, amount: uint256) -> bool:
    self.allowance[msg.sender][spender] = amount
    log Approval(owner=msg.sender, spender=spender, amount=amount)
    return True
"""


# --- tests -----------------------------------------------------------------


def test_hashmap_balance_whole_tx_differential(tmp_path):
    import boa

    venom_path = compile_runtime_venom(SRC_HASHMAP, tmp_path / "runtime.venom")
    contract = boa.loads(SRC_HASHMAP)
    lean = LeanVenom(venom_path)

    setbal = selector("setbal(address,uint256)")
    getbal = selector("getbal(address)")

    # low-address probes (0x02, 0x04) cover the named-slot collisions the
    # NOT(addr) scheme targets.
    for addr, val in [(0x1234, 99), (0x02, 1), (0x04, 7), (0xDEADBEEF, 2**200 - 1)]:
        contract.setbal(addr_str(addr), val)
        assert lean.call(setbal + word(addr) + word(val)).status == "stop"

        boa_bal = contract.getbal(addr_str(addr))
        r = lean.call(getbal + word(addr))
        assert r.status == "return"
        assert r.as_int == boa_bal == val


def test_erc20_mint_transfer_differential(tmp_path):
    """mint + transfer with events, msg.sender, return bool, and a
    safe-math revert (overdraw) that rolls back identically."""
    import boa

    venom_path = compile_runtime_venom(SRC_BAL, tmp_path / "bal.venom")
    c = boa.loads(SRC_BAL)
    lean = LeanVenom(venom_path)

    mint = selector("mint(address,uint256)")
    transfer = selector("transfer(address,uint256)")
    balanceOf = selector("balanceOf(address)")

    def both_balance(addr):
        boa_bal = c.balanceOf(addr_str(addr))
        r = lean.call(balanceOf + word(addr))
        assert r.status == "return"
        assert r.as_int == boa_bal
        return boa_bal

    # mint 1000 to A
    c.mint(addr_str(A), 1000)
    assert lean.call(mint + word(A) + word(1000)).status == "stop"
    assert both_balance(A) == 1000

    # transfer A -> B, 300, as A (msg.sender)
    c.transfer(addr_str(B), 300, sender=addr_str(A))
    t = lean.call(transfer + word(B) + word(300), caller=A)
    assert t.status == "return" and t.as_int == 1  # returns True
    assert both_balance(A) == 700
    assert both_balance(B) == 300

    # overdraw: A transfers more than its balance -> revert on both, no change
    with pytest.raises(Exception):
        c.transfer(addr_str(B), 10**30, sender=addr_str(A))
    r = lean.call(transfer + word(B) + word(10**30), caller=A)
    assert r.status == "revert"
    assert both_balance(A) == 700  # rolled back
    assert both_balance(B) == 300


def test_erc20_allowance_nested_keccak_differential(tmp_path):
    """approve writes a *nested* HashMap slot (double keccak); both engines
    derive the same slot and read back the same allowance."""
    import boa

    venom_path = compile_runtime_venom(SRC_ALLOW, tmp_path / "allow.venom")
    c = boa.loads(SRC_ALLOW)
    lean = LeanVenom(venom_path)

    approve = selector("approve(address,uint256)")
    allowance = selector("allowance(address,address)")

    # A approves B for 555
    c.approve(addr_str(B), 555, sender=addr_str(A))
    r = lean.call(approve + word(B) + word(555), caller=A)
    assert r.status == "return" and r.as_int == 1

    boa_allow = c.allowance(addr_str(A), addr_str(B))
    g = lean.call(allowance + word(A) + word(B))
    assert g.status == "return"
    assert g.as_int == boa_allow == 555


# triple-nested HashMap => triple keccak slot (deepest memory keccak staging).
SRC_TRIPLE = """
m: public(HashMap[address, HashMap[address, HashMap[address, uint256]]])

@external
def setm(a: address, b: address, cc: address, v: uint256):
    self.m[a][b][cc] = v
"""


def test_triple_nested_mapping_keccak_differential(tmp_path):
    """A *triple*-nested HashMap write derives its slot via three chained
    keccak256(slot ++ key) stagings in memory; both engines compute the same
    slot and read back the same value."""
    import boa

    venom_path = compile_runtime_venom(SRC_TRIPLE, tmp_path / "triple.venom")
    c = boa.loads(SRC_TRIPLE)
    lean = LeanVenom(venom_path)

    setm = selector("setm(address,address,address,uint256)")
    getm = selector("m(address,address,address)")
    C = 0x00000000000000000000000000000000000000CC

    c.setm(addr_str(A), addr_str(B), addr_str(C), 1234)
    assert lean.call(setm + word(A) + word(B) + word(C) + word(1234)).status == "stop"

    boa_v = c.m(addr_str(A), addr_str(B), addr_str(C))
    g = lean.call(getm + word(A) + word(B) + word(C))
    assert g.status == "return"
    assert g.as_int == boa_v == 1234
