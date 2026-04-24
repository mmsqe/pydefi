"""Tests for blob dedup and :class:`CalldataTemplate`.

Covers:
- ``Program`` reuses a single data section when identical calldata is passed
  to multiple ``call_contract`` invocations.
- ``Program.template`` pre-encodes a zero-filled blob and records per-param
  offsets; invoking it with different args produces different patches but
  the same blob.
- ``call_contract`` accepts a ``CalldataPayload`` transparently, merges
  template patches with any caller-supplied patches, and still executes
  correctly end-to-end.
"""

from __future__ import annotations

from hexbytes import HexBytes

from pydefi.vm import CalldataPayload, CalldataTemplate, Program
from tests.conftest import mini_evm

_TARGET = HexBytes("0x" + "aa" * 20)


def _build_and_run(prog: Program) -> None:
    bc = prog.build()
    result = mini_evm(bc)
    assert not result.is_error, f"unexpected revert: {result.output.hex()}"


class TestBlobDedup:
    def test_identical_calldata_shares_one_data_section(self):
        p = Program()
        data = b"\xaa" * 100
        p.call_contract(_TARGET, data)
        p.call_contract(_TARGET, data)
        p.call_contract(_TARGET, data)
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 1
        assert len(p._blob_labels) == 1

    def test_distinct_calldata_gets_distinct_sections(self):
        p = Program()
        p.call_contract(_TARGET, b"\x00" * 100)
        p.call_contract(_TARGET, b"\xff" * 100)
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 2

    def test_padding_equivalence_still_dedupes(self):
        # A 64-byte and a 65-byte blob have different padded forms, so they
        # do NOT share a section — this pins that assumption.
        p = Program()
        p.call_contract(_TARGET, b"\x11" * 64)
        p.call_contract(_TARGET, b"\x11" * 65)
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 2


class TestCalldataTemplate:
    def test_named_params_extracted_from_signature(self):
        tmpl = CalldataTemplate("transfer(address to, uint256 amount)")
        assert tmpl.signature == "transfer(address,uint256)"
        assert tmpl.param_names == ["to", "amount"]
        assert tmpl.param_types == ["address", "uint256"]

    def test_unnamed_params_positional_only(self):
        tmpl = CalldataTemplate("transfer(address,uint256)")
        assert tmpl.param_names == []

    def test_explicit_names_override_absence(self):
        tmpl = CalldataTemplate("transfer(address,uint256)", names=["recipient", "amt"])
        assert tmpl.param_names == ["recipient", "amt"]

    def test_blob_selector_prefix(self):
        # Static-only invocation encodes into the blob entirely; selector is
        # the first 4 bytes of whatever the template produces.
        tmpl = CalldataTemplate("transfer(address,uint256)")
        payload = tmpl(0, 0)
        assert payload.blob.startswith(bytes.fromhex("a9059cbb"))  # transfer selector

    def test_scalar_static_args_lift_to_patches_not_blob(self):
        # Scalar-static args (address, uint256) are lifted out of the blob
        # into MSTORE patches so the blob stays byte-identical across
        # invocations — enables Program._blob_labels dedup across rows that
        # differ only in scalar values.
        tmpl = CalldataTemplate("transfer(address,uint256)")
        p1 = tmpl(0xABAB, 100)
        p2 = tmpl(0xCDCD, 200)
        assert p1.blob == p2.blob, "scalar-static args must not affect the blob"
        # Selector + two zero head slots.
        assert p1.blob == bytes.fromhex("a9059cbb") + bytes(64)
        # Both args became patches.
        assert p1.patches == {4: 0xABAB, 36: 100}
        assert p2.patches == {4: 0xCDCD, 36: 200}

    def test_runtime_value_args_appear_as_patches(self):
        p = Program()
        amount = p.const(42)
        tmpl = p.template("transfer(address,uint256)")
        payload = tmpl(0xABAB, amount)
        # Both args are scalar-static so both are patches; recipient is a
        # literal int patch, amount is an SSA Value patch.
        assert set(payload.patches.keys()) == {4, 36}
        assert payload.patches[4] == 0xABAB
        assert payload.patches[36] is amount
        # Blob head is zero-filled (statics don't bake in).
        assert payload.blob[4:36] == bytes(32)

    def test_zero_literals_produce_no_patches(self):
        tmpl = CalldataTemplate("transfer(address,uint256)")
        payload = tmpl(0, 0)
        assert payload.patches == {}

    def test_positional_vs_keyword_produce_same_payload(self):
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        recipient = p.const(0x60BC267D1242494CA0BA2A944EC0BA223C68B0E2)
        amount = p.const(10000)
        by_pos = tmpl(recipient, amount)
        by_kw = tmpl(to=recipient, amount=amount)
        assert by_pos.blob == by_kw.blob
        assert by_pos.patches == by_kw.patches

    def test_unknown_kwarg_rejected(self):
        tmpl = CalldataTemplate("transfer(address to, uint256 amount)")
        try:
            tmpl(to=1, bogus=2)
        except TypeError as e:
            assert "unexpected keyword argument" in str(e)
        else:
            raise AssertionError("expected TypeError")

    def test_missing_kwarg_rejected(self):
        tmpl = CalldataTemplate("transfer(address to, uint256 amount)")
        try:
            tmpl(to=1)
        except TypeError as e:
            assert "missing keyword argument" in str(e)
        else:
            raise AssertionError("expected TypeError")

    def test_mixed_positional_and_keyword_rejected(self):
        tmpl = CalldataTemplate("transfer(address to, uint256 amount)")
        try:
            tmpl(1, amount=2)
        except TypeError as e:
            assert "positional" in str(e) and "kwargs" in str(e)
        else:
            raise AssertionError("expected TypeError")


class TestRuntimeValueInContainers:
    """Regression: runtime ``Value`` handles nested inside tuple/list ABI
    container shapes must get the correct element/component ABI type so
    their placeholder is encoder-friendly.  Before the fix, the recursion
    passed ``abi_type=""`` for container leaves → address placeholder fell
    back to ``0`` → ``AddressEncoder`` rejected it."""

    def test_address_array_with_runtime_value(self):
        p = Program()
        v = p.const(0xCAFEBABE)
        tmpl = p.template("f(address[] addrs)")
        payload = tmpl(addrs=[v])
        # Dynamic array tail places the runtime value at the element's
        # calldata offset — must be resolvable (non-empty).
        assert len(payload.patches) == 1
        # No raise at template-build time is the key assertion; end-to-end
        # execution is covered by test_payload_executes.
        p.call_contract(_TARGET, payload)
        p.stop()
        _build_and_run(p)

    def test_tuple_with_runtime_address_leaf(self):
        p = Program()
        v = p.const(0xABAB_ABAB)
        tmpl = p.template("g((address,uint256) t)")
        payload = tmpl(t=(v, 1))
        # Address slot at head offset 4; uint256 static slot at 36.
        assert 4 in payload.patches
        p.call_contract(_TARGET, payload)
        p.stop()
        _build_and_run(p)

    def test_nested_tuple_with_array_of_values(self):
        # (address[], uint256) — nested dynamic array of runtime Values
        # inside a tuple.  The fix must walk both the tuple component
        # splitter and the array-element stripper to arrive at "address"
        # for the inner leaf.
        p = Program()
        v = p.const(0xDEAD_BEEF)
        tmpl = p.template("h((address[],uint256) t)")
        payload = tmpl(t=([v], 1))
        assert len(payload.patches) == 1

    def test_mixed_static_and_value_in_tuple(self):
        p = Program()
        amount = p.const(500)
        tmpl = p.template("i((address,uint256) t)")
        # Static address + runtime uint256 — uint256 component at offset 36.
        payload = tmpl(t=(0xAABB, amount))
        assert payload.patches == {36: amount}

    def test_fixed_size_array_of_values(self):
        # Fixed-size arrays split the same way as dynamic: element type
        # comes from stripping [N].
        p = Program()
        a, b, c = p.const(1), p.const(2), p.const(3)
        tmpl = p.template("j(uint256[3] xs)")
        payload = tmpl(xs=[a, b, c])
        assert len(payload.patches) == 3


class TestCallContractAcceptsPayload:
    def test_payload_goes_through_call_contract(self):
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        p.call_contract(_TARGET, tmpl(to=0xABAB, amount=100))
        p.stop()
        _build_and_run(p)

    def test_same_template_reused_shares_data_section(self):
        # All-runtime-Value args: the static portion of the blob (selector +
        # placeholder zero slots) is identical across invocations → dedup
        # keeps them on one data section.
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        recipient = p.const(0xAAAA)
        amount = p.const(1000)
        for _ in range(5):
            p.call_contract(_TARGET, tmpl(to=recipient, amount=amount))
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 1

    def test_distinct_scalar_static_args_still_dedup(self):
        # Scalar-static args go through the patch path, so per-row
        # differences in amount / recipient do NOT create new data
        # sections.  Regression guard: issue.md H2.
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        for i in range(5):
            p.call_contract(_TARGET, tmpl(to=0xAAAA + i, amount=1000 + i))
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 1

    def test_distinct_dynamic_args_produce_distinct_sections(self):
        # Dynamic-type args (bytes/string/T[]) still bake into the blob,
        # so distinct dynamic values correctly get distinct sections.
        p = Program()
        tmpl = p.template("swap(uint256 amountIn, bytes data)")
        for i in range(3):
            p.call_contract(_TARGET, tmpl(amountIn=100, data=bytes([i, i, i])))
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 3

    def test_same_dynamic_args_still_dedup(self):
        # Two invocations with identical dynamic args share a section even
        # if the scalar args differ.
        p = Program()
        tmpl = p.template("swap(uint256 amountIn, bytes data)")
        for i in range(3):
            p.call_contract(_TARGET, tmpl(amountIn=100 + i, data=b"\x01\x02\x03"))
        p.stop()
        _build_and_run(p)
        assert len(p._ctx.data_segment) == 1

    def test_explicit_patches_merge_over_template(self):
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        dynamic_amount = p.const(999)
        # tmpl wrote amount=100 into its patches via static literal; we override
        # offset 36 with an SSA value that wins.
        p.call_contract(
            _TARGET,
            tmpl(to=0xABAB, amount=100),
            patches={36: dynamic_amount},
        )
        p.stop()
        _build_and_run(p)

    def test_bulk_calls_with_runtime_values_share_one_section(self):
        # End-to-end: bulk calls through a template where the per-call
        # differences are runtime Values (not static literals) collapse to
        # one data section + per-call MSTORE patches.
        p = Program()
        xfer = p.template("transfer(address to, uint256 amount)")
        recipient = p.const(0xAA01)
        for i in range(4):
            amount = p.const(1000 + i)
            p.call_contract(_TARGET, xfer(to=recipient, amount=amount))
        p.stop()
        bc = p.build()
        assert len(p._ctx.data_segment) == 1
        # sanity: contract still executes cleanly
        result = mini_evm(bc)
        assert not result.is_error


class TestCalldataPayloadPublicAPI:
    def test_class_is_exported(self):
        # The class should be importable from the package root for users who
        # want to type-annotate callers that accept payloads.
        from pydefi.vm import CalldataPayload as Exported  # noqa: F401

        assert Exported is CalldataPayload
