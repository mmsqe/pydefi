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

    def test_blob_is_zero_filled_with_selector_prefix(self):
        tmpl = CalldataTemplate("transfer(address,uint256)")
        blob = tmpl._blob
        assert blob.startswith(bytes.fromhex("a9059cbb"))  # transfer selector
        assert blob[4:] == b"\x00" * 64

    def test_static_int_args_encoded_as_patches_not_in_blob(self):
        # With dedup in mind: the blob stays all-zero so separate tmpl(...)
        # invocations with different literal args share one data section.
        tmpl = CalldataTemplate("transfer(address,uint256)")
        p1 = tmpl(0xABAB, 100)
        p2 = tmpl(0xCDCD, 200)
        assert p1.blob == p2.blob
        # literal != 0 → patch emitted for each non-zero slot
        assert p1.patches == {4: 0xABAB, 36: 100}
        assert p2.patches == {4: 0xCDCD, 36: 200}

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


class TestCallContractAcceptsPayload:
    def test_payload_goes_through_call_contract(self):
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        p.call_contract(_TARGET, tmpl(to=0xABAB, amount=100))
        p.stop()
        _build_and_run(p)

    def test_same_template_reused_shares_data_section(self):
        p = Program()
        tmpl = p.template("transfer(address to, uint256 amount)")
        for i in range(5):
            p.call_contract(_TARGET, tmpl(to=0xAAAA + i, amount=1000 + i))
        p.stop()
        _build_and_run(p)
        # The shared zero-filled blob is one section; all 5 invocations reuse it.
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

    def test_bulk_calls_stay_on_one_data_section(self):
        # End-to-end: repeated calls through the same template don't bloat
        # the bytecode with repeated copies of the blob.
        p = Program()
        xfer = p.template("transfer(address to, uint256 amount)")
        for recipient in (0xAA01, 0xAA02, 0xAA03, 0xAA04):
            p.call_contract(_TARGET, xfer(to=recipient, amount=1000))
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
