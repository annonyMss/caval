"""Reproducibility & correctness tests for the frozen tool-identity encoder (F5).

The central property under test is DETERMINISM: the encoder is a fixed feature
extractor, so the same text must map to the same bytes across calls, across a
fresh instance, and whether or not the on-disk cache is used. If any of these
fail, conformal calibration and every downstream number become irreproducible.

Runnable two ways:
    pytest tests/test_tool_encoder.py
    python tests/test_tool_encoder.py        # no pytest needed
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from caval.trace_graph import ToolIdentityEncoder, EMB_DIM
from corpus.tool_descriptions import build_map, identity_string, collect_tool_docs

TOOLS = [
    "send_direct_message: Send a direct message from author to recipient.",
    "send_channel_message: Post a message to a Slack channel.",
    "read_channel_messages: Read the messages from the given channel.",
    "transfer_money: Transfer funds from the user's account to an IBAN.",
]


def _fresh_encoder(use_cache=False, cache_path=None):
    return ToolIdentityEncoder(cache_path=cache_path, use_cache=use_cache)


def test_encode_shape_and_dim():
    enc = _fresh_encoder()
    a = enc.encode(TOOLS)
    assert a.shape == (len(TOOLS), EMB_DIM), a.shape
    assert a.dtype == np.float32, a.dtype


def test_encode_deterministic_same_instance():
    enc = _fresh_encoder()
    a = enc.encode(TOOLS)
    b = enc.encode(TOOLS)
    assert np.array_equal(a, b), "same instance produced different bytes"


def test_encode_deterministic_fresh_instance():
    a = _fresh_encoder().encode(TOOLS)
    b = _fresh_encoder().encode(TOOLS)
    assert np.array_equal(a, b), "fresh instance produced different bytes"


def test_cache_roundtrip_identical():
    """Cached read must equal freshly-computed bytes (cache must not perturb)."""
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "cache.npz")
        e1 = ToolIdentityEncoder(cache_path=cp, use_cache=True)
        fresh = e1.encode(TOOLS)                 # computes + writes cache
        assert os.path.exists(cp), "cache not persisted"
        e2 = ToolIdentityEncoder(cache_path=cp, use_cache=True)
        cached = e2.encode(TOOLS)                # reads from cache only
        assert np.array_equal(fresh, cached), "cache changed the values"


def test_rows_l2_normalized():
    a = _fresh_encoder().encode(TOOLS)
    norms = np.linalg.norm(a, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms


def test_vocab_matrix_zero_row_and_alignment():
    vocab = {"send_direct_message": 1, "read_channel_messages": 2, "transfer_money": 3}
    idmap = {n: identity_string(n, {}) for n in vocab}  # name-only ok for this test
    enc = _fresh_encoder()
    mat = enc.build_vocab_matrix(vocab, idmap)
    assert mat.shape == (max(vocab.values()) + 1, EMB_DIM), mat.shape
    assert np.allclose(mat[0], 0.0), "row 0 (unknown/no-tool slot) must be zeros"
    # each tool row equals the direct encoding of its identity string
    for name, tid in vocab.items():
        direct = enc.encode([idmap[name]])[0]
        assert np.array_equal(mat[tid], direct), f"row {tid} misaligned for {name}"


def test_identity_geometry_semantic():
    """Sanity: two 'send message' tools are closer than send vs transfer_money.

    This is what makes an UNSEEN tool land near its semantic neighbours (the D4
    generalization argument). Thresholds are loose -- we assert ordering, not a value.
    """
    a = _fresh_encoder().encode(TOOLS)
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    send_send = cos(a[0], a[1])
    send_transfer = cos(a[0], a[3])
    assert send_send > send_transfer, (send_send, send_transfer)


def test_description_extraction_finds_slack_tools():
    """The AST description scan finds real slack tools (skips if source absent)."""
    docs, n_files = collect_tool_docs()
    if n_files == 0:
        print("  [skip] no suite tool modules on disk")
        return
    assert "send_direct_message" in docs, "known slack tool not documented"
    assert len(docs["send_direct_message"]) > 0


def test_build_map_coverage_shape():
    vocab = {"send_direct_message": 1, "read_channel_messages": 2, "definitely_not_a_tool_xyz": 3}
    mapping, cov = build_map(vocab)
    assert set(mapping) == set(vocab)
    assert cov["n_tools"] == 3
    # unknown tool falls back to bare name
    assert mapping["definitely_not_a_tool_xyz"] == "definitely_not_a_tool_xyz"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed of {len(fns)}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
