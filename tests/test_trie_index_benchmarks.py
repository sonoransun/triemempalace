"""
test_trie_index_benchmarks.py — Latency micro-benchmarks for TrieIndex.

Marked ``@pytest.mark.benchmark`` so they stay out of the default run
(``pyproject.toml`` deselects ``benchmark`` / ``slow`` / ``stress``).
Invoke with::

    pytest -m benchmark tests/test_trie_index_benchmarks.py -v

Each test builds a synthetic palace and measures warm-path latency for
the operation that most shapes real query performance: single-token
lookups, multi-keyword AND, prefix scans, temporal windows, the
single-drawer write hot path, and the full rebuild pipeline.

The thresholds below are **regression gates**, not targets. They are
deliberately loose (3–10× the observed p50 on an M-class Apple Silicon)
so CI on slower runners doesn't flap, while still catching any change
that re-introduces the old SQLite-era latency cliff (tens of
milliseconds or worse).
"""

import os
import random
import string
import time

import pytest

from mempalace.trie_index import TrieIndex

pytestmark = pytest.mark.benchmark


# ── Shared palace-builder fixture ────────────────────────────────────


def _percentiles(samples_ns: list[int]) -> tuple[float, float]:
    """Return (p50, p99) in microseconds."""
    s = sorted(samples_ns)
    n = len(s)
    p50 = s[n // 2] / 1000.0
    p99 = s[min(n - 1, int(n * 0.99))] / 1000.0
    return p50, p99


def _random_token(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


@pytest.fixture
def large_trie(tmp_dir):
    """Build a trie with 2,000 drawers × ~10 tokens each, strict filed_at order.

    Drawers are inserted in monotonically increasing ``filed_at`` order (50
    drawers per day across 40 days, starting 2026-01-01) so the temporal
    fast path is active — that's the realistic case for live-mined palaces.
    """
    db_path = os.path.join(tmp_dir, "bench_trie.lmdb")
    trie = TrieIndex(db_path=db_path)

    rng = random.Random(0xBEEF)
    # Pre-pick a token vocabulary so lookups can deterministically hit.
    vocab = [_random_token(rng) for _ in range(200)]
    wings = [f"wing_{i}" for i in range(4)]
    rooms = [f"room_{i}" for i in range(8)]

    items = []
    from datetime import date, timedelta

    epoch = date(2026, 1, 1)
    for i in range(2000):
        toks = " ".join(rng.sample(vocab, 10))
        day = epoch + timedelta(days=i // 50)  # 50 drawers per day × 40 days
        items.append(
            (
                f"drawer_{i:05d}",
                toks,
                {
                    "wing": wings[i % len(wings)],
                    "room": rooms[i % len(rooms)],
                    "filed_at": f"{day.isoformat()}T00:00:00",
                },
            )
        )
    trie.add_batch(items)

    # Stash the vocabulary on the trie object so tests can pick a
    # guaranteed-present token without peeking at LMDB directly.
    trie._bench_vocab = vocab  # type: ignore[attr-defined]
    trie._bench_wings = wings  # type: ignore[attr-defined]
    yield trie
    trie.close()


# ── Benchmarks ───────────────────────────────────────────────────────


def test_bench_single_token_lookup_warm(large_trie):
    """Warm lookup of a known token — the hot path for every keyword query."""
    vocab = large_trie._bench_vocab  # type: ignore[attr-defined]
    # Warm up the LRU cache and the mmap pages.
    for _ in range(50):
        large_trie.lookup(vocab[0])

    samples: list[int] = []
    token = vocab[0]
    for _ in range(2000):
        t0 = time.perf_counter_ns()
        large_trie.lookup(token)
        samples.append(time.perf_counter_ns() - t0)
    p50, p99 = _percentiles(samples)
    print(f"\n  single_token_lookup_warm: p50={p50:.1f}µs p99={p99:.1f}µs")
    # Measured baseline ~48 μs on M-class Apple Silicon; 3× slack for CI.
    assert p50 < 200.0, f"p50 regressed to {p50:.1f}µs"
    assert p99 < 800.0, f"p99 regressed to {p99:.1f}µs"


def test_bench_multi_keyword_and_warm(large_trie):
    """AND-intersection of three known tokens — the main hybrid-search hot path."""
    vocab = large_trie._bench_vocab  # type: ignore[attr-defined]
    kws = [vocab[0], vocab[10], vocab[20]]
    for _ in range(50):
        large_trie.keyword_search(kws, mode="all")

    samples: list[int] = []
    for _ in range(2000):
        t0 = time.perf_counter_ns()
        large_trie.keyword_search(kws, mode="all")
        samples.append(time.perf_counter_ns() - t0)
    p50, p99 = _percentiles(samples)
    print(f"\n  multi_keyword_and_warm:   p50={p50:.1f}µs p99={p99:.1f}µs")
    # Measured baseline ~4 μs; this is almost entirely Roaring C-intersect.
    assert p50 < 50.0, f"p50 regressed to {p50:.1f}µs"
    assert p99 < 200.0, f"p99 regressed to {p99:.1f}µs"


def test_bench_prefix_lookup_warm(large_trie):
    """2-character prefix — expands to ~25 matching tokens in the seeded vocab."""
    # Pick a prefix that actually hits; the vocab is lowercase ASCII.
    vocab = large_trie._bench_vocab  # type: ignore[attr-defined]
    prefix = vocab[0][:2]  # 2-char prefix of a known token
    for _ in range(20):
        large_trie.lookup(prefix, prefix=True)

    samples: list[int] = []
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        large_trie.lookup(prefix, prefix=True)
        samples.append(time.perf_counter_ns() - t0)
    p50, p99 = _percentiles(samples)
    print(f"\n  prefix_lookup_warm:       p50={p50:.1f}µs p99={p99:.1f}µs")
    # Measured baseline ~50 μs; 5× slack for CI.
    assert p50 < 300.0, f"p50 regressed to {p50:.1f}µs"


def test_bench_temporal_window_warm(large_trie):
    """Single-token lookup combined with a tight ``since``/``until`` window.

    Uses a 2-day window (~100 drawers) so the bench measures the temporal
    *filter* step dominated by the monotonic fast path + bitmap AND, not
    the drawer_id → drawer_id resolution which scales linearly with hit
    count. On a fixture with strict filed_at monotonicity the fast path
    returns a ``BitMap(range(...))`` in ~1 μs.
    """
    vocab = large_trie._bench_vocab  # type: ignore[attr-defined]
    token = vocab[0]
    for _ in range(20):
        large_trie.lookup(token, since="2026-01-10", until="2026-01-11")

    samples: list[int] = []
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        large_trie.lookup(token, since="2026-01-10", until="2026-01-11")
        samples.append(time.perf_counter_ns() - t0)
    p50, p99 = _percentiles(samples)
    print(f"\n  temporal_window_warm:     p50={p50:.1f}µs p99={p99:.1f}µs")
    # Confirm the fast path was actually used.
    mode = large_trie.stats().get("last_query_mode")
    assert mode == "mono_fast", f"expected mono_fast temporal path, got {mode!r}"
    # Measured baseline ~6 μs on the monotonic fast path.
    assert p50 < 80.0, f"p50 regressed to {p50:.1f}µs"


def test_bench_add_drawer_hot(large_trie):
    """Single-drawer write — the MCP ``tool_add_drawer`` hot path."""
    samples: list[int] = []
    for i in range(500):
        did = f"hot_drawer_{i:05d}"
        t0 = time.perf_counter_ns()
        large_trie.add_drawer(
            did,
            "alpha beta gamma delta epsilon unique_token_" + str(i),
            {"wing": "wing_0", "room": "room_0", "filed_at": "2026-04-01T00:00:00"},
        )
        samples.append(time.perf_counter_ns() - t0)
    p50, p99 = _percentiles(samples)
    print(f"\n  add_drawer_hot:           p50={p50:.1f}µs p99={p99:.1f}µs")
    # Measured baseline ~30 μs (LMDB+Roaring+mono state tracking). The
    # SQLite backend was several ms per call; v2 holds well under 200 μs.
    assert p50 < 200.0, f"p50 regressed to {p50:.1f}µs"
    assert p99 < 2000.0, f"p99 regressed to {p99:.1f}µs"


def test_bench_singleton_construction(tmp_path):
    """Second-and-subsequent TrieIndex(path) calls must be near-free.

    The singleton registry exists specifically to make the searcher's
    per-call ``TrieIndex(db_path=...)`` collapse to a dict lookup, so
    repeated construction must average under a couple of microseconds.
    """
    path = str(tmp_path / "singleton_bench.lmdb")
    t1 = TrieIndex(db_path=path)  # pays the open cost
    try:
        samples: list[int] = []
        for _ in range(10_000):
            t0 = time.perf_counter_ns()
            _ = TrieIndex(db_path=path)
            samples.append(time.perf_counter_ns() - t0)
        p50, p99 = _percentiles(samples)
        print(f"\n  singleton_construction:   p50={p50:.2f}µs p99={p99:.2f}µs")
        # Measured baseline ~0.5 μs (dict lookup); 10× slack for CI jitter.
        assert p50 < 5.0, f"p50 regressed to {p50:.2f}µs — singleton cache broken?"
    finally:
        t1.close()


def test_bench_bulk_add_batch(tmp_dir):
    """Wall-clock for bulk ``add_batch`` — the core of ``rebuild_from_collection``.

    We skip Chroma here because spinning up the embedding model and
    writing 500 documents to a fresh collection is dominated by ONNX
    inference and has nothing to do with the trie's throughput. This
    test instead measures the trie-side work directly — tokenize, intern,
    bitmap update, LMDB commit — on 5,000 synthetic drawers.
    """
    rng = random.Random(0xFEED)
    vocab = [_random_token(rng) for _ in range(100)]
    items = []
    for i in range(5000):
        items.append(
            (
                f"bulk_drawer_{i:05d}",
                " ".join(rng.sample(vocab, 8)),
                {
                    "wing": f"wing_{i % 3}",
                    "room": f"room_{i % 5}",
                    "filed_at": f"2026-{(i % 3) + 1:02d}-{(i % 27) + 1:02d}T00:00:00",
                },
            )
        )

    trie_path = os.path.join(tmp_dir, "bulk_bench.lmdb")
    trie = TrieIndex(db_path=trie_path)
    try:
        t0 = time.perf_counter_ns()
        trie.add_batch(items)
        elapsed_s = (time.perf_counter_ns() - t0) / 1e9
        per_drawer_us = (elapsed_s * 1e6) / len(items)
        print(
            f"\n  bulk_add_batch (5000 drawers): {elapsed_s * 1000:.0f} ms "
            f"({per_drawer_us:.1f} µs/drawer)"
        )
        # Measured baseline ~30 ms (6 μs/drawer) on the dev box. Keep the
        # gate loose enough for CI, tight enough to catch regressions.
        assert elapsed_s < 1.0, f"bulk add regressed to {elapsed_s:.2f}s"
    finally:
        trie.close()
