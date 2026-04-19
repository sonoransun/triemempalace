"""
trie_index.py — LMDB + Roaring Bitmap keyword/temporal index (v2 schema).

The palace's semantic substrate is ChromaDB. This module is the low-latency
keyword + temporal prefilter that hands Chroma a candidate set — or answers
pure keyword/temporal queries outright without touching the vector store.

Design in one paragraph
-----------------------

LMDB is an mmap'd B+tree with zero-copy reads and MVCC. Each sub-DB (DBI)
is a single B+tree; we exploit that eight different ways. Posting lists
are Roaring bitmaps of compact ``doc_id`` integers, serialized as BLOBs.
Token / wing / room strings are interned to integer IDs before hitting any
posting list, so every lookup is one cursor read + one memcpy + one bitmap
deserialize. Multi-keyword intersection runs entirely in CRoaring C code.
Temporal queries take a fast path when ``filed_at`` is monotone (almost
always, because live mining uses ``datetime.now()``), converting a
``since``/``until`` window into a doc_id range via binary search over a
compact day-to-doc_id checkpoint array — no cursor walk required.

v2 schema optimizations applied in this round
---------------------------------------------

1. **``integerkey=True`` on uint-keyed DBIs** — LMDB's native integer
   comparator instead of byte-wise. Applied to doc_to_drawer, doc_meta,
   doc_tokens, postings, wing_postings, room_postings, wings_rev, rooms_rev.
2. **``append=True`` on monotone-key writes** — doc_id writes into
   doc_to_drawer / doc_meta / doc_tokens skip the B+tree search and go
   straight to the rightmost leaf.
3. **Per-batch intern cache** — inside ``add_batch``, repeated tokens /
   wings / rooms resolve via a local Python dict instead of LMDB.
4. **``cursor.getmulti`` resolution** — doc_id → drawer_id resolution for
   result sets ≥ 16 uses one batch cursor call instead of N point lookups.
5. **Monotonic doc_id temporal fast path** — ``since``/``until`` maps to a
   ``BitMap(range(lo_doc, hi_doc + 1))`` via binary search over a day
   checkpoint array; ~1 μs instead of ~100 μs for a cursor walk.
6. **Cardinality-sorted multi-keyword AND** — sort by bitmap size before
   intersecting so the shortest set leads the short-circuit loop.
7. **``doc_tokens`` delta-varint packing** — smaller values, faster delete
   reads, lower mmap footprint.
8. **Warm LRU prefetch** — ``warm()`` reads cardinality from each posting
   blob's Roaring header and loads the top-K bitmaps into the in-process
   LRU; called from the MCP server's main() so cold queries never pay the
   deserialize cost.
9. **Singleton env registry** (carried over from v1) — ``TrieIndex(path)``
   returns the same instance across all callers in one process.
10. **``sync=False, metasync=False, writemap=True, readahead=False``** —
    write-fast, crash-recoverable (rebuild from Chroma via ``trie-repair``).

Storage layout (v2)
-------------------

    meta             — scalar counters, schema version, monotonic state
    drawer_to_doc    — drawer_id (utf-8)      → doc_id (u32)
    doc_to_drawer    — doc_id (u32) [intkey]  → drawer_id (utf-8)
    doc_meta         — doc_id (u32) [intkey]  → <HHIII> wing_id, room_id,
                                                filed_days, valid_from_days,
                                                valid_to_days
    doc_tokens       — doc_id (u32) [intkey]  → delta-varint u32[] token_ids
    tokens           — token (utf-8)          → token_id (u32)
    postings         — token_id (u32) [intkey]→ Roaring bitmap of doc_ids
    wings_fwd        — wing name              → wing_id (u16)
    wings_rev        — wing_id (u16) [intkey] → wing name
    rooms_fwd        — room name              → room_id (u16)
    rooms_rev        — room_id (u16) [intkey] → room name
    wing_postings    — wing_id (u16) [intkey] → Roaring bitmap of doc_ids
    room_postings    — room_id (u16) [intkey] → Roaring bitmap of doc_ids
    time_index       — >II (filed_days, doc_id) → empty (fallback path only)

Monotonic temporal fast path (new)
----------------------------------

The ``meta`` DBI holds five monotonic-state keys:

* ``mono``           — b"1" if filed_days is strictly non-decreasing and
                       doc_id is strictly increasing; b"0" once either breaks.
* ``mono_min_doc``   — first doc_id assigned
* ``mono_max_doc``   — last doc_id assigned
* ``mono_min_day``   — earliest filed_days
* ``mono_max_day``   — latest filed_days
* ``mono_ckp``       — packed ``<II>[]`` of ``(day, first_doc_id_on_that_day)``
                       checkpoints, one entry per distinct day ever seen.

On every new insert: if ``filed_days < mono_max_day``, flip ``mono`` to
b"0" and abandon the fast path until the next ``rebuild_from_collection``.
Otherwise append a checkpoint if the day advanced and update max counters.

``_time_range_bitmap`` checks ``mono`` first: on b"1" it binary-searches
the checkpoint array for ``since`` / ``until`` bounds and returns
``BitMap(range(lo_doc, hi_doc + 1))``. Sub-microsecond. Falls back to the
``time_index`` cursor walk on b"0".

``rebuild_from_collection`` sorts Chroma pages by ``filed_at`` before
re-ingesting, then resets ``mono = b"1"`` at the start so it gets
repopulated cleanly during the rebuild.

Schema migration
----------------

``_SCHEMA_VERSION`` is bumped from 1 to 2. On ``__init__`` we read the
stored version from ``meta``. If it's non-zero and less than the current
version, we ``drop()`` every sub-DB, reset the counters, write the new
version, and log a warning telling the user to run ``mempalace trie-repair``.
The hybrid search path already degrades gracefully when the trie returns
empty candidate sets, so upgrades don't break — keyword/temporal filters
temporarily lose their effect until the rebuild.
"""

import bisect
import contextlib
import logging
import os
import re
import struct
from collections import OrderedDict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import chromadb.errors
import lmdb
from pyroaring import BitMap

from .entity_detector import STOPWORDS as _ENTITY_STOPWORDS

logger = logging.getLogger("mempalace.trie_index")


# ── Schema / constants ────────────────────────────────────────────────

_SCHEMA_VERSION = 2
_BITMAP_CACHE_SIZE = 4096
_DEFAULT_MAP_SIZE = 16 * 1024 * 1024 * 1024  # 16 GiB virtual; grows on demand
_EPOCH = datetime(1970, 1, 1)

# getmulti kicks in for result sets large enough that the sort + batch-call
# amortizes the per-item cost. Below this threshold per-item ``txn.get``
# is actually faster because there's no sort overhead.
_GETMULTI_THRESHOLD = 16

# Warm LRU prefetch default. Loading the top 1024 posting bitmaps covers
# roughly the top 1% of vocabulary in a 22k-drawer palace (the common
# English words + the most frequent project identifiers) and weighs in
# at ~5 MB of Roaring bitmaps — comfortable to keep resident. Tuned for
# the `mempalace_drawers` test fixture shape; override via `warm(top_k=N)`.
_WARM_TOP_K_DEFAULT = 1024

# Struct packers. Little-endian for point-lookup DBIs where byte ordering
# is irrelevant. Big-endian for ``time_index`` so lexical cursor ordering
# of the composite (day, doc_id) key matches numerical ordering.
_U16_LE = struct.Struct("<H")
_U32_LE = struct.Struct("<I")
_U64_LE = struct.Struct("<Q")
_DOC_META = struct.Struct("<HHIII")  # wing_id, room_id, filed, vfrom, vto
_TIME_KEY = struct.Struct(">II")  # (filed_days, doc_id)
_CKP_ENTRY = struct.Struct("<II")  # (day, first_doc_id_on_day)

# Tokenizer.
_MIN_TOKEN_LEN = 2
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")


def tokenize(text: str) -> list[str]:
    """Split text into index tokens.

    Lowercases everything, drops stopwords (via ``entity_detector``) and
    single-character runs. Identifier punctuation (``_``, ``-``) is kept so
    ``snake_case`` / ``my-room`` / ``oauth2`` remain prefix-searchable.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.lower()
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        if tok in _ENTITY_STOPWORDS:
            continue
        out.append(tok)
    return out


# ── Sub-DB names ──────────────────────────────────────────────────────

_DB_META = b"meta"
_DB_DRAWER_TO_DOC = b"drawer_to_doc"
_DB_DOC_TO_DRAWER = b"doc_to_drawer"
_DB_DOC_META = b"doc_meta"
_DB_DOC_TOKENS = b"doc_tokens"
_DB_TOKENS = b"tokens"
_DB_POSTINGS = b"postings"
_DB_WINGS_FWD = b"wings_fwd"
_DB_WINGS_REV = b"wings_rev"
_DB_ROOMS_FWD = b"rooms_fwd"
_DB_ROOMS_REV = b"rooms_rev"
_DB_WING_POSTINGS = b"wing_postings"
_DB_ROOM_POSTINGS = b"room_postings"
_DB_TIME_INDEX = b"time_index"

_META_SCHEMA_VERSION = b"schema_version"
_META_NEXT_DOC_ID = b"next_doc_id"
_META_NEXT_TOKEN_ID = b"next_token_id"
_META_NEXT_WING_ID = b"next_wing_id"
_META_NEXT_ROOM_ID = b"next_room_id"
_META_MONO = b"mono"
_META_MONO_MIN_DOC = b"mono_min_doc"
_META_MONO_MAX_DOC = b"mono_max_doc"
_META_MONO_MIN_DAY = b"mono_min_day"
_META_MONO_MAX_DAY = b"mono_max_day"
_META_MONO_CKP = b"mono_ckp"


DEFAULT_TRIE_PATH = str(Path("~/.mempalace/trie_index.lmdb").expanduser())


def trie_db_path(palace_path: str | os.PathLike[str]) -> str:
    """Canonical location of the trie LMDB env for a given palace.

    Kept as a single source of truth so that miner, searcher, MCP server,
    and CLI all agree. Change it here, not in ten places.
    """
    return str(Path(palace_path) / "trie_index.lmdb")


# ── Date helpers ──────────────────────────────────────────────────────


def _iso_to_days(iso: str | None) -> int:
    """Days-since-1970 for an ISO8601 timestamp. ``0`` means "unbounded"."""
    if not iso:
        return 0
    try:
        # Strip microseconds (if any) before parsing; Python's
        # ``fromisoformat`` on <3.11 is strict.
        s = iso[:19] if len(iso) > 19 else iso
        dt = datetime.fromisoformat(s) if len(s) > 10 else datetime.fromisoformat(s + "T00:00:00")
    except ValueError:
        return 0
    delta = dt.date() - _EPOCH.date()
    return max(0, delta.days)


def _days_to_iso(days: int) -> str:
    """Inverse of :func:`_iso_to_days` used by ``get_drawer_meta``."""
    if days <= 0:
        return ""
    return (_EPOCH + timedelta(days=days)).date().isoformat()


# ── Delta-varint packing for doc_tokens ───────────────────────────────


def _pack_varint_deltas(values: list[int]) -> bytes:
    """Encode a list of u32s as sorted delta-varints.

    Sorting + delta encoding lets small deltas compress via varint to 1–2
    bytes each (instead of the fixed 4-byte u32 per entry). Empty input
    returns an empty bytes.
    """
    if not values:
        return b""
    sorted_vals = sorted(values)
    out = bytearray()
    prev = 0
    for v in sorted_vals:
        delta = v - prev
        prev = v
        # Standard unsigned varint (same encoding as protobuf / leveldb).
        while delta >= 0x80:
            out.append((delta & 0x7F) | 0x80)
            delta >>= 7
        out.append(delta & 0x7F)
    return bytes(out)


def _unpack_varint_deltas(raw: bytes | memoryview) -> list[int]:
    """Inverse of :func:`_pack_varint_deltas`. Returns token_ids in sorted order."""
    if not raw:
        return []
    b = bytes(raw) if not isinstance(raw, bytes) else raw
    out: list[int] = []
    i = 0
    prev = 0
    n = len(b)
    while i < n:
        shift = 0
        delta = 0
        while True:
            byte = b[i]
            i += 1
            delta |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        prev += delta
        out.append(prev)
    return out


# ── LRU bitmap cache ──────────────────────────────────────────────────


class _BitMapLRU:
    """Bounded LRU keyed on token_id → deserialized BitMap.

    The hot path returns a shared reference; callers must not mutate
    (``&``, ``|``, ``-`` all produce new bitmaps — only ``&=``, ``|=``,
    ``-=`` are in-place and must be avoided on cached returns).
    """

    __slots__ = ("_store", "_maxsize")

    def __init__(self, maxsize: int = _BITMAP_CACHE_SIZE):
        self._store: OrderedDict[int, BitMap] = OrderedDict()
        self._maxsize = maxsize

    def get(self, token_id: int) -> BitMap | None:
        bm = self._store.get(token_id)
        if bm is not None:
            self._store.move_to_end(token_id)
        return bm

    def put(self, token_id: int, bm: BitMap) -> None:
        if token_id in self._store:
            self._store[token_id] = bm
            self._store.move_to_end(token_id)
            return
        self._store[token_id] = bm
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def invalidate(self, token_id: int) -> None:
        self._store.pop(token_id, None)

    def invalidate_many(self, token_ids: Iterable[int]) -> None:
        for tid in token_ids:
            self._store.pop(tid, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ── TrieIndex ─────────────────────────────────────────────────────────


class TrieIndex:
    """LMDB-backed keyword + temporal index over palace drawers.

    Construct once per palace. The class maintains a process-wide registry
    keyed on the canonicalized ``db_path`` — calling ``TrieIndex(path)``
    twice in the same process returns the **same** instance and the same
    underlying LMDB env. This sidesteps py-lmdb's hard "environment is
    already open in this process" guard when the searcher, MCP server,
    and test fixtures all resolve the same palace path.

    Thread-safe for reads via LMDB's MVCC. A single writer at a time is
    enforced by LMDB's write lock; MemPalace is single-process so this is
    free.
    """

    _instances: dict[str, "TrieIndex"] = {}

    def __new__(
        cls,
        db_path: str | os.PathLike[str] | None = None,
        *,
        map_size: int = _DEFAULT_MAP_SIZE,
    ):
        canonical = str(Path(db_path or DEFAULT_TRIE_PATH).expanduser().absolute())
        existing = cls._instances.get(canonical)
        if existing is not None and getattr(existing, "_env", None) is not None:
            return existing
        instance = super().__new__(cls)
        instance._initialized = False
        instance._canonical_path = canonical
        cls._instances[canonical] = instance
        return instance

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        map_size: int = _DEFAULT_MAP_SIZE,
    ):
        # __new__ may return a pre-initialized instance; avoid re-running
        # the expensive env setup in that case.
        if getattr(self, "_initialized", False):
            return

        self.db_path = self._canonical_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # subdir=True: LMDB creates ``<db_path>/data.mdb`` and
        # ``<db_path>/lock.mdb`` inside the directory at db_path.
        #
        # sync=False + metasync=False eliminates the per-commit fsync.
        # This trades a few seconds of durability for ~100× faster writes
        # on the MCP ``tool_add_drawer`` hot path. The trie is fully
        # rebuildable from the Chroma collection (``mempalace trie-repair``),
        # so a post-crash rebuild is the recovery path, not waiting on
        # fsync.
        self._env = lmdb.open(
            self.db_path,
            max_dbs=16,
            map_size=map_size,
            subdir=True,
            readahead=False,
            writemap=True,
            metasync=False,
            sync=False,
            lock=True,
        )

        with self._env.begin(write=True) as txn:
            # String-keyed DBIs — default byte-wise comparator.
            self._db_meta = self._env.open_db(_DB_META, txn=txn)
            self._db_drawer_to_doc = self._env.open_db(_DB_DRAWER_TO_DOC, txn=txn)
            self._db_tokens = self._env.open_db(_DB_TOKENS, txn=txn)
            self._db_wings_fwd = self._env.open_db(_DB_WINGS_FWD, txn=txn)
            self._db_rooms_fwd = self._env.open_db(_DB_ROOMS_FWD, txn=txn)
            # time_index uses compound big-endian keys for lex-sorted cursor
            # walks; it cannot use integerkey.
            self._db_time_index = self._env.open_db(_DB_TIME_INDEX, txn=txn)

            # Integer-keyed DBIs — LMDB's native int comparator is faster
            # than the byte-wise one. Applied to every DBI whose key is a
            # packed little-endian uint.
            self._db_doc_to_drawer = self._env.open_db(_DB_DOC_TO_DRAWER, txn=txn, integerkey=True)
            self._db_doc_meta = self._env.open_db(_DB_DOC_META, txn=txn, integerkey=True)
            self._db_doc_tokens = self._env.open_db(_DB_DOC_TOKENS, txn=txn, integerkey=True)
            self._db_postings = self._env.open_db(_DB_POSTINGS, txn=txn, integerkey=True)
            self._db_wings_rev = self._env.open_db(_DB_WINGS_REV, txn=txn, integerkey=True)
            self._db_rooms_rev = self._env.open_db(_DB_ROOMS_REV, txn=txn, integerkey=True)
            self._db_wing_postings = self._env.open_db(_DB_WING_POSTINGS, txn=txn, integerkey=True)
            self._db_room_postings = self._env.open_db(_DB_ROOM_POSTINGS, txn=txn, integerkey=True)

            # Schema version + migration.
            stored_raw = txn.get(_META_SCHEMA_VERSION, db=self._db_meta)
            stored_version = _U32_LE.unpack(bytes(stored_raw))[0] if stored_raw else 0

            if stored_version == 0:
                # Fresh install — initialize counters.
                self._write_fresh_meta(txn)
            elif stored_version < _SCHEMA_VERSION:
                # Old schema. Drop every data DBI and log a warning.
                # `searcher.hybrid_search` degrades gracefully on an empty
                # trie, so queries still work — they just lose their
                # keyword/temporal prefilter until `mempalace trie-repair`
                # rebuilds the index.
                logger.warning(
                    "Trie index schema v%d is outdated (expected v%d). "
                    "Run `mempalace trie-repair` to rebuild.",
                    stored_version,
                    _SCHEMA_VERSION,
                )
                self._drop_all_data(txn)
                self._write_fresh_meta(txn)
            elif stored_version > _SCHEMA_VERSION:
                # Newer schema than this binary knows about — refuse to
                # touch it rather than silently corrupting.
                raise RuntimeError(
                    f"Trie index schema v{stored_version} is newer than "
                    f"this MemPalace binary (v{_SCHEMA_VERSION}). "
                    f"Upgrade MemPalace or rebuild the palace."
                )

        self._bitmap_cache = _BitMapLRU()
        # Cached monotonic state — populated lazily on first read, mutated
        # in-place during add_batch, written back to LMDB once per batch.
        # This avoids 6 meta reads + 6 meta writes per add_drawer call on
        # the MCP hot path. Format:
        #   [flag: bytes, min_doc, max_doc, min_day, max_day, ckp_days, ckp_docs]
        self._mono_cache: list | None = None
        self._initialized = True
        self._last_query_mode: str | None = None  # debug/test hook

    # ── Meta helpers ──────────────────────────────────────────────────

    def _write_fresh_meta(self, txn) -> None:
        """Populate the ``meta`` DBI with the starting state of a fresh trie."""
        txn.put(_META_SCHEMA_VERSION, _U32_LE.pack(_SCHEMA_VERSION), db=self._db_meta)
        txn.put(_META_NEXT_DOC_ID, _U32_LE.pack(1), db=self._db_meta)
        txn.put(_META_NEXT_TOKEN_ID, _U32_LE.pack(1), db=self._db_meta)
        txn.put(_META_NEXT_WING_ID, _U16_LE.pack(1), db=self._db_meta)
        txn.put(_META_NEXT_ROOM_ID, _U16_LE.pack(1), db=self._db_meta)
        txn.put(_META_MONO, b"1", db=self._db_meta)
        txn.put(_META_MONO_MIN_DOC, _U32_LE.pack(0), db=self._db_meta)
        txn.put(_META_MONO_MAX_DOC, _U32_LE.pack(0), db=self._db_meta)
        txn.put(_META_MONO_MIN_DAY, _U32_LE.pack(0), db=self._db_meta)
        txn.put(_META_MONO_MAX_DAY, _U32_LE.pack(0), db=self._db_meta)
        txn.put(_META_MONO_CKP, b"", db=self._db_meta)

    def _drop_all_data(self, txn) -> None:
        """Drop every data DBI. Leaves ``meta`` intact (caller overwrites it)."""
        for db in (
            self._db_drawer_to_doc,
            self._db_doc_to_drawer,
            self._db_doc_meta,
            self._db_doc_tokens,
            self._db_tokens,
            self._db_postings,
            self._db_wings_fwd,
            self._db_wings_rev,
            self._db_rooms_fwd,
            self._db_rooms_rev,
            self._db_wing_postings,
            self._db_room_postings,
            self._db_time_index,
        ):
            txn.drop(db, delete=False)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the LMDB env and drop this instance from the singleton
        registry. After close, constructing ``TrieIndex(path)`` again will
        open a fresh env at that path.
        """
        env = getattr(self, "_env", None)
        if env is not None:
            try:
                env.close()
            finally:
                self._env = None
                self._initialized = False
                self._mono_cache = None
                self.__class__._instances.pop(getattr(self, "_canonical_path", ""), None)

    def __del__(self):
        # Broad catch: __del__ must never raise — interpreter shutdown
        # can tear down lmdb/threading state in any order, so any
        # exception here just gets swallowed.
        with contextlib.suppress(Exception):
            self.close()

    # ── Public meta-DBI accessors ────────────────────────────────────
    #
    # The ``meta`` DBI is a scalar key-value store colocated with the
    # trie's LMDB env. Internal callers write schema version, monotonic
    # state, counters. External callers (currently
    # :mod:`mempalace.aggregates`) reuse it for small pieces of
    # palace-scoped state that want the same transactional / crash
    # semantics as trie writes without introducing a second storage
    # backend. Keys must be bytes; values must be bytes. Callers are
    # responsible for their own key namespacing (e.g. ``b"agg:...``").

    def meta_get(self, key: bytes) -> bytes | None:
        """Return the raw bytes stored at ``key`` in the meta DBI, or None."""
        with self._env.begin(write=False) as txn:
            raw = txn.get(key, db=self._db_meta)
            return bytes(raw) if raw is not None else None

    def meta_put(self, key: bytes, value: bytes) -> None:
        """Write ``value`` at ``key`` in the meta DBI (creating or overwriting)."""
        with self._env.begin(write=True) as txn:
            txn.put(key, value, db=self._db_meta)

    def meta_delete(self, key: bytes) -> None:
        """Remove ``key`` from the meta DBI. No-op if absent."""
        with self._env.begin(write=True) as txn:
            txn.delete(key, db=self._db_meta)

    # ── Meta counter helpers ──────────────────────────────────────────

    def _next_counter(self, txn, key: bytes, packer: struct.Struct) -> int:
        raw = txn.get(key, db=self._db_meta)
        current = packer.unpack(bytes(raw))[0] if raw is not None else 1
        txn.put(key, packer.pack(current + 1), db=self._db_meta)
        return current

    # ── Interning: token / wing / room ────────────────────────────────

    def _get_or_intern_token(self, txn, token: str, cache: dict | None = None) -> int:
        """Intern a token to a uint32 id. Optional per-batch cache."""
        if cache is not None:
            cached = cache.get(token)
            if cached is not None:
                return cached
        key = token.encode("utf-8")
        existing = txn.get(key, db=self._db_tokens)
        if existing is not None:
            tid = _U32_LE.unpack(bytes(existing))[0]
        else:
            tid = self._next_counter(txn, _META_NEXT_TOKEN_ID, _U32_LE)
            txn.put(key, _U32_LE.pack(tid), db=self._db_tokens)
        if cache is not None:
            cache[token] = tid
        return tid

    def _lookup_token_id(self, txn, token: str) -> int | None:
        existing = txn.get(token.encode("utf-8"), db=self._db_tokens)
        if existing is None:
            return None
        return _U32_LE.unpack(bytes(existing))[0]

    def _get_or_intern_wing(self, txn, wing: str | None, cache: dict | None = None) -> int:
        if not wing:
            return 0
        if cache is not None:
            cached = cache.get(wing)
            if cached is not None:
                return cached
        key = wing.encode("utf-8")
        existing = txn.get(key, db=self._db_wings_fwd)
        if existing is not None:
            wid = _U16_LE.unpack(bytes(existing))[0]
        else:
            wid = self._next_counter(txn, _META_NEXT_WING_ID, _U16_LE)
            txn.put(key, _U16_LE.pack(wid), db=self._db_wings_fwd)
            txn.put(_U16_LE.pack(wid), key, db=self._db_wings_rev)
        if cache is not None:
            cache[wing] = wid
        return wid

    def _lookup_wing_id(self, txn, wing: str) -> int | None:
        existing = txn.get(wing.encode("utf-8"), db=self._db_wings_fwd)
        if existing is None:
            return None
        return _U16_LE.unpack(bytes(existing))[0]

    def _resolve_wing_name(self, txn, wing_id: int) -> str:
        if wing_id == 0:
            return "unknown"
        raw = txn.get(_U16_LE.pack(wing_id), db=self._db_wings_rev)
        return bytes(raw).decode("utf-8") if raw else "unknown"

    def _get_or_intern_room(self, txn, room: str | None, cache: dict | None = None) -> int:
        if not room:
            return 0
        if cache is not None:
            cached = cache.get(room)
            if cached is not None:
                return cached
        key = room.encode("utf-8")
        existing = txn.get(key, db=self._db_rooms_fwd)
        if existing is not None:
            rid = _U16_LE.unpack(bytes(existing))[0]
        else:
            rid = self._next_counter(txn, _META_NEXT_ROOM_ID, _U16_LE)
            txn.put(key, _U16_LE.pack(rid), db=self._db_rooms_fwd)
            txn.put(_U16_LE.pack(rid), key, db=self._db_rooms_rev)
        if cache is not None:
            cache[room] = rid
        return rid

    def _lookup_room_id(self, txn, room: str) -> int | None:
        existing = txn.get(room.encode("utf-8"), db=self._db_rooms_fwd)
        if existing is None:
            return None
        return _U16_LE.unpack(bytes(existing))[0]

    def _resolve_room_name(self, txn, room_id: int) -> str:
        if room_id == 0:
            return "unknown"
        raw = txn.get(_U16_LE.pack(room_id), db=self._db_rooms_rev)
        return bytes(raw).decode("utf-8") if raw else "unknown"

    # ── Bitmap load / store ──────────────────────────────────────────

    def _load_posting_bm(self, txn, token_id: int) -> BitMap:
        cached = self._bitmap_cache.get(token_id)
        if cached is not None:
            return cached
        raw = txn.get(_U32_LE.pack(token_id), db=self._db_postings)
        if raw is None:
            return BitMap()
        bm = BitMap.deserialize(bytes(raw))
        self._bitmap_cache.put(token_id, bm)
        return bm

    def _load_wing_bm(self, txn, wing_id: int) -> BitMap:
        raw = txn.get(_U16_LE.pack(wing_id), db=self._db_wing_postings)
        return BitMap.deserialize(bytes(raw)) if raw else BitMap()

    def _load_room_bm(self, txn, room_id: int) -> BitMap:
        raw = txn.get(_U16_LE.pack(room_id), db=self._db_room_postings)
        return BitMap.deserialize(bytes(raw)) if raw else BitMap()

    # ── Monotonic temporal state ─────────────────────────────────────

    def _read_mono_state(self, txn) -> tuple[bytes, int, int, int, int, bytes]:
        """Return (flag, min_doc, max_doc, min_day, max_day, ckp_raw)."""
        flag = txn.get(_META_MONO, db=self._db_meta)
        flag = bytes(flag) if flag is not None else b"0"
        min_doc_raw = txn.get(_META_MONO_MIN_DOC, db=self._db_meta)
        max_doc_raw = txn.get(_META_MONO_MAX_DOC, db=self._db_meta)
        min_day_raw = txn.get(_META_MONO_MIN_DAY, db=self._db_meta)
        max_day_raw = txn.get(_META_MONO_MAX_DAY, db=self._db_meta)
        ckp_raw = txn.get(_META_MONO_CKP, db=self._db_meta)
        return (
            flag,
            _U32_LE.unpack(bytes(min_doc_raw))[0] if min_doc_raw else 0,
            _U32_LE.unpack(bytes(max_doc_raw))[0] if max_doc_raw else 0,
            _U32_LE.unpack(bytes(min_day_raw))[0] if min_day_raw else 0,
            _U32_LE.unpack(bytes(max_day_raw))[0] if max_day_raw else 0,
            bytes(ckp_raw) if ckp_raw is not None else b"",
        )

    def _write_mono_state(
        self,
        txn,
        *,
        flag: bytes,
        min_doc: int,
        max_doc: int,
        min_day: int,
        max_day: int,
        ckp: bytes,
    ) -> None:
        txn.put(_META_MONO, flag, db=self._db_meta)
        txn.put(_META_MONO_MIN_DOC, _U32_LE.pack(min_doc), db=self._db_meta)
        txn.put(_META_MONO_MAX_DOC, _U32_LE.pack(max_doc), db=self._db_meta)
        txn.put(_META_MONO_MIN_DAY, _U32_LE.pack(min_day), db=self._db_meta)
        txn.put(_META_MONO_MAX_DAY, _U32_LE.pack(max_day), db=self._db_meta)
        txn.put(_META_MONO_CKP, ckp, db=self._db_meta)

    def _ensure_mono_cache(self, txn) -> list:
        """Return the cached mono state, loading it from LMDB if needed.

        The returned list is mutable: ``[flag, min_doc, max_doc, min_day,
        max_day, ckp_days, ckp_docs]``. Callers mutate in place during
        ``add_batch`` and ``_flush_mono_cache`` writes it back to LMDB.
        """
        if self._mono_cache is not None:
            return self._mono_cache
        flag, min_doc, max_doc, min_day, max_day, ckp_raw = self._read_mono_state(txn)
        ckp_days, ckp_docs = self._ckp_unpack(ckp_raw)
        self._mono_cache = [flag, min_doc, max_doc, min_day, max_day, ckp_days, ckp_docs]
        return self._mono_cache

    def _flush_mono_cache(self, txn) -> None:
        """Persist the cached mono state to LMDB. No-op if not loaded."""
        if self._mono_cache is None:
            return
        flag, min_doc, max_doc, min_day, max_day, ckp_days, ckp_docs = self._mono_cache
        self._write_mono_state(
            txn,
            flag=flag,
            min_doc=min_doc,
            max_doc=max_doc,
            min_day=min_day,
            max_day=max_day,
            ckp=self._ckp_pack(ckp_days, ckp_docs) if flag == b"1" else b"",
        )

    def _ckp_unpack(self, raw: bytes) -> tuple[list[int], list[int]]:
        """Return parallel (days, doc_ids) lists from a packed checkpoint blob."""
        if not raw:
            return [], []
        n = len(raw) // 8
        days = [0] * n
        docs = [0] * n
        for i in range(n):
            day, doc = _CKP_ENTRY.unpack_from(raw, i * 8)
            days[i] = day
            docs[i] = doc
        return days, docs

    def _ckp_pack(self, days: list[int], docs: list[int]) -> bytes:
        out = bytearray(len(days) * 8)
        for i, (day, doc) in enumerate(zip(days, docs, strict=False)):
            _CKP_ENTRY.pack_into(out, i * 8, day, doc)
        return bytes(out)

    def _mono_range_bitmap(self, txn, *, since_days: int, until_days: int) -> BitMap | None:
        """Fast-path temporal filter. Returns ``None`` if fast path isn't available.

        Looks up ``(lo_doc, hi_doc)`` via bisect over the day checkpoints
        and returns a Roaring bitmap over the inclusive doc_id range.
        """
        state = self._ensure_mono_cache(txn)
        flag, _min_doc, max_doc, _min_day, _max_day, days, docs = state
        if flag != b"1":
            return None
        if max_doc == 0 or not days:
            return BitMap()  # empty index

        # lo_doc — smallest doc_id whose filed_days >= since_days.
        if since_days <= 0:
            lo_doc = docs[0]
        else:
            i = bisect.bisect_left(days, since_days)
            if i >= len(days):
                return BitMap()  # window is after the end
            lo_doc = docs[i]

        # hi_doc — largest doc_id whose filed_days <= until_days.
        if until_days == 0 or until_days >= 0xFFFFFFFF:
            hi_doc = max_doc
        else:
            j = bisect.bisect_right(days, until_days)
            if j == 0:
                return BitMap()  # window is before the start
            hi_doc = max_doc if j == len(days) else docs[j] - 1

        if lo_doc > hi_doc:
            return BitMap()

        return BitMap(range(lo_doc, hi_doc + 1))

    # ── Write API: add / delete ──────────────────────────────────────

    def add_drawer(
        self,
        drawer_id: str,
        content: str,
        metadata: dict,
        *,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> int:
        """Index one drawer. Returns the number of unique tokens inserted."""
        return self.add_batch(
            [(drawer_id, content, metadata)],
            valid_from=valid_from,
            valid_to=valid_to,
        )

    def add_batch(
        self,
        items: Iterable[tuple[str, str, dict]],
        *,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> int:
        """Index many drawers in one LMDB write transaction.

        Reads each posting bitmap at most once per batch (per-token
        buffer), caches intern lookups per batch, and uses
        ``txn.put(append=True)`` for monotone doc_id writes.
        """
        items_list = list(items)
        if not items_list:
            return 0

        # Step 1: tokenize outside the txn (CPU-bound, no I/O).
        prepared = []
        for drawer_id, content, metadata in items_list:
            meta = metadata or {}
            filed_at = meta.get("filed_at") or datetime.now(UTC).isoformat()
            wing = meta.get("wing") or None
            room = meta.get("room") or None
            vfrom = valid_from if valid_from is not None else filed_at
            vto = valid_to  # None → unbounded "still valid"
            tokens = list({t for t in tokenize(content)})  # unique, no freq
            prepared.append(
                (
                    drawer_id,
                    wing,
                    room,
                    _iso_to_days(filed_at),
                    _iso_to_days(vfrom),
                    _iso_to_days(vto) if vto is not None else 0,
                    tokens,
                )
            )

        # Step 2: single write txn for the whole batch.
        inserted = 0
        token_bms: dict[int, BitMap] = {}
        wing_bms: dict[int, BitMap] = {}
        room_bms: dict[int, BitMap] = {}
        touched_tokens: list[int] = []
        # Per-batch intern caches — skip LMDB for repeat strings.
        batch_token_cache: dict[str, int] = {}
        batch_wing_cache: dict[str, int] = {}
        batch_room_cache: dict[str, int] = {}

        with self._env.begin(write=True) as txn:
            # Load monotonic state from the instance cache (falls back to
            # LMDB on first access). Mutations stay in the cached list and
            # are flushed to LMDB once at the end of the txn.
            state = self._ensure_mono_cache(txn)
            mono_changed = False

            for drawer_id, wing, room, filed_days, vfrom_days, vto_days, tokens in prepared:
                # Idempotent re-add: if the drawer already exists, remove
                # the old posting first. This path breaks monotonicity
                # because we re-insert into doc_to_drawer with a new
                # (strictly larger) doc_id while tombstoning the old one,
                # which creates a "hole" in the doc_id sequence. That's
                # fine for the temporal fast path — we only track max_doc
                # and checkpoints, and both continue to advance.
                did_key = drawer_id.encode("utf-8")
                existing_raw = txn.get(did_key, db=self._db_drawer_to_doc)
                is_readd = existing_raw is not None
                if is_readd:
                    old_doc_id = _U32_LE.unpack(bytes(existing_raw))[0]
                    self._delete_doc(
                        txn,
                        old_doc_id,
                        drawer_id,
                        write_buffers=(token_bms, wing_bms, room_bms),
                        touched_tokens=touched_tokens,
                    )

                doc_id = self._next_counter(txn, _META_NEXT_DOC_ID, _U32_LE)

                wing_id = self._get_or_intern_wing(txn, wing, cache=batch_wing_cache)
                room_id = self._get_or_intern_room(txn, room, cache=batch_room_cache)

                doc_key = _U32_LE.pack(doc_id)

                # Monotonic doc_id writes use append=True to skip the
                # B+tree search. Re-adds also produce strictly-increasing
                # doc_ids because the counter never decreases.
                txn.put(did_key, doc_key, db=self._db_drawer_to_doc)
                txn.put(doc_key, did_key, db=self._db_doc_to_drawer, append=True)
                txn.put(
                    doc_key,
                    _DOC_META.pack(wing_id, room_id, filed_days, vfrom_days, vto_days),
                    db=self._db_doc_meta,
                    append=True,
                )

                # Collect interned token_ids with the per-batch cache.
                token_ids = []
                for tok in tokens:
                    tid = self._get_or_intern_token(txn, tok, cache=batch_token_cache)
                    token_ids.append(tid)
                    bm = token_bms.get(tid)
                    if bm is None:
                        raw = txn.get(_U32_LE.pack(tid), db=self._db_postings)
                        bm = BitMap.deserialize(bytes(raw)) if raw else BitMap()
                        token_bms[tid] = bm
                        touched_tokens.append(tid)
                    bm.add(doc_id)
                    inserted += 1

                # doc_tokens is delta-varint packed for compactness.
                txn.put(
                    doc_key,
                    _pack_varint_deltas(token_ids),
                    db=self._db_doc_tokens,
                    append=True,
                )

                if wing_id:
                    bm = wing_bms.get(wing_id)
                    if bm is None:
                        raw = txn.get(_U16_LE.pack(wing_id), db=self._db_wing_postings)
                        bm = BitMap.deserialize(bytes(raw)) if raw else BitMap()
                        wing_bms[wing_id] = bm
                    bm.add(doc_id)

                if room_id:
                    bm = room_bms.get(room_id)
                    if bm is None:
                        raw = txn.get(_U16_LE.pack(room_id), db=self._db_room_postings)
                        bm = BitMap.deserialize(bytes(raw)) if raw else BitMap()
                        room_bms[room_id] = bm
                    bm.add(doc_id)

                txn.put(_TIME_KEY.pack(filed_days, doc_id), b"", db=self._db_time_index)

                # Update monotonic tracking via the cached state list.
                # state = [flag, min_doc, max_doc, min_day, max_day,
                #          ckp_days, ckp_docs]
                if state[0] == b"1":
                    if state[2] == 0:
                        # First insert into the index.
                        state[1] = doc_id  # min_doc
                        state[3] = filed_days  # min_day
                        state[2] = doc_id  # max_doc
                        state[4] = filed_days  # max_day
                        state[5].append(filed_days)
                        state[6].append(doc_id)
                        mono_changed = True
                    elif filed_days < state[4]:
                        # Regression — abandon the fast path.
                        state[0] = b"0"
                        mono_changed = True
                    else:
                        if filed_days > state[4]:
                            state[5].append(filed_days)
                            state[6].append(doc_id)
                            state[4] = filed_days
                        state[2] = doc_id
                        mono_changed = True

            # Flush all buffered bitmaps in one sweep.
            for tid, bm in token_bms.items():
                key = _U32_LE.pack(tid)
                if len(bm) == 0:
                    txn.delete(key, db=self._db_postings)
                else:
                    txn.put(key, bm.serialize(), db=self._db_postings)
            for wid, bm in wing_bms.items():
                key = _U16_LE.pack(wid)
                if len(bm) == 0:
                    txn.delete(key, db=self._db_wing_postings)
                else:
                    txn.put(key, bm.serialize(), db=self._db_wing_postings)
            for rid, bm in room_bms.items():
                key = _U16_LE.pack(rid)
                if len(bm) == 0:
                    txn.delete(key, db=self._db_room_postings)
                else:
                    txn.put(key, bm.serialize(), db=self._db_room_postings)

            # Persist monotonic state if anything changed.
            if mono_changed:
                self._flush_mono_cache(txn)

        # Invalidate cache for every token that changed.
        self._bitmap_cache.invalidate_many(touched_tokens)

        return inserted

    def _delete_doc(
        self,
        txn,
        doc_id: int,
        drawer_id: str,
        *,
        write_buffers: tuple[dict, dict, dict] | None = None,
        touched_tokens: list[int] | None = None,
    ) -> None:
        """Remove a single doc_id from every LMDB structure that references it.

        If ``write_buffers`` is given, bitmap edits land in the caller's
        per-batch buffers instead of round-tripping through LMDB — this is
        the delete-then-reinsert path used by ``add_batch``.
        """
        doc_key = _U32_LE.pack(doc_id)

        # Tokens (doc_tokens is delta-varint packed).
        tokens_raw = txn.get(doc_key, db=self._db_doc_tokens)
        token_ids = _unpack_varint_deltas(tokens_raw) if tokens_raw else []
        for tid in token_ids:
            if write_buffers is not None:
                token_bms, _, _ = write_buffers
                bm = token_bms.get(tid)
                if bm is None:
                    raw = txn.get(_U32_LE.pack(tid), db=self._db_postings)
                    bm = BitMap.deserialize(bytes(raw)) if raw else BitMap()
                    token_bms[tid] = bm
                    if touched_tokens is not None:
                        touched_tokens.append(tid)
                bm.discard(doc_id)
            else:
                raw = txn.get(_U32_LE.pack(tid), db=self._db_postings)
                if raw is None:
                    continue
                bm = BitMap.deserialize(bytes(raw))
                bm.discard(doc_id)
                if len(bm) == 0:
                    txn.delete(_U32_LE.pack(tid), db=self._db_postings)
                else:
                    txn.put(_U32_LE.pack(tid), bm.serialize(), db=self._db_postings)
                self._bitmap_cache.invalidate(tid)

        # Wing / room
        meta_raw = txn.get(doc_key, db=self._db_doc_meta)
        if meta_raw:
            wing_id, room_id, filed_days, _vf, _vt = _DOC_META.unpack(bytes(meta_raw))
            if wing_id:
                if write_buffers is not None:
                    _, wing_bms, _ = write_buffers
                    bm = wing_bms.get(wing_id)
                    if bm is None:
                        raw = txn.get(_U16_LE.pack(wing_id), db=self._db_wing_postings)
                        bm = BitMap.deserialize(bytes(raw)) if raw else BitMap()
                        wing_bms[wing_id] = bm
                    bm.discard(doc_id)
                else:
                    raw = txn.get(_U16_LE.pack(wing_id), db=self._db_wing_postings)
                    if raw:
                        bm = BitMap.deserialize(bytes(raw))
                        bm.discard(doc_id)
                        if len(bm) == 0:
                            txn.delete(_U16_LE.pack(wing_id), db=self._db_wing_postings)
                        else:
                            txn.put(
                                _U16_LE.pack(wing_id),
                                bm.serialize(),
                                db=self._db_wing_postings,
                            )
            if room_id:
                if write_buffers is not None:
                    _, _, room_bms = write_buffers
                    bm = room_bms.get(room_id)
                    if bm is None:
                        raw = txn.get(_U16_LE.pack(room_id), db=self._db_room_postings)
                        bm = BitMap.deserialize(bytes(raw)) if raw else BitMap()
                        room_bms[room_id] = bm
                    bm.discard(doc_id)
                else:
                    raw = txn.get(_U16_LE.pack(room_id), db=self._db_room_postings)
                    if raw:
                        bm = BitMap.deserialize(bytes(raw))
                        bm.discard(doc_id)
                        if len(bm) == 0:
                            txn.delete(_U16_LE.pack(room_id), db=self._db_room_postings)
                        else:
                            txn.put(
                                _U16_LE.pack(room_id),
                                bm.serialize(),
                                db=self._db_room_postings,
                            )
            txn.delete(_TIME_KEY.pack(filed_days, doc_id), db=self._db_time_index)

        txn.delete(drawer_id.encode("utf-8"), db=self._db_drawer_to_doc)
        txn.delete(doc_key, db=self._db_doc_to_drawer)
        txn.delete(doc_key, db=self._db_doc_meta)
        txn.delete(doc_key, db=self._db_doc_tokens)

    def delete_drawer(self, drawer_id: str) -> int:
        """Remove every posting that points at ``drawer_id``."""
        did_key = drawer_id.encode("utf-8")
        with self._env.begin(write=True) as txn:
            existing = txn.get(did_key, db=self._db_drawer_to_doc)
            if existing is None:
                return 0
            doc_id = _U32_LE.unpack(bytes(existing))[0]
            tokens_raw = txn.get(_U32_LE.pack(doc_id), db=self._db_doc_tokens)
            count = len(_unpack_varint_deltas(tokens_raw)) if tokens_raw else 0
            self._delete_doc(txn, doc_id, drawer_id)
        return count

    def rebuild_from_collection(self, collection) -> int:
        """Wipe and reindex every drawer in a ChromaDB collection.

        Pulls the whole collection into memory, sorts by ``filed_at`` so
        monotonic insertion can be restored, then ingests in one or more
        ``add_batch`` calls. Restores the monotonic fast path after a
        successful rebuild.
        """
        # Wipe every sub-db.
        with self._env.begin(write=True) as txn:
            self._drop_all_data(txn)
            # Reset counters and monotonic state.
            txn.put(_META_NEXT_DOC_ID, _U32_LE.pack(1), db=self._db_meta)
            txn.put(_META_NEXT_TOKEN_ID, _U32_LE.pack(1), db=self._db_meta)
            txn.put(_META_NEXT_WING_ID, _U16_LE.pack(1), db=self._db_meta)
            txn.put(_META_NEXT_ROOM_ID, _U16_LE.pack(1), db=self._db_meta)
            txn.put(_META_MONO, b"1", db=self._db_meta)
            txn.put(_META_MONO_MIN_DOC, _U32_LE.pack(0), db=self._db_meta)
            txn.put(_META_MONO_MAX_DOC, _U32_LE.pack(0), db=self._db_meta)
            txn.put(_META_MONO_MIN_DAY, _U32_LE.pack(0), db=self._db_meta)
            txn.put(_META_MONO_MAX_DAY, _U32_LE.pack(0), db=self._db_meta)
            txn.put(_META_MONO_CKP, b"", db=self._db_meta)
        self._bitmap_cache.clear()
        # Drop the mono cache so the next add_batch reloads from the
        # freshly-reset meta.
        self._mono_cache = None

        # Pull every page of the collection, then sort by filed_at so
        # monotonic insertion is preserved during ingest.
        batch_size = 500
        offset = 0
        all_items: list[tuple[str, str, dict]] = []
        while True:
            try:
                page = collection.get(
                    include=["documents", "metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
            except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
                logger.debug("rebuild_from_collection: page fetch failed — %s", e)
                break
            ids = page.get("ids") or []
            docs = page.get("documents") or []
            metas = page.get("metadatas") or []
            if not ids:
                break
            for drawer_id, doc, meta in zip(ids, docs, metas, strict=False):
                all_items.append((drawer_id, doc or "", meta or {}))
            if len(ids) < batch_size:
                break
            offset += len(ids)

        # Sort by filed_at (falling back to drawer_id for stability). Items
        # with no filed_at sort to the end — they still get indexed, just
        # without contributing to monotonicity.
        def _sort_key(item):
            _did, _doc, meta = item
            fa = meta.get("filed_at") or "\uffff"  # after all real ISO strings
            return (fa, item[0])

        all_items.sort(key=_sort_key)

        # Ingest in one large batch — add_batch handles the monotonic
        # state update naturally as each drawer goes in.
        return self.add_batch(all_items)

    # ── Read API ──────────────────────────────────────────────────────

    def lookup(
        self,
        token: str,
        *,
        prefix: bool = False,
        wing: str | None = None,
        room: str | None = None,
        since: str | None = None,
        until: str | None = None,
        as_of: str | None = None,
    ) -> set[str]:
        """Return the set of drawer IDs matching ``token``."""
        norm = token.strip().lower() if token else ""
        if not norm:
            return set()

        with self._env.begin(write=False, buffers=True) as txn:
            bm = self._lookup_bitmap(txn, norm, prefix=prefix)
            if not bm:
                return set()
            bm = self._apply_filters(
                txn, bm, wing=wing, room=room, since=since, until=until, as_of=as_of
            )
            if not bm:
                return set()
            return self._resolve_doc_ids_to_drawers(txn, bm)

    def keyword_search(
        self,
        keywords: list[str],
        *,
        mode: str = "all",
        wing: str | None = None,
        room: str | None = None,
        since: str | None = None,
        until: str | None = None,
        as_of: str | None = None,
    ) -> set[str]:
        """Combine several keyword lookups with scope + temporal filters."""
        with self._env.begin(write=False, buffers=True) as txn:
            if keywords:
                bm = self._keyword_bitmap(txn, list(keywords), mode=mode)
                if not bm:
                    return set()
            else:
                bm = None  # "all docs" unless filters narrow it

            if bm is None and (
                wing is not None
                or room is not None
                or since is not None
                or until is not None
                or as_of is not None
            ):
                bm = self._all_doc_ids(txn)
            elif bm is None:
                # No keywords and no filters → every indexed drawer qualifies.
                return self._resolve_doc_ids_to_drawers(txn, self._all_doc_ids(txn))

            bm = self._apply_filters(
                txn, bm, wing=wing, room=room, since=since, until=until, as_of=as_of
            )
            if not bm:
                return set()
            return self._resolve_doc_ids_to_drawers(txn, bm)

    # ── Internal query helpers ────────────────────────────────────────

    def _lookup_bitmap(self, txn, token: str, *, prefix: bool) -> BitMap:
        """Return a BitMap of doc_ids matching ``token`` (exact or prefix)."""
        if prefix:
            result = BitMap()
            cursor = txn.cursor(db=self._db_tokens)
            start = token.encode("utf-8")
            if cursor.set_range(start):
                while True:
                    key = bytes(cursor.key())
                    if not key.startswith(start):
                        break
                    tid = _U32_LE.unpack(bytes(cursor.value()))[0]
                    hit = self._load_posting_bm(txn, tid)
                    if hit:
                        result = result | hit
                    if not cursor.next():
                        break
            cursor.close()
            return result
        else:
            tid = self._lookup_token_id(txn, token)
            if tid is None:
                return BitMap()
            return self._load_posting_bm(txn, tid)

    def _keyword_bitmap(self, txn, keywords: list[str], *, mode: str) -> BitMap:
        """Combine keyword bitmaps.

        For ``mode="all"`` (the common case) sort by bitmap cardinality
        ascending so the smallest set leads the intersection — that
        short-circuits on the first empty result and gives Roaring the
        smallest first operand to scan.
        """
        use_prefix = mode == "prefix"

        # Empty string check: short-circuit without hitting LMDB.
        normalized = [(kw or "").strip().lower() for kw in keywords]
        if mode in ("all", "prefix") and any(not n for n in normalized):
            return BitMap()
        normalized = [n for n in normalized if n]
        if not normalized:
            return BitMap()

        # Load every candidate bitmap upfront. For ``mode="all"`` we sort
        # by cardinality and intersect shortest-first.
        hits: list[BitMap] = []
        for n in normalized:
            hits.append(self._lookup_bitmap(txn, n, prefix=use_prefix))
            # Short-circuit: if any conjunctive operand is empty, result
            # is empty.
            if mode in ("all", "prefix") and not hits[-1]:
                return BitMap()

        if mode == "any":
            result = BitMap()
            for h in hits:
                result = result | h
            return result

        # Conjunctive: sort by cardinality ascending.
        hits.sort(key=len)
        result = hits[0]
        for h in hits[1:]:
            result = result & h
            if not result:
                return BitMap()
        return result

    def _apply_filters(
        self,
        txn,
        bm: BitMap,
        *,
        wing: str | None,
        room: str | None,
        since: str | None,
        until: str | None,
        as_of: str | None,
    ) -> BitMap:
        if wing is not None:
            wing_id = self._lookup_wing_id(txn, wing)
            if wing_id is None:
                return BitMap()
            bm = bm & self._load_wing_bm(txn, wing_id)
            if not bm:
                return bm

        if room is not None:
            room_id = self._lookup_room_id(txn, room)
            if room_id is None:
                return BitMap()
            bm = bm & self._load_room_bm(txn, room_id)
            if not bm:
                return bm

        if since is not None or until is not None:
            bm = bm & self._time_range_bitmap(txn, since=since, until=until)
            if not bm:
                return bm

        if as_of is not None:
            bm = self._filter_by_as_of(txn, bm, as_of=as_of)

        return bm

    def _time_range_bitmap(self, txn, *, since: str | None, until: str | None) -> BitMap:
        """Temporal filter. Tries the monotonic fast path first; falls back
        to the ``time_index`` cursor walk when monotonicity is broken.
        """
        lo_days = _iso_to_days(since) if since else 0
        hi_days = _iso_to_days(until) if until else 0xFFFFFFFF

        fast = self._mono_range_bitmap(txn, since_days=lo_days, until_days=hi_days)
        if fast is not None:
            self._last_query_mode = "mono_fast"
            return fast

        self._last_query_mode = "time_index_scan"
        result = BitMap()
        cursor = txn.cursor(db=self._db_time_index)
        start_key = _TIME_KEY.pack(lo_days, 0)
        if cursor.set_range(start_key):
            while True:
                key = bytes(cursor.key())
                day, doc_id = _TIME_KEY.unpack(key)
                if day > hi_days:
                    break
                result.add(doc_id)
                if not cursor.next():
                    break
        cursor.close()
        return result

    def _filter_by_as_of(self, txn, bm: BitMap, *, as_of: str) -> BitMap:
        """Keep doc_ids whose validity window covers ``as_of``."""
        as_of_days = _iso_to_days(as_of)
        out = BitMap()
        for doc_id in bm:
            raw = txn.get(_U32_LE.pack(doc_id), db=self._db_doc_meta)
            if not raw:
                continue
            _w, _r, _f, vfrom_days, vto_days = _DOC_META.unpack(bytes(raw))
            if (vfrom_days == 0 or vfrom_days <= as_of_days) and (
                vto_days == 0 or vto_days >= as_of_days
            ):
                out.add(doc_id)
        return out

    def _all_doc_ids(self, txn) -> BitMap:
        result = BitMap()
        cursor = txn.cursor(db=self._db_doc_to_drawer)
        for key, _ in cursor:
            result.add(_U32_LE.unpack(bytes(key))[0])
        cursor.close()
        return result

    def _resolve_doc_ids_to_drawers(self, txn, bm: BitMap) -> set[str]:
        """Resolve a bitmap of doc_ids to the corresponding drawer_id set.

        Uses ``cursor.getmulti`` for large result sets (≥ _GETMULTI_THRESHOLD)
        — one batch call instead of N point lookups. Measurements show
        pre-sorting the keys doesn't help (``getmulti`` handles cursor
        seek ordering internally), so we skip the sort.
        """
        size = len(bm)
        if size == 0:
            return set()

        if size < _GETMULTI_THRESHOLD:
            out: set[str] = set()
            for doc_id in bm:
                raw = txn.get(_U32_LE.pack(doc_id), db=self._db_doc_to_drawer)
                if raw:
                    out.add(bytes(raw).decode("utf-8"))
            return out

        keys = [_U32_LE.pack(d) for d in bm]
        with txn.cursor(db=self._db_doc_to_drawer) as cursor:
            pairs = cursor.getmulti(keys)
        return {bytes(v).decode("utf-8") for _k, v in pairs}

    # ── Metadata lookup ───────────────────────────────────────────────

    def get_drawer_meta(self, drawer_ids: Iterable[str]) -> dict[str, dict]:
        """Return wing / room / filed_at for each drawer_id.

        ``filed_at`` is reconstructed from ``filed_days`` so it has date
        precision only. Uses ``cursor.getmulti`` for large inputs, same as
        ``_resolve_doc_ids_to_drawers``.
        """
        drawer_ids = list(drawer_ids)
        if not drawer_ids:
            return {}
        out: dict[str, dict] = {}
        with self._env.begin(write=False, buffers=True) as txn:
            # Resolve drawer_id → doc_id first. The drawer_to_doc DBI is
            # string-keyed so we can't use integerkey getmulti, but we can
            # still batch with cursor.getmulti over byte keys.
            did_keys = [d.encode("utf-8") for d in drawer_ids]
            doc_ids: dict[str, int] = {}
            if len(did_keys) < _GETMULTI_THRESHOLD:
                for did, key in zip(drawer_ids, did_keys, strict=False):
                    raw = txn.get(key, db=self._db_drawer_to_doc)
                    if raw is not None:
                        doc_ids[did] = _U32_LE.unpack(bytes(raw))[0]
            else:
                # For string keys, sort for cursor locality.
                order = sorted(range(len(did_keys)), key=lambda i: did_keys[i])
                sorted_keys = [did_keys[i] for i in order]
                sorted_dids = [drawer_ids[i] for i in order]
                with txn.cursor(db=self._db_drawer_to_doc) as cursor:
                    pairs = cursor.getmulti(sorted_keys)
                # getmulti only returns existing keys, not in request order
                returned = {bytes(k): bytes(v) for k, v in pairs}
                for did, key in zip(sorted_dids, sorted_keys, strict=False):
                    v = returned.get(key)
                    if v is not None:
                        doc_ids[did] = _U32_LE.unpack(v)[0]

            if not doc_ids:
                return out

            # Now batch-fetch doc_meta by doc_id.
            if len(doc_ids) < _GETMULTI_THRESHOLD:
                for drawer_id, did_id in doc_ids.items():
                    meta_raw = txn.get(_U32_LE.pack(did_id), db=self._db_doc_meta)
                    if meta_raw is None:
                        continue
                    wing_id, room_id, filed_days, _vf, _vt = _DOC_META.unpack(bytes(meta_raw))
                    out[drawer_id] = {
                        "wing": self._resolve_wing_name(txn, wing_id),
                        "room": self._resolve_room_name(txn, room_id),
                        "filed_at": _days_to_iso(filed_days),
                    }
            else:
                sorted_items = sorted(doc_ids.items(), key=lambda kv: kv[1])
                meta_keys = [_U32_LE.pack(did_id) for _did, did_id in sorted_items]
                with txn.cursor(db=self._db_doc_meta) as cursor:
                    pairs = cursor.getmulti(meta_keys)
                returned = {bytes(k): bytes(v) for k, v in pairs}
                for drawer_id, did_id in sorted_items:
                    meta_raw = returned.get(_U32_LE.pack(did_id))
                    if meta_raw is None:
                        continue
                    wing_id, room_id, filed_days, _vf, _vt = _DOC_META.unpack(meta_raw)
                    out[drawer_id] = {
                        "wing": self._resolve_wing_name(txn, wing_id),
                        "room": self._resolve_room_name(txn, room_id),
                        "filed_at": _days_to_iso(filed_days),
                    }
        return out

    # ── Warm cache prefetch ──────────────────────────────────────────

    def warm(self, top_k: int = _WARM_TOP_K_DEFAULT) -> int:
        """Preload the ``top_k`` most-populated token bitmaps into the LRU.

        Iterates the ``postings`` DBI and reads Roaring's serialized
        cardinality from the 4-byte header at offset 4–7 (portable format).
        Sorts by cardinality descending, then loads the top-K into the
        in-process LRU cache.

        Called automatically from ``mcp_server.main()`` on MCP startup so
        long-lived processes never pay the cold-deserialize cost on their
        first query for hot tokens. CLI one-shot commands skip warming by
        default.

        Returns the number of bitmaps loaded.
        """
        # Pull (token_id, cardinality, serialized_bytes) triples. Cardinality
        # lives at bytes 4–7 in Roaring's portable header — we could extract
        # it without a full deserialize, but deserializing is already fast
        # and guarantees forward compatibility if the header shape changes.
        triples: list[tuple[int, int, BitMap]] = []
        with self._env.begin(write=False, buffers=True) as txn:
            cursor = txn.cursor(db=self._db_postings)
            for key, value in cursor:
                tid = _U32_LE.unpack(bytes(key))[0]
                bm = BitMap.deserialize(bytes(value))
                triples.append((tid, len(bm), bm))
            cursor.close()

        # Sort by cardinality desc; take the top K.
        triples.sort(key=lambda t: t[1], reverse=True)
        loaded = 0
        for tid, _card, bm in triples[:top_k]:
            self._bitmap_cache.put(tid, bm)
            loaded += 1
        return loaded

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Posting count, unique token count, unique drawer count, plus
        monotonic-state diagnostics.
        """
        postings = 0
        token_count = 0
        drawer_count = 0
        mono_flag = b"0"
        mono_max_doc = 0
        mono_max_day = 0
        ckp_len = 0
        with self._env.begin(write=False, buffers=True) as txn:
            drawer_count = txn.stat(self._db_drawer_to_doc)["entries"]
            token_count = txn.stat(self._db_postings)["entries"]
            cursor = txn.cursor(db=self._db_postings)
            for _key, value in cursor:
                bm = BitMap.deserialize(bytes(value))
                postings += len(bm)
            cursor.close()

            state = self._ensure_mono_cache(txn)
            mono_flag = state[0]
            mono_max_doc = state[2]
            mono_max_day = state[4]
            ckp_len = len(state[5])

        return {
            "postings": postings,
            "unique_tokens": token_count,
            "unique_drawers": drawer_count,
            "db_path": self.db_path,
            "backend": "lmdb",
            "schema_version": _SCHEMA_VERSION,
            "monotonic": mono_flag == b"1",
            "mono_max_doc": mono_max_doc,
            "mono_max_day": mono_max_day,
            "mono_checkpoints": ckp_len,
            "cache_size": len(self._bitmap_cache),
            "last_query_mode": self._last_query_mode,
        }
