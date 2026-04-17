# Closets — Planned Feature

> **Status: not implemented in v3.** This doc describes the roadmap
> design for closets. The current release stores drawers (verbatim
> chunks) in the `mempalace_drawers` collection; there is no
> `mempalace_closets` collection and no closet-aware code path in
> `searcher.py`. See [CLAUDE.md](../CLAUDE.md) for the architecture
> that is actually shipping today, and the README's roadmap section
> for delivery status.

## What closets will be

Drawers hold your verbatim content. **Closets** are the planned index
layer — compact, searchable summaries that point at one or more
drawers:

```
CLOSET: "built auth system|Ben;Igor|→drawer_api_auth_a1b2c3"
         ↑ topic           ↑ entities  ↑ points to this drawer
```

The intent is that an agent searching "who built the auth?" hits the
closet first (a fast scan of short summary text), then opens the
referenced drawer to get the verbatim content. This gives smaller
embedding payloads for the index pass while preserving the
"verbatim-first" guarantee in the data the agent ultimately reads.

## Why it isn't shipped yet

The benchmark numbers MemPalace publishes (~96.6% LongMemEval recall)
come from **raw drawer search** over `mempalace_drawers`. Closet-based
retrieval has not yet beaten that baseline in a way that justifies the
extra ingestion cost, so the closet writer / reader was held back. The
feature is tracked on the roadmap so the wing/room/closet/drawer
vocabulary remains a coherent design even though only the drawer rung
of the ladder is wired up.

What does exist today:

- The taxonomy is documented (wings → halls → rooms → closets →
  drawers) — see CLAUDE.md.
- The AAAK Dialect (`mempalace/dialect.py`) is shippable as a
  standalone summarizer, ready to encode closet payloads when the
  collection lands.
- The trie index (`mempalace/trie_index.py`) provides keyword and
  temporal prefilters over drawers — closets, when they ship, will
  reuse the same secondary index.

What is **not** in the codebase:

- No `mempalace_closets` ChromaDB collection.
- No `get_closets_collection`, `build_closet_lines`,
  `upsert_closet_lines`, or `purge_file_closets` helpers in
  `palace.py`.
- No `_extract_drawer_ids_from_closet` or `_closet_first_hits`
  branch in `searcher.py`.
- The `mempalace compress` command writes AAAK-compressed drawers to
  a `mempalace_compressed` collection (see CLAUDE.md), but that is a
  separate offline step — not the live closet index this doc
  describes.

## Planned shape (subject to change)

Each closet line will be one atomic topic pointer. The same
`source_file` will be allowed to span multiple closets when a single
file produces more topics than fit in one summary, and re-mining a
file will purge that file's closets before writing a fresh set so
stale topics never accumulate. Search will fall back to direct drawer
search whenever closets are missing or filtered out, so closets will
always be a ranking signal — never a gate.

Once the work lands, this doc will be rewritten to describe the live
implementation; the test in `tests/test_readme_claims.py` that
verifies `searcher.py` imports `get_closets_collection` is the
landing-day signal.
