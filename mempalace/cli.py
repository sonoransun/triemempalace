#!/usr/bin/env python3
"""
MemPalace — Give your AI a memory. No API key required.

Two ways to ingest:
  Projects:      mempalace mine ~/projects/my_app          (code, docs, notes)
  Conversations: mempalace mine ~/chats/ --mode convos     (Claude, ChatGPT, Slack)

Same palace. Same search. Different ingest strategies.

Commands:
    mempalace init <dir>                  Detect rooms from folder structure
    mempalace split <dir>                 Split concatenated mega-files into per-session files
    mempalace mine <dir>                  Mine project files (default)
    mempalace mine <dir> --mode convos    Mine conversation exports
    mempalace search "query"              Find anything, exact words
    mempalace mcp                         Show MCP setup command
    mempalace wake-up                     Show L0 + L1 wake-up context
    mempalace wake-up --wing my_app       Wake-up for a specific project
    mempalace status                      Show what's been filed

Examples:
    mempalace init ~/projects/my_app
    mempalace mine ~/projects/my_app
    mempalace mine ~/chats/claude-sessions --mode convos
    mempalace search "why did we switch to GraphQL"
    mempalace search "pricing discussion" --wing my_app --room costs
"""

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path

import lmdb

from .config import MempalaceConfig

logger = logging.getLogger("mempalace.cli")


def cmd_init(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from .entity_detector import confirm_entities, detect_entities, scan_for_detection
    from .room_detector_local import detect_rooms_local

    # Pass 1: auto-detect people and projects from file content
    print(f"\n  Scanning for entities in: {args.dir}")
    files = scan_for_detection(args.dir)
    if files:
        print(f"  Reading {len(files)} files...")
        detected = detect_entities(files)
        total = len(detected["people"]) + len(detected["projects"]) + len(detected["uncertain"])
        if total > 0:
            confirmed = confirm_entities(detected, yes=getattr(args, "yes", False))
            # Save confirmed entities to <project>/entities.json for the miner
            if confirmed["people"] or confirmed["projects"]:
                entities_path = Path(args.dir).expanduser().resolve() / "entities.json"
                with open(entities_path, "w") as f:
                    json.dump(confirmed, f, indent=2)
                print(f"  Entities saved: {entities_path}")
        else:
            print("  No entities detected — proceeding with directory-based rooms.")

    # Pass 2: detect rooms from folder structure
    detect_rooms_local(project_dir=args.dir, yes=getattr(args, "yes", False))
    MempalaceConfig().init()


def cmd_mine(args: argparse.Namespace) -> None:
    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )
    include_ignored = []
    for raw in args.include_ignored or []:
        include_ignored.extend(part.strip() for part in raw.split(",") if part.strip())

    model = getattr(args, "model", None)
    if model == "all":
        print("  mine: --model all is read-only. Pick a concrete slug for writes.")
        sys.exit(2)

    if args.mode == "convos":
        from .convo_miner import mine_convos

        mine_convos(
            convo_dir=args.dir,
            palace_path=palace_path,
            wing=args.wing,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            extract_mode=args.extract,
            model=model,
        )
    else:
        from .miner import mine

        mine(
            project_dir=args.dir,
            palace_path=palace_path,
            wing_override=args.wing,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            respect_gitignore=not args.no_gitignore,
            include_ignored=include_ignored,
            model=model,
        )

    # ── Optional: knowledge-graph extraction post-step ──────────────
    if getattr(args, "extract_kg", False) and not args.dry_run:
        from .kg_extract import extract_from_palace

        kg_mode = getattr(args, "kg_extract_mode", "heuristic")
        kg_model = getattr(args, "kg_extract_model", "llama3.1:8b")
        print(f"\n  Running KG extraction ({kg_mode})…")
        try:
            stats = extract_from_palace(
                palace_path,
                mode=kg_mode,
                model=kg_model,
            )
            print(
                f"  KG extraction: scanned {stats['drawers_scanned']} drawers, "
                f"added {stats['triples_added']} triples "
                f"({stats['errors']} errors)"
            )
        except Exception as e:
            # Never crash the mine for a KG extraction failure —
            # drawers are already persisted.
            print(f"  KG extraction skipped: {e}")
            logger.warning("cmd_mine: KG extraction failed — %s", e)


def cmd_search(args: argparse.Namespace) -> None:
    from .searcher import SearchError, search
    from .trie_index import TrieIndex, trie_db_path

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )

    # --keyword and --keyword-prefix flags flatten into a single list; if
    # any prefix keywords are present, we switch the whole query into
    # prefix mode. Mixing exact + prefix in one call is not supported for
    # now (keeps the precedence story simple).
    exact = list(args.keyword or [])
    prefix = list(args.keyword_prefix or [])
    if exact and prefix:
        print("  --keyword and --keyword-prefix cannot be combined in one query.")
        sys.exit(2)
    keywords = exact + prefix
    keyword_mode = "prefix" if prefix else "all"

    # Optional warm-up: pre-load hot posting bitmaps into the in-process
    # LRU. Useful for interactive sessions that plan to run many queries;
    # pointless for one-shot CLI invocations.
    if getattr(args, "warm_trie", False):
        try:
            trie_path = trie_db_path(palace_path)
            if Path(trie_path).is_dir():
                loaded = TrieIndex(db_path=trie_path).warm()
                print(f"  Trie warm: loaded {loaded} hot bitmaps")
        except (lmdb.Error, OSError, ValueError) as e:
            print(f"  Trie warm skipped: {e}")

    try:
        search(
            query=args.query,
            palace_path=palace_path,
            wing=args.wing,
            room=args.room,
            n_results=args.results,
            keywords=keywords or None,
            keyword_mode=keyword_mode,
            since=args.since,
            until=args.until,
            as_of=args.as_of,
            model=getattr(args, "model", None),
            compress=getattr(args, "compress", "auto"),
            token_budget=getattr(args, "token_budget", None),
            dup_threshold=getattr(args, "dup_threshold", 0.7),
            sent_threshold=getattr(args, "sent_threshold", 0.75),
            novelty_threshold=getattr(args, "novelty_threshold", 0.2),
            rerank=(None if getattr(args, "rerank", "none") == "none" else args.rerank),
            rerank_prune=getattr(args, "rerank_prune", True),
            enable_kg_ppr=getattr(args, "enable_kg_ppr", False),
        )
    except SearchError:
        sys.exit(1)


def cmd_wakeup(args: argparse.Namespace) -> None:
    """Show L0 (identity) + L1 (essential story) — the wake-up context."""
    from .layers import MemoryStack

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )
    stack = MemoryStack(palace_path=palace_path)

    text = stack.wake_up(wing=args.wing)
    tokens = len(text) // 4
    print(f"Wake-up text (~{tokens} tokens):")
    print("=" * 50)
    print(text)


def cmd_split(args: argparse.Namespace) -> None:
    """Split concatenated transcript mega-files into per-session files."""
    import sys

    from .split_mega_files import main as split_main

    # Rebuild argv for split_mega_files argparse
    # Expand ~ and resolve to absolute path so split_mega_files sees a real path
    argv = ["--source", str(Path(args.dir).expanduser().resolve())]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.dry_run:
        argv.append("--dry-run")
    if args.min_sessions != 2:
        argv += ["--min-sessions", str(args.min_sessions)]

    old_argv = sys.argv
    sys.argv = ["mempalace split"] + argv
    try:
        split_main()
    finally:
        sys.argv = old_argv


def cmd_status(args: argparse.Namespace) -> None:
    from .miner import status
    from .trie_index import TrieIndex, trie_db_path

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )
    status(palace_path=palace_path)

    # Trie index stats — cheap and helpful for verifying the secondary
    # index is populated alongside the Chroma drawers.
    trie_db = trie_db_path(palace_path)
    if Path(trie_db).exists():
        try:
            stats = TrieIndex(db_path=trie_db).stats()
            print(
                f"  Trie:   {stats['postings']:,} postings, "
                f"{stats['unique_tokens']:,} unique tokens, "
                f"{stats['unique_drawers']:,} drawers indexed"
            )
            print(f"  Trie DB: {trie_db}\n")
        except (lmdb.Error, OSError, ValueError) as e:
            print(f"  Trie:   (error reading {trie_db}: {e})\n")


def cmd_repair(args: argparse.Namespace) -> None:
    """Rebuild the Chroma vector index and the LMDB trie index.

    Reads every drawer out of the existing ``mempalace_drawers``
    collection, creates a ``<palace>.backup`` snapshot, drops and
    recreates the collection, re-files every drawer, then hands the
    fresh collection to ``TrieIndex.rebuild_from_collection`` so the
    keyword/temporal index matches the vectors.
    """
    import shutil

    import chromadb.errors

    from .aggregates import hydrate_drawer_metadata
    from .palace_io import delete_collection, drop_collection_cache, open_collection

    config = MempalaceConfig()
    palace_path = str(Path(args.palace).expanduser()) if args.palace else config.palace_path
    db_path = str(Path(palace_path) / "chroma.sqlite3")

    if not Path(palace_path).is_dir():
        print(f"\n  No palace found at {palace_path}")
        return
    if not Path(db_path).is_file():
        print(f"\n  No palace database found at {db_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    # Backup once for the whole palace, before touching any model collection.
    palace_path = palace_path.rstrip(os.sep)
    backup_path = palace_path + ".backup"
    if Path(backup_path).exists():
        shutil.rmtree(backup_path)
    print(f"  Backing up to {backup_path}...")
    shutil.copytree(palace_path, backup_path)

    last_rebuilt_col = None
    grand_total = 0
    for slug in config.enabled_embedding_models:
        # Read existing drawers for this model.
        try:
            col = open_collection(palace_path, model=slug)
            total = col.count()
        except (OSError, chromadb.errors.ChromaError, ValueError) as e:
            print(f"  [{slug}] skipped: {e}")
            continue

        print(f"\n  [{slug}] {total} drawers")
        if total == 0:
            continue

        batch_size = 5000
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict] = []
        offset = 0
        while offset < total:
            batch = col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
            all_ids.extend(batch["ids"])
            all_docs.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += batch_size

        # Drop and recreate this model's collection. drop_collection_cache
        # invalidates the read handle so the create=True call below opens
        # a fresh one.
        delete_collection(palace_path, model=slug)
        drop_collection_cache(palace_path)
        new_col = open_collection(palace_path, model=slug, create=True)

        # Backfill ``hall`` on every drawer during re-file so old palaces
        # gain the new metadata field without a separate pass. Preserves
        # any pre-set hall (e.g. ``hall_diary`` from diary entries).
        backfilled = 0
        for j, meta in enumerate(all_metas):
            if isinstance(meta, dict) and not meta.get("hall"):
                hydrate_drawer_metadata(meta, all_docs[j] or "")
                backfilled += 1

        filed = 0
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            batch_docs = all_docs[i : i + batch_size]
            batch_metas = all_metas[i : i + batch_size]
            new_col.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
            filed += len(batch_ids)
            print(f"  [{slug}] re-filed {filed}/{len(all_ids)} drawers...")
        if backfilled:
            print(f"  [{slug}] backfilled hall metadata on {backfilled} drawer(s)")

        grand_total += filed
        last_rebuilt_col = new_col

    if grand_total == 0:
        print("  Nothing to repair.")
        return

    # Rebuild the trie index from the default-model collection so the
    # keyword/temporal side stays consistent with the vectors. The trie
    # is embedding-agnostic so we only rebuild it once.
    try:
        from .trie_index import TrieIndex, trie_db_path

        trie = TrieIndex(db_path=trie_db_path(palace_path))
        trie_count = trie.rebuild_from_collection(last_rebuilt_col)
        print(f"  Trie: reindexed {trie_count:,} postings from {grand_total} drawers")
        _remove_legacy_sqlite_trie(palace_path)
    except Exception as e:
        # Broad catch: trie rebuild involves lmdb (lmdb.Error), chromadb
        # (ChromaError), and filesystem (OSError). The CLI prints a single
        # warning and continues — the repair completes even if the trie
        # side fails.
        print(f"  Trie rebuild skipped: {e}")

    # Rebuild the hierarchical aggregate collections. Uses the just-
    # rebuilt drawer metadata (including the backfilled ``hall`` field)
    # so every wing/hall/room that contains at least one drawer ends
    # up with a fresh aggregate embedding. Gated by --rebuild-aggregates
    # (default on); users who don't want aggregates can skip the pass.
    if getattr(args, "rebuild_aggregates", True):
        try:
            from . import aggregates as _agg_module

            print("\n  Rebuilding wing/hall/room aggregate embeddings...")
            agg_result = _agg_module.rebuild_all(palace_path)
            print(
                f"  Aggregates: {agg_result['total']} containers "
                f"(wings={agg_result['by_level']['wing']} "
                f"halls={agg_result['by_level']['hall']} "
                f"rooms={agg_result['by_level']['room']}) "
                f"across {len(agg_result['slugs'])} model(s)"
            )
        except Exception as e:
            print(f"  Aggregate rebuild skipped: {e}")

    print(f"\n  Repair complete. {grand_total} drawers rebuilt.")
    print(f"  Backup saved at {backup_path}")
    print(f"\n{'=' * 55}\n")


def _remove_legacy_sqlite_trie(palace_path: str) -> None:
    """Delete any pre-LMDB ``trie_index.sqlite3`` file left behind by an
    older MemPalace version.

    Safe to call repeatedly — no-op when the file is absent. Keeps the
    palace directory tidy after the migration to LMDB.
    """
    legacy = str(Path(palace_path) / "trie_index.sqlite3")
    if Path(legacy).exists() and not Path(legacy).is_dir():
        try:
            os.remove(legacy)
            print(f"  Removed legacy trie file: {legacy}")
        except OSError as e:
            print(f"  Could not remove legacy trie file {legacy}: {e}")


def cmd_aggregates(args: argparse.Namespace) -> None:
    """Dispatch the ``mempalace aggregates`` sub-action.

    Sub-actions:
      rebuild      — recompute and upsert wing/hall/room aggregate
                     embeddings for every dirty container (or all
                     containers with ``--all``).
      status       — show dirty/clean counts per level and the most
                     recent rebuild timestamp.
    """
    from . import aggregates as _agg

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )
    action = getattr(args, "aggregates_action", None)

    if action == "status":
        dirty = _agg.list_dirty(palace_path)
        last = _agg.latest_rebuilt_any(palace_path)
        cfg = MempalaceConfig()
        print(f"\n{'=' * 55}")
        print("  MemPalace Aggregates — Status")
        print(f"{'=' * 55}")
        print(f"  Palace:   {palace_path}")
        print(f"  Enabled:  {cfg.aggregate_enabled}")
        print(f"  Top-K:    {cfg.aggregate_top_k}")
        print(f"  Threshold:{cfg.aggregate_rebuild_threshold}")
        print(f"  Weights:  {cfg.aggregate_weights}")
        print("  Dirty:")
        for level in _agg.LEVELS:
            print(f"    {level:8} {len(dirty.get(level, [])):4}")
        print(f"  Last rebuild: {last or 'never'}")
        print(f"{'=' * 55}\n")
        return

    if action == "rebuild":
        slug = getattr(args, "model", None)
        slugs = [slug] if slug else None

        specific = []
        for level, attr in (("wing", "wing"), ("hall", "hall"), ("room", "room")):
            value = getattr(args, attr, None)
            if value:
                specific.append((level, value))

        if getattr(args, "all", False):
            print("  Rebuilding every container in the palace...")
            if getattr(args, "dry_run", False):
                dirty = _agg.list_dirty(palace_path)
                print(
                    f"  (dry-run) would rebuild dirty={sum(len(v) for v in dirty.values())} "
                    "plus every container reachable via drawer metadata."
                )
                return
            result = _agg.rebuild_all(palace_path, slugs=slugs)
        elif specific:
            if getattr(args, "dry_run", False):
                print(f"  (dry-run) would rebuild containers: {specific}")
                return
            by_level: dict[str, int] = {level: 0 for level in _agg.LEVELS}
            slugs_list = slugs or (MempalaceConfig().enabled_embedding_models or ["default"])
            for level, container in specific:
                for s in slugs_list:
                    by_level[level] += _agg.rebuild_containers(
                        palace_path, level=level, containers=[container], slug=s
                    )
                _agg.clear_dirty(palace_path, level, [container])
            result = {"by_level": by_level, "total": sum(by_level.values()), "slugs": slugs_list}
        else:
            dirty = _agg.list_dirty(palace_path)
            total_dirty = sum(len(v) for v in dirty.values())
            if total_dirty == 0:
                print("  No dirty containers. Nothing to rebuild.")
                return
            if getattr(args, "dry_run", False):
                print(f"  (dry-run) would rebuild {total_dirty} dirty containers.")
                for level in _agg.LEVELS:
                    print(f"    {level}: {dirty.get(level, [])}")
                return
            result = _agg.rebuild_dirty(palace_path, slugs=slugs)

        print(f"\n  Rebuilt: {result['total']}")
        for level, n in result["by_level"].items():
            print(f"    {level:8} {n}")
        print(f"  Slugs: {', '.join(result['slugs'])}")
        return

    # No action specified — print usage hint.
    print("  Usage: mempalace aggregates {rebuild|status}")
    sys.exit(2)


def cmd_models(args: argparse.Namespace) -> None:
    """Dispatch the ``mempalace models`` sub-action.

    Sub-actions:
      list         — print every spec with installed/enabled/default/drawers columns
      download     — eagerly load a model's weights (triggers the backend download)
      enable       — add a slug to config.enabled_embedding_models
      disable      — remove a slug from config.enabled_embedding_models
      set-default  — set config.default_embedding_model
    """
    from . import embeddings as _embeddings
    from .palace_io import open_collection

    action = getattr(args, "models_action", None)
    cfg = MempalaceConfig()

    if action == "list" or action is None:
        enabled = set(cfg.enabled_embedding_models)
        default_slug = cfg.default_embedding_model
        print()
        print(f"  {'SLUG':22} {'BACKEND':22} {'INST':5} {'ENAB':5} {'DRAWERS':8} NAME")
        print(f"  {'-' * 22} {'-' * 22} {'-' * 5} {'-' * 5} {'-' * 8} {'-' * 30}")
        for spec in _embeddings.list_specs():
            drawers = 0
            if spec.slug in enabled:
                try:
                    col = open_collection(cfg.palace_path, model=spec.slug)
                    drawers = col.count() if col is not None else 0
                except Exception as e:
                    # Broad catch: any model-specific open failure is
                    # reported as "0 drawers". The list view tolerates
                    # partial data.
                    logger.debug("cmd_models list: %s count failed — %s", spec.slug, e)
                    drawers = 0
            inst = "YES" if _embeddings.is_installed(spec) else "no"
            enab = "YES" if spec.slug in enabled else "no"
            star = " *" if spec.slug == default_slug else "  "
            print(
                f"  {spec.slug:22} {spec.backend:22} {inst:5} {enab:5} "
                f"{drawers:8} {spec.display_name}{star}"
            )
        print()
        print(f"  default model: {default_slug}")
        print(f"  enabled: {', '.join(sorted(enabled))}")
        print()
        return

    # All the other actions take a slug argument.
    slug = getattr(args, "slug", None)
    if not slug:
        print("  missing slug. See `mempalace models list` for options.")
        sys.exit(2)

    try:
        spec = _embeddings.get_spec(slug)
    except KeyError as e:
        print(f"  {e}")
        sys.exit(2)

    if action == "download":
        if not _embeddings.is_installed(spec):
            extras = ", ".join(spec.extras_required)
            print(
                f"  {slug}: install extras first — "
                f"pip install 'mempalace[embeddings-{spec.backend}]' "
                f"({extras})"
            )
            sys.exit(2)
        print(f"  downloading weights for {slug} ({spec.model_id})...")
        try:
            fn = _embeddings.load_embedding_function(slug)
            if fn is None:
                print("  (chroma default — nothing to download)")
                return
            # Force the lazy init by computing one tiny embedding.
            fn(["warmup"])
            print(f"  ok — {slug} ready.")
        except Exception as e:
            # Broad catch: the download path can fail with network errors,
            # HuggingFace auth errors, ollama errors, onnx runtime errors,
            # or out-of-disk — each backend surfaces its own exception
            # hierarchy. Users just need the message.
            print(f"  download failed: {e}")
            sys.exit(1)
        return

    if action == "enable":
        enabled = list(cfg.enabled_embedding_models)
        if slug not in enabled:
            enabled.append(slug)
        cfg.save_embedding_config(enabled=enabled)
        print(f"  enabled: {', '.join(enabled)}")
        return

    if action == "disable":
        if slug == "default":
            print("  cannot disable 'default' — it's the legacy fallback.")
            sys.exit(2)
        enabled = [s for s in cfg.enabled_embedding_models if s != slug]
        cfg.save_embedding_config(enabled=enabled)
        print(f"  enabled: {', '.join(enabled)}")
        return

    if action == "set-default":
        enabled = list(cfg.enabled_embedding_models)
        if slug not in enabled:
            enabled.append(slug)
        cfg.save_embedding_config(default=slug, enabled=enabled)
        print(f"  default: {slug}")
        return

    print(f"  unknown action: {action}")
    sys.exit(2)


def cmd_kg_extract(args: argparse.Namespace) -> None:
    """Run knowledge-graph extraction retroactively on an already-mined palace.

    Walks every drawer in the Chroma collection and feeds its content
    through the chosen extractor (heuristic or Ollama). Each extracted
    triple is added to ``knowledge_graph.sqlite3`` via
    ``KnowledgeGraph.add_triple``, which dedupes on
    ``(subject, predicate, object)`` and merges the source_closet
    drawer IDs so the same triple asserted by many drawers becomes
    one row with a JSON list of sources and a saturated confidence.

    Safe to re-run: idempotent per drawer because the KG path dedupes.
    """
    from .kg_extract import extract_from_palace

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )

    if not Path(palace_path).is_dir():
        print(f"\n  No palace found at {palace_path}")
        sys.exit(1)

    mode = getattr(args, "mode", "heuristic")
    model = getattr(args, "model", "llama3.1:8b")

    print(f"\n  Extracting KG triples from {palace_path}")
    print(f"  Mode: {mode}" + (f" ({model})" if mode == "ollama" else ""))
    print()

    stats = extract_from_palace(palace_path, mode=mode, model=model)
    print(f"  Drawers scanned:  {stats['drawers_scanned']:,}")
    print(f"  Triples added:    {stats['triples_added']:,}")
    print(f"  Errors:           {stats['errors']:,}")
    print()


def cmd_rerankers(args: argparse.Namespace) -> None:
    """Dispatch the ``mempalace rerankers`` sub-action.

    Sub-actions:
      list  — print every registered reranker with install status +
              whether it supports per-token pruning.

    Kept small on purpose — rerankers are a read-time feature and
    don't have enable/disable/set-default semantics like embeddings do.
    Users pick a reranker per ``mempalace search --rerank <slug>``
    call or set ``default_rerank_mode`` directly in config.json.
    """
    from . import rerank as _rerank_module

    action = getattr(args, "rerankers_action", None)

    if action == "list" or action is None:
        print()
        print(f"  {'SLUG':12} {'BACKEND':24} {'INST':5} {'PRUNE':6}  DISPLAY NAME")
        print(f"  {'-' * 12} {'-' * 24} {'-' * 5} {'-' * 6}  {'-' * 40}")
        for spec in _rerank_module.list_reranker_specs():
            inst = "YES" if _rerank_module.is_installed(spec) else "no"
            prune = "YES" if spec.supports_pruning else "no"
            print(f"  {spec.slug:12} {spec.backend:24} {inst:5} {prune:6}  {spec.display_name}")
        print()
        print(
            "  Use --rerank <slug> on `mempalace search` or set "
            "default_rerank_mode in ~/.mempalace/config.json."
        )
        print()
        return

    print(f"  unknown rerankers action: {action}")
    sys.exit(2)


def cmd_trie_repair(args: argparse.Namespace) -> None:
    """Rebuild the trie index from the existing Chroma collection.

    Unlike ``cmd_repair`` this does not touch the ChromaDB collection or
    create a palace backup — it only rewrites the LMDB env at
    ``<palace>/trie_index.lmdb`` from whatever is currently in
    ``mempalace_drawers``. Use it after upgrading from a palace that was
    mined before the trie existed or with the previous SQLite backend.
    """
    import chromadb.errors

    from .palace_io import open_collection
    from .trie_index import TrieIndex, trie_db_path

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )

    if not Path(palace_path).is_dir():
        print(f"\n  No palace found at {palace_path}")
        return

    try:
        col = open_collection(palace_path)
    except (OSError, chromadb.errors.ChromaError, ValueError) as e:
        print(f"\n  Could not open palace: {e}")
        return

    total_drawers = col.count()
    print(f"\n{'=' * 55}")
    print("  MemPalace Trie Repair")
    print(f"{'=' * 55}")
    print(f"  Palace:  {palace_path}")
    print(f"  Drawers: {total_drawers:,}")
    print(f"{'-' * 55}\n")

    trie = TrieIndex(db_path=trie_db_path(palace_path))
    inserted = trie.rebuild_from_collection(col)
    stats = trie.stats()

    print(f"  Postings filed:  {inserted:,}")
    print(f"  Unique tokens:   {stats['unique_tokens']:,}")
    print(f"  Unique drawers:  {stats['unique_drawers']:,}")
    print(f"\n  Done. Trie index at {stats['db_path']}")

    # Clean up any leftover SQLite file from the previous backend.
    _remove_legacy_sqlite_trie(palace_path)

    print(f"{'=' * 55}\n")


def cmd_hook(args: argparse.Namespace) -> None:
    """Run hook logic: reads JSON from stdin, outputs JSON to stdout."""
    from .hooks_cli import run_hook

    run_hook(hook_name=args.hook, harness=args.harness)


def cmd_instructions(args: argparse.Namespace) -> None:
    """Output skill instructions to stdout."""
    from .instructions_cli import run_instructions

    run_instructions(name=args.name)


def cmd_mcp(args):
    """Show how to wire MemPalace into MCP-capable hosts."""
    base_server_cmd = "python -m mempalace.mcp_server"

    if args.palace:
        resolved_palace = str(Path(args.palace).expanduser())
        server_cmd = f"{base_server_cmd} --palace {shlex.quote(resolved_palace)}"
    else:
        server_cmd = base_server_cmd

    print("MemPalace MCP quick setup:")
    print(f"  claude mcp add mempalace -- {server_cmd}")
    print("\nRun the server directly:")
    print(f"  {server_cmd}")

    if not args.palace:
        print("\nOptional custom palace:")
        print(f"  claude mcp add mempalace -- {base_server_cmd} --palace /path/to/palace")
        print(f"  {base_server_cmd} --palace /path/to/palace")


def cmd_compress(args):
    """Compress drawers in a wing using AAAK Dialect."""
    import chromadb.errors

    from .dialect import Dialect
    from .palace_io import open_collection

    palace_path = (
        str(Path(args.palace).expanduser()) if args.palace else MempalaceConfig().palace_path
    )

    # Load dialect (with optional entity config)
    config_path = args.config
    if not config_path:
        for candidate in ["entities.json", str(Path(palace_path) / "entities.json")]:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        dialect = Dialect.from_config(config_path)
        print(f"  Loaded entity config: {config_path}")
    else:
        dialect = Dialect()

    # Connect to palace's default-model collection (read side).
    try:
        col = open_collection(palace_path)
    except (OSError, chromadb.errors.ChromaError, ValueError) as e:
        logger.debug("cmd_compress: palace open failed — %s", e)
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        sys.exit(1)

    # Query drawers in batches to avoid SQLite variable limit (~999)
    where = {"wing": args.wing} if args.wing else None
    batch_size = 500
    docs, metas, ids = [], [], []
    offset = 0
    while True:
        try:
            kwargs = {"include": ["documents", "metadatas"], "limit": batch_size, "offset": offset}
            if where:
                kwargs["where"] = where
            batch = col.get(**kwargs)
        except Exception as e:
            # Broad catch: col may be a mock in tests (RuntimeError) or the
            # real thing (ChromaError). Either way we stop paging.
            if not docs:
                print(f"\n  Error reading drawers: {e}")
                sys.exit(1)
            break
        batch_docs = batch.get("documents", [])
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(batch.get("metadatas", []))
        ids.extend(batch.get("ids", []))
        offset += len(batch_docs)
        if len(batch_docs) < batch_size:
            break

    if not docs:
        wing_label = f" in wing '{args.wing}'" if args.wing else ""
        print(f"\n  No drawers found{wing_label}.")
        return

    print(
        f"\n  Compressing {len(docs)} drawers"
        + (f" in wing '{args.wing}'" if args.wing else "")
        + "..."
    )
    print()

    total_original = 0
    total_compressed = 0
    compressed_entries = []

    for doc, meta, doc_id in zip(docs, metas, ids, strict=False):
        compressed = dialect.compress(doc, metadata=meta)
        stats = dialect.compression_stats(doc, compressed)

        total_original += stats["original_chars"]
        total_compressed += stats["summary_chars"]

        compressed_entries.append((doc_id, compressed, meta, stats))

        if args.dry_run:
            wing_name = meta.get("wing", "?")
            room_name = meta.get("room", "?")
            source = Path(meta.get("source_file", "?")).name
            print(f"  [{wing_name}/{room_name}] {source}")
            print(
                f"    {stats['original_tokens_est']}t -> {stats['summary_tokens_est']}t ({stats['size_ratio']:.1f}x)"
            )
            print(f"    {compressed}")
            print()

    # Store compressed versions (unless dry-run)
    if not args.dry_run:
        try:
            comp_col = open_collection(
                palace_path,
                create=True,
                collection_name_override="mempalace_compressed",
            )
            for doc_id, compressed, meta, stats in compressed_entries:
                comp_meta = dict(meta)
                comp_meta["compression_ratio"] = round(stats["size_ratio"], 1)
                comp_meta["original_tokens"] = stats["original_tokens_est"]
                comp_col.upsert(
                    ids=[doc_id],
                    documents=[compressed],
                    metadatas=[comp_meta],
                )
            print(
                f"  Stored {len(compressed_entries)} compressed drawers in 'mempalace_compressed' collection."
            )
        except Exception as e:
            # Broad catch: CLI boundary; the store path may fail for many
            # chroma-specific reasons (schema mismatch, disk full, etc.).
            print(f"  Error storing compressed drawers: {e}")
            sys.exit(1)

    # Summary
    ratio = total_original / max(total_compressed, 1)
    # Estimate tokens from char count (~3.8 chars/token for English text)
    orig_tokens = max(1, int(total_original / 3.8))
    comp_tokens = max(1, int(total_compressed / 3.8))
    print(f"  Total: {orig_tokens:,}t -> {comp_tokens:,}t ({ratio:.1f}x compression)")
    if args.dry_run:
        print("  (dry run -- nothing stored)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemPalace — Give your AI a memory. No API key required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Where the palace lives (default: from ~/.mempalace/config.json or ~/.mempalace/palace)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Detect rooms from your folder structure")
    p_init.add_argument("dir", help="Project directory to set up")
    p_init.add_argument(
        "--yes",
        action="store_true",
        help="Auto-accept all detected entities (non-interactive)",
    )

    # mine
    p_mine = sub.add_parser("mine", help="Mine files into the palace")
    p_mine.add_argument("dir", help="Directory to mine")
    p_mine.add_argument(
        "--mode",
        choices=["projects", "convos"],
        default="projects",
        help="Ingest mode: 'projects' for code/docs (default), 'convos' for chat exports",
    )
    p_mine.add_argument("--wing", default=None, help="Wing name (default: directory name)")
    p_mine.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore files when scanning project files",
    )
    p_mine.add_argument(
        "--include-ignored",
        action="append",
        default=[],
        help="Always scan these project-relative paths even if ignored; repeat or pass comma-separated paths",
    )
    p_mine.add_argument(
        "--agent",
        default="mempalace",
        help="Your name — recorded on every drawer (default: mempalace)",
    )
    p_mine.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    p_mine.add_argument(
        "--dry-run", action="store_true", help="Show what would be filed without filing"
    )
    p_mine.add_argument(
        "--extract",
        choices=["exchange", "general"],
        default="exchange",
        help="Extraction strategy for convos mode: 'exchange' (default) or 'general' (5 memory types)",
    )
    p_mine.add_argument(
        "--extract-kg",
        dest="extract_kg",
        action="store_true",
        help=(
            "Automatically extract knowledge-graph triples from each "
            "mined drawer and populate the KG. Uses the heuristic "
            "extractor by default; pass --kg-extract-mode ollama to "
            "use a local Ollama LLM instead."
        ),
    )
    p_mine.add_argument(
        "--kg-extract-mode",
        dest="kg_extract_mode",
        choices=["heuristic", "ollama"],
        default="heuristic",
        help=(
            "KG triple extraction strategy. 'heuristic' (default) uses "
            "pattern matching — zero new deps. 'ollama' uses a local "
            "Ollama LLM — requires the kg-extract-ollama extra: "
            "pip install 'mempalace[kg-extract-ollama]'."
        ),
    )
    p_mine.add_argument(
        "--kg-extract-model",
        dest="kg_extract_model",
        default="llama3.1:8b",
        help=(
            "Ollama model to use for --kg-extract-mode ollama "
            "(default: llama3.1:8b). Ignored in heuristic mode."
        ),
    )
    p_mine.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model slug to mine into (default: palace default). "
            "For code: jina-code-v2 (8k context, CodeSearchNet-trained). "
            "For long LLM conversations: nomic-text-v1.5 (8k context). "
            "For MTEB-proven general retrieval: mxbai-large. "
            "Each model writes to its own Chroma collection. "
            "Run `mempalace models list` for the full registry. "
            "See docs/MODEL_SELECTION.md for the decision guide."
        ),
    )

    # search
    p_search = sub.add_parser("search", help="Find anything, exact words")
    p_search.add_argument("query", help="What to search for (empty string = keyword/temporal only)")
    p_search.add_argument("--wing", default=None, help="Limit to one project")
    p_search.add_argument("--room", default=None, help="Limit to one room")
    p_search.add_argument("--results", type=int, default=5, help="Number of results")
    p_search.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Require an exact keyword via the trie index (repeat for AND)",
    )
    p_search.add_argument(
        "--keyword-prefix",
        action="append",
        default=[],
        help="Require a keyword prefix via the trie index (repeat for AND)",
    )
    p_search.add_argument(
        "--since",
        default=None,
        help="Only drawers filed on or after this ISO date",
    )
    p_search.add_argument(
        "--until",
        default=None,
        help="Only drawers filed on or before this ISO date",
    )
    p_search.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        help="Point-in-time: drawers whose validity window covers this date",
    )
    p_search.add_argument(
        "--warm-trie",
        dest="warm_trie",
        action="store_true",
        help="Preload the hot posting bitmaps into memory before querying",
    )
    p_search.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model slug (default: palace default). "
            "For code: jina-code-v2. For long LLM conversations: nomic-text-v1.5. "
            "For MTEB-proven general retrieval: mxbai-large. "
            "Pass 'all' to fan out across every enabled model with RRF fusion "
            "(k=60); fan-out auto-deduplicates overlapping drawers. "
            "See docs/MODEL_SELECTION.md for the decision guide."
        ),
    )
    p_search.add_argument(
        "--compress",
        choices=["auto", "none", "dedupe", "sentences", "aggressive", "llmlingua2"],
        default="auto",
        help=(
            "Result-set compression mode. 'auto' = dedupe on --model all, "
            "none otherwise. 'dedupe' clusters near-duplicate drawers, "
            "'sentences' also drops repeated sentences, 'aggressive' adds "
            "a novelty gate and honors --token-budget. 'llmlingua2' uses "
            "Microsoft's learned token-level compressor (requires the "
            "compress-llmlingua extra: pip install 'mempalace[compress-llmlingua]')."
        ),
    )
    p_search.add_argument(
        "--rerank",
        choices=["none", "provence", "bge"],
        default="none",
        help=(
            "Cross-encoder reranker to apply after semantic search. "
            "'none' (default) preserves the legacy ranking. "
            "'provence' loads naver/provence-reranker-debertav3-v1 and "
            "jointly reranks + per-token prunes each hit (requires the "
            "rerank-provence extra: pip install 'mempalace[rerank-provence]'). "
            "'bge' loads BAAI/bge-reranker-v2-m3 via fastembed ONNX for "
            "pure rerank without torch (requires the rerank-bge extra)."
        ),
    )
    p_search.add_argument(
        "--no-rerank-prune",
        dest="rerank_prune",
        action="store_false",
        default=True,
        help=(
            "Disable per-token pruning when using --rerank provence. "
            "Provence will still rerank, but the original drawer text "
            "flows through compression unchanged."
        ),
    )
    p_search.add_argument(
        "--kg-ppr",
        dest="enable_kg_ppr",
        action="store_true",
        help=(
            "Enable HippoRAG-style Personalized PageRank fusion over "
            "the knowledge graph. Extracts proper nouns from the "
            "query, runs PPR seeded on those entities, and unions "
            "the top-ranked drawers into the Chroma candidate set. "
            "Requires triples in the KG (populate via "
            "`mempalace mine --extract-kg` or `mempalace kg-extract`)."
        ),
    )
    p_search.add_argument(
        "--token-budget",
        dest="token_budget",
        type=int,
        default=None,
        help="Max output tokens; only honored in --compress aggressive mode.",
    )
    p_search.add_argument(
        "--dup-threshold",
        dest="dup_threshold",
        type=float,
        default=0.7,
        help=(
            "Drawer-level Jaccard cutoff for --compress dedupe (0..1, default 0.7). "
            "Lower values merge more aggressively."
        ),
    )
    p_search.add_argument(
        "--sent-threshold",
        dest="sent_threshold",
        type=float,
        default=0.75,
        help=(
            "Sentence-level bigram Jaccard cutoff for --compress sentences (0..1, default 0.75)."
        ),
    )
    p_search.add_argument(
        "--novelty-threshold",
        dest="novelty_threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum novel-trigram fraction for --compress aggressive "
            "(0..1, default 0.2). Below this, a hit is dropped."
        ),
    )

    # compress
    p_compress = sub.add_parser(
        "compress", help="Compress drawers using AAAK Dialect (~30x reduction)"
    )
    p_compress.add_argument("--wing", default=None, help="Wing to compress (default: all wings)")
    p_compress.add_argument(
        "--dry-run", action="store_true", help="Preview compression without storing"
    )
    p_compress.add_argument(
        "--config", default=None, help="Entity config JSON (e.g. entities.json)"
    )

    # wake-up
    p_wakeup = sub.add_parser("wake-up", help="Show L0 + L1 wake-up context (~600-900 tokens)")
    p_wakeup.add_argument("--wing", default=None, help="Wake-up for a specific project/wing")

    # split
    p_split = sub.add_parser(
        "split",
        help="Split concatenated transcript mega-files into per-session files (run before mine)",
    )
    p_split.add_argument("dir", help="Directory containing transcript files")
    p_split.add_argument(
        "--output-dir",
        default=None,
        help="Write split files here (default: same directory as source files)",
    )
    p_split.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be split without writing files",
    )
    p_split.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Only split files containing at least N sessions (default: 2)",
    )

    # hook
    p_hook = sub.add_parser(
        "hook",
        help="Run hook logic (reads JSON from stdin, outputs JSON to stdout)",
    )
    hook_sub = p_hook.add_subparsers(dest="hook_action")
    p_hook_run = hook_sub.add_parser("run", help="Execute a hook")
    p_hook_run.add_argument(
        "--hook",
        required=True,
        choices=["session-start", "stop", "precompact"],
        help="Hook name to run",
    )
    p_hook_run.add_argument(
        "--harness",
        required=True,
        choices=["claude-code", "codex"],
        help="Harness type (determines stdin JSON format)",
    )

    # instructions
    p_instructions = sub.add_parser(
        "instructions",
        help="Output skill instructions to stdout",
    )
    instructions_sub = p_instructions.add_subparsers(dest="instructions_name")
    for instr_name in ["init", "search", "mine", "help", "status"]:
        instructions_sub.add_parser(instr_name, help=f"Output {instr_name} instructions")

    # repair
    p_repair = sub.add_parser(
        "repair",
        help="Rebuild palace vector index from stored data (fixes segfaults after corruption)",
    )
    p_repair.add_argument(
        "--yes", action="store_true", help="Skip confirmation for destructive changes"
    )
    p_repair.add_argument(
        "--rebuild-aggregates",
        dest="rebuild_aggregates",
        action="store_true",
        default=True,
        help="Rebuild wing/hall/room aggregate embeddings after repair (default)",
    )
    p_repair.add_argument(
        "--no-rebuild-aggregates",
        dest="rebuild_aggregates",
        action="store_false",
        help="Skip aggregate rebuild after repair",
    )

    # trie-repair
    sub.add_parser(
        "trie-repair",
        help="Rebuild only the trie index from the existing Chroma collection",
    )

    # aggregates
    p_aggregates = sub.add_parser(
        "aggregates",
        help="Manage hierarchical wing/hall/room aggregate embeddings",
    )
    aggregates_sub = p_aggregates.add_subparsers(dest="aggregates_action")
    aggregates_sub.add_parser(
        "status",
        help="Show dirty/clean aggregate counts and last rebuild timestamp",
    )
    p_agg_rebuild = aggregates_sub.add_parser(
        "rebuild",
        help="Recompute wing/hall/room aggregate embeddings",
    )
    p_agg_rebuild.add_argument("--wing", default=None, help="Rebuild one wing aggregate")
    p_agg_rebuild.add_argument("--hall", default=None, help="Rebuild one hall aggregate")
    p_agg_rebuild.add_argument("--room", default=None, help="Rebuild one room aggregate")
    p_agg_rebuild.add_argument(
        "--all", action="store_true", help="Rebuild every container in the palace"
    )
    p_agg_rebuild.add_argument(
        "--model",
        default=None,
        help="Rebuild aggregates for a specific model slug (default: every enabled slug)",
    )
    p_agg_rebuild.add_argument(
        "--dry-run", action="store_true", help="Show what would be rebuilt without writing"
    )

    # models
    p_models = sub.add_parser(
        "models",
        help="Manage embedding models: list, enable/disable, download weights",
    )
    models_sub = p_models.add_subparsers(dest="models_action")
    models_sub.add_parser("list", help="List every embedding model with install/enable status")
    p_models_download = models_sub.add_parser(
        "download", help="Download an embedding model's weights eagerly"
    )
    p_models_download.add_argument("slug", help="Model slug from the registry")
    p_models_enable = models_sub.add_parser(
        "enable", help="Enable a model (add to enabled_embedding_models in config.json)"
    )
    p_models_enable.add_argument("slug", help="Model slug")
    p_models_disable = models_sub.add_parser(
        "disable", help="Disable a model (remove from enabled_embedding_models)"
    )
    p_models_disable.add_argument("slug", help="Model slug")
    p_models_default = models_sub.add_parser(
        "set-default", help="Set the default embedding model for mine/search"
    )
    p_models_default.add_argument("slug", help="Model slug")

    # rerankers
    p_rerankers = sub.add_parser(
        "rerankers",
        help="List cross-encoder rerankers (Provence, BGE) and their install status",
    )
    rerankers_sub = p_rerankers.add_subparsers(dest="rerankers_action")
    rerankers_sub.add_parser(
        "list",
        help="List every registered reranker with install + pruning status",
    )

    # kg-extract (retroactive)
    p_kg_extract = sub.add_parser(
        "kg-extract",
        help=(
            "Retroactively extract knowledge-graph triples from every "
            "drawer in an existing palace. Idempotent — safe to re-run."
        ),
    )
    p_kg_extract.add_argument(
        "--mode",
        choices=["heuristic", "ollama"],
        default="heuristic",
        help="Extraction strategy (default: heuristic, zero deps)",
    )
    p_kg_extract.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model for --mode ollama (default: llama3.1:8b)",
    )

    # mcp
    sub.add_parser(
        "mcp",
        help="Show MCP setup command for connecting MemPalace to your AI client",
    )

    # status
    sub.add_parser("status", help="Show what's been filed")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle two-level subcommands
    if args.command == "hook":
        if not getattr(args, "hook_action", None):
            p_hook.print_help()
            return
        cmd_hook(args)
        return

    if args.command == "instructions":
        name = getattr(args, "instructions_name", None)
        if not name:
            p_instructions.print_help()
            return
        args.name = name
        cmd_instructions(args)
        return

    if args.command == "models":
        cmd_models(args)
        return

    if args.command == "aggregates":
        cmd_aggregates(args)
        return

    if args.command == "rerankers":
        cmd_rerankers(args)
        return

    dispatch = {
        "init": cmd_init,
        "mine": cmd_mine,
        "split": cmd_split,
        "search": cmd_search,
        "mcp": cmd_mcp,
        "compress": cmd_compress,
        "wake-up": cmd_wakeup,
        "repair": cmd_repair,
        "trie-repair": cmd_trie_repair,
        "kg-extract": cmd_kg_extract,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
