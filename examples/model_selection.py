#!/usr/bin/env python3
"""Example: pick the right embedding model for your workload.

MemPalace ships seven embedding models. The default works for everything,
but the right specialized model can noticeably improve retrieval:

  - jina-code-v2     → source code (8k context, trained on CodeSearchNet)
  - nomic-text-v1.5  → long LLM conversations (8k context)
  - mxbai-large      → MTEB top-5 general retrieval (pulls torch)
  - bge-small-en     → budget general retrieval, same speed as default
  - ollama-nomic     → local Ollama server, zero extra download
  - ollama-mxbai     → local Ollama server, 1024-dim quality
  - default          → built-in ONNX, short English prose, no extras

This walkthrough shows the full workflow: install → enable → mine →
search single model → fan out with --model all.

See docs/MODEL_SELECTION.md for the full decision table and troubleshooting.
"""

import sys

project_dir = sys.argv[1] if len(sys.argv) > 1 else "~/projects/myrepo"
chats_dir = sys.argv[2] if len(sys.argv) > 2 else "~/chats"

print("# Step 1: Install the fastembed backend")
print("#   (unlocks jina-code-v2, nomic-text-v1.5, bge-small-en)")
print("  pip install 'mempalace[embeddings-fastembed]'")
print()
print("# Step 2: See every model with install / enable / drawer-count status")
print("  mempalace models list")
print()
print("# Step 3: Download the code-specialized model eagerly")
print("#   (otherwise the first mine call pays the download cost)")
print("  mempalace models download jina-code-v2")
print()
print("# Step 4: Enable it in the palace config")
print("  mempalace models enable jina-code-v2")
print()
print("# Step 5 (optional): Make it the default so --model can be omitted")
print("  mempalace models set-default jina-code-v2")
print()
print("# Step 6: Mine your code repo")
print("#   Writes to the mempalace_drawers__jina_code_v2 collection,")
print("#   leaving the default collection untouched.")
print(f"  mempalace mine {project_dir} --model jina-code-v2")
print()
print("# Step 7: Search — single model")
print('  mempalace search "where is auth verified" --model jina-code-v2')
print()
print("# Step 8: Search — RRF fan-out across every enabled model")
print("#   Auto-deduplicates drawers that surface in multiple collections.")
print('  mempalace search "where is auth verified" --model all')
print()
print("# Bonus: enable a second model for long LLM conversations")
print("  mempalace models enable nomic-text-v1.5")
print(f"  mempalace mine {chats_dir} --mode convos --model nomic-text-v1.5")
print('  mempalace search "what did we decide about auth" --model all')
