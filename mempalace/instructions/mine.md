# MemPalace Mine

When the user invokes this skill, follow these steps:

## 1. Ask what to mine

Ask the user what they want to mine and where the source data is located.
Clarify:
- Is it a project directory (code, docs, notes)?
- Is it conversation exports (Claude, ChatGPT, Slack)?
- Do they want auto-classification (decisions, milestones, problems)?

## 2. Choose the mining mode

There are three mining modes:

### Project mining

    mempalace mine <dir>

Mines code files, documentation, and notes from a project directory.

### Conversation mining

    mempalace mine <dir> --mode convos

Mines conversation exports from Claude, ChatGPT, or Slack into the palace.

### General extraction (auto-classify)

    mempalace mine <dir> --mode convos --extract general

Auto-classifies mined content into decisions, milestones, and problems.

## 2b. Consider a specialized embedding model (optional)

By default MemPalace uses the built-in ONNX all-MiniLM-L6-v2 model, which
works well for general English prose. If the user is mining one of these
specialized workloads, suggest (but don't force) a dedicated model:

- **Source code repositories** → `--model jina-code-v2`
  (trained on CodeSearchNet, 8192-token context)
- **Long LLM conversations / decision logs / architecture docs** →
  `--model nomic-text-v1.5` (8192-token context means no
  mid-drawer truncation)
- **MTEB-proven general retrieval, torch already installed** →
  `--model mxbai-large` (1024-dim, MTEB top-5)

Before mining with a specialized model, confirm the extras are
installed and the slug is enabled:

    mempalace models list                      # shows install/enable status
    mempalace models enable <slug>             # if not yet enabled

Each model writes to its own Chroma collection, so switching `--model`
is non-destructive — the existing `default` collection stays put. See
docs/MODEL_SELECTION.md for the full decision table.

## 3. Optionally split mega-files first

If the source directory contains very large files, suggest splitting them
before mining:

    mempalace split <dir> [--dry-run]

Use --dry-run first to preview what will be split without making changes.

## 4. Optionally tag with a wing

If the user wants to organize mined content under a specific wing, add the
--wing flag:

    mempalace mine <dir> --wing <name>

## 5. Show progress and results

Run the selected mining command and display progress as it executes. After
completion, summarize the results including:
- Number of items mined
- Categories or classifications applied
- Any warnings or skipped files

## 6. Suggest next steps

After mining completes, suggest the user try:
- /mempalace:search -- search the newly mined content
- /mempalace:status -- check the current state of their palace
- Mine more data from additional sources
