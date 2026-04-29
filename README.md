# claude-recall

> Semantic search across your Claude Code conversation history. Local-first, private, fast.

Every Claude Code session vanishes the moment you close it. `recall` indexes
your `~/.claude/projects/` JSONL files into a single SQLite database and
gives you a fast CLI to find any past turn — by keyword now, by meaning soon.

```text
$ recall search "sqlite vec hybrid"
1. 2026-04-29  claude-recall  score=-18.4
   how should we store the chunk vectors — sqlite-vec or lancedb?
   [assistant] for <10M vectors sqlite-vec wins on ops simplicity…

2. 2026-04-22  gstack         score=-14.1
   …
```

## Status

**Week 1 / 6** — BM25 search shipping. Hybrid (BM25 + bge-m3 vector) lands in
Week 2 with a published Recall@10 / nDCG eval. See [PLAN.md](PLAN.md).

## Install (dev)

```bash
git clone https://github.com/<you>/claude-recall
cd claude-recall
uv sync
uv run recall index
uv run recall search "your query"
```

PyPI release will follow v0.2 (end of Week 4).

## Commands

| Command | What it does |
|---|---|
| `recall index` | Scan `~/.claude/projects/` and index new/changed sessions (incremental via mtime + sha256). |
| `recall search <q>` | BM25 search. `--limit N`, `--project NAME`. |
| `recall stats` | Index size and DB path. |

Coming in Week 2+:
- `--rerank` — bge-reranker-v2-m3 cross-encoder
- `recall show <session>` — render a full session
- `recall inject <chunk>` — copy a turn to the clipboard for paste-into-Claude
- `recall serve` — local FastAPI + HTMX UI

## Architecture

```
~/.claude/projects/*.jsonl
    │
    ▼
parsers/claude_code.py    typed Message records
    │
    ▼
ingest/chunker.py         per-turn chunks (one user msg + following assistant)
    │
    ▼
store/                    SQLite + FTS5 (vec0 added in Week 2)
    │
    ▼
search/bm25.py            FTS5 MATCH + bm25() ranking
    │
    ▼
cli.py                    Typer app
```

Why per-turn chunks? Per-message loses Q-A pairing; sliding-window inflates
the index 3-5×. A turn averages ~1.2k tokens — perfect for `bge-m3`'s 8k
context, no truncation needed.

Why SQLite + FTS5 + (later) sqlite-vec? Single file, zero ops, ships with the
wheel, hybrid search is a JOIN away. Beats Chroma at this scale.

## Eval (placeholder — landing Week 2)

| Method | Recall@10 | nDCG@10 | p95 latency |
|---|---|---|---|
| BM25 only | TODO | TODO | TODO |
| Vector (bge-m3) | TODO | TODO | TODO |
| Hybrid (RRF) | TODO | TODO | TODO |
| Hybrid + rerank | TODO | TODO | TODO |

Eval queries: 50 hand-labeled `(query, relevant_chunk_id)` pairs from real
developer sessions. Set lives in `tests/fixtures/queries.jsonl` once it lands.

## Roadmap

- [x] **Week 1** — Typer CLI, SQLite + FTS5, incremental ingest, BM25 search
- [ ] **Week 2** — bge-m3 embeddings, sqlite-vec, RRF hybrid, 50-query eval
- [ ] **Week 3** — `recall show`/`inject`/`export`, `--watch` daemon, filters
- [ ] **Week 4** — v0.2 on PyPI, Show HN, eval blog post
- [ ] **Week 5** — `recall serve` (FastAPI + HTMX local UI)
- [ ] **Week 6** — v1.0, Pro tier (cloud sync, Voyage embedder), Product Hunt

## Privacy

Everything stays on your machine. The index is a single SQLite file under your
OS's user data dir (`~/.local/share/claude-recall/recall.db` on Linux). No
network calls are made by the OSS build.

## License

Apache-2.0
