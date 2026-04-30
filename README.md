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

**Week 3 / 6** — Daily-driver UX is live: `show`, `inject`, `export`, the
`watch` daemon, a `--rerank` flag, and an inline filter DSL
(`project:` `since:` `role:` `tool:`). Hybrid + reranker eval numbers below.
See [PLAN.md](PLAN.md).

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
| `recall index` | Scan `~/.claude/projects/` and index + embed new/changed sessions (incremental via mtime + sha256). `--no-embed` for BM25-only. |
| `recall search <q>` | Hybrid BM25 + vector search by default. `--mode bm25\|vector\|hybrid`, `--rerank`, `--limit N`, `--project NAME`. Inline filters: `project:foo since:7d role:user tool:Bash`. |
| `recall show <session>` | Render a session as Markdown. Accepts an 8+ char prefix; use `--turn N` to view one turn. |
| `recall inject <chunk>` | Copy a chunk's text to your clipboard so you can paste it into a new Claude session. |
| `recall export <session> -o file.md` | Export a session to disk as Markdown. |
| `recall watch` | Re-index in the background as JSONL files change (debounced; uses polling on WSL2). |
| `recall stats` | Index size, vector count, DB path. |
| `python -m claude_recall.eval.run` | Run the labeled eval set, print Recall@10 / MRR / nDCG@10, write `benchmarks/eval_results.md`. |

Coming in Week 4+:
- v0.2 on PyPI + Show HN
- `recall serve` — local FastAPI + HTMX UI
- bge-m3 + query expansion + ablation blog post

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
embed/local.py            sentence-transformers (bge-small-zh) + disk cache
    │
    ▼
store/                    SQLite + FTS5 + sqlite-vec (vec0 virtual table)
    │
    ▼
search/                   bm25 · vector (KNN) · hybrid (RRF) · rerank (cross-encoder)
    │
    ▼
cli.py                    Typer app
```

Why per-turn chunks? Per-message loses Q-A pairing; sliding-window inflates
the index 3-5×. A turn averages ~1.2k tokens — perfect for `bge-m3`'s 8k
context, no truncation needed.

Why SQLite + FTS5 + (later) sqlite-vec? Single file, zero ops, ships with the
wheel, hybrid search is a JOIN away. Beats Chroma at this scale.

## Eval

51 hand-labeled `(query, relevant_chunk_id)` pairs from real developer
sessions in `tests/fixtures/queries.jsonl`. Index size: 6,593 chunks across
127 sessions / 6 projects. Reproduce with `python -m claude_recall.eval.run`.

| Method | Recall@10 | MRR | nDCG@10 | p95 ms |
|---|---:|---:|---:|---:|
| BM25 (FTS5) | 0.216 | 0.125 | 0.148 | 2 |
| Vector (`bge-small-zh-v1.5`, 512d) | 0.353 | 0.175 | 0.217 | 13 |
| **Hybrid (RRF)** | **0.392** | 0.175 | 0.228 | 16 |
| **Hybrid + rerank (`bge-reranker-base`)** | **0.471** | **0.230** | **0.289** | 214 |

Hybrid + rerank gives **+118% Recall@10** and **+95% nDCG@10** over BM25.
Reranker latency is dominated by CPU cross-encoder inference; GPU or
`bge-reranker-v2-m3-onnx` will reduce it. Numbers are CPU-only on a WSL2
Ryzen laptop.

Why the absolute numbers look modest: the eval queries are deliberately
**short** (median 5 chars) and developer-domain-specific, e.g. `prefab`,
`figma示例`. That is the realistic distribution for "I vaguely remember
talking about this last month" — and the gap between methods, not the
absolute floor, is what matters. Query expansion and bge-m3 (8k context,
multilingual) are next.

## Roadmap

- [x] **Week 1** — Typer CLI, SQLite + FTS5, incremental ingest, BM25 search
- [x] **Week 2** — bge-small-zh embeddings, sqlite-vec, RRF hybrid, reranker, 51-query eval
- [x] **Week 3** — `show`/`inject`/`export`, `watch` daemon, filter DSL, `--rerank` flag
- [ ] **Week 4** — v0.2 on PyPI, Show HN, eval blog post
- [ ] **Week 5** — `recall serve` (FastAPI + HTMX local UI)
- [ ] **Week 6** — v1.0, Pro tier (cloud sync, Voyage embedder), Product Hunt

## Privacy

Everything stays on your machine. The index is a single SQLite file under your
OS's user data dir (`~/.local/share/claude-recall/recall.db` on Linux). No
network calls are made by the OSS build.

## License

Apache-2.0
