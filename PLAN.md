# claude-recall — Comprehensive Plan

A tool that indexes and semantically searches Claude Code conversation history. Existing prototype at `/home/libingbing/claude-recall/parse.py` already iterates 15.7k messages across 5 projects. Below is the path from prototype to shippable v1.0 + monetization.

---

## Half A: Engineering Plan

### A.1 Architecture (one-line summary)

A local-first Python CLI that watches `~/.claude/projects/`, chunks messages turn-by-turn, embeds them with a **local bge-m3** model, stores vectors + metadata + FTS in **a single SQLite file via sqlite-vec + FTS5**, and exposes search through a Typer CLI. A FastAPI + HTMX web UI ships in Phase 2 as the Pro entry point.

### A.2 Tech Choices (opinionated)

| Layer | Pick | Why (one alternative rejected) |
|---|---|---|
| Language | Python 3.11+ | User is fluent; ecosystem matches indexing/embeddings work. |
| Packaging | **uv + pyproject.toml + src layout** | uv is 10x faster than poetry, lockfiles are stable, default in AI infra repos. Reject poetry: slower, heavier. |
| CLI | **Typer** | Click under the hood + typed signatures + auto-help. Reject click: more boilerplate. |
| Embeddings (default) | **bge-m3 via FlagEmbedding or sentence-transformers** (1024-dim, multilingual, 8k context) | Fully local, free forever, multilingual (matters: user is in China and code mixes EN/ZH). Quality competitive with voyage-3-lite. Reject voyage-3-lite: $0.06/1M tok and adds privacy concerns for OSS users. Reject OpenAI text-embedding-3-small: cloud dependency hurts the local-first pitch. Pro tier opts into voyage-3-lite for ~5pt nDCG bump. |
| Vector store | **sqlite-vec + FTS5 in one DB** | Single file, zero ops, ships embedded with the wheel, hybrid search trivial via JOIN. Reject lancedb (extra dep, no real win at <10M rows), chroma (heavyweight server reflex, mediocre under 100k vectors), duckdb-vss (great but FTS story is weaker). |
| Reranker | **bge-reranker-v2-m3** (cross-encoder, lazy-loaded) | Used only on top-50 candidates. Reject Cohere rerank-3: cloud-only. |
| Web UI (Phase 2) | **FastAPI + HTMX + Tailwind** | Same Python codebase, no node toolchain, deploys as a single binary via uvicorn. Reject SvelteKit/Next: forces JS build pipeline that adds friction for a solo grad-student maintainer. |
| Watcher | **watchdog** | Cross-platform inotify abstraction. WSL2-safe with polling fallback. |
| Telemetry | **PostHog (self-hostable) or simple HTTPS POST**, opt-in | Off by default; one prompt on first `recall index`. |

### A.3 File Layout

```
~/claude-recall/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE                      # Apache-2.0
├── .github/workflows/ci.yml
├── src/claude_recall/
│   ├── __init__.py
│   ├── __main__.py              # python -m claude_recall
│   ├── cli.py                   # Typer app, command surface
│   ├── config.py                # XDG-aware paths, env, defaults
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── claude_code.py       # JSONL schema for ~/.claude/projects
│   │   ├── cursor.py            # Phase 3
│   │   └── base.py              # Parser protocol
│   ├── ingest/
│   │   ├── scanner.py           # rglob + mtime/hash diff
│   │   ├── chunker.py           # turn-based chunking
│   │   └── pipeline.py          # parse → chunk → embed → upsert
│   ├── embed/
│   │   ├── base.py              # Embedder protocol
│   │   ├── bge_m3.py            # local default
│   │   ├── voyage.py            # Pro option
│   │   └── cache.py             # disk LRU on chunk hash
│   ├── store/
│   │   ├── schema.sql
│   │   ├── db.py                # sqlite-vec + FTS5 connection
│   │   └── repo.py              # SessionRepo, MessageRepo, ChunkRepo
│   ├── search/
│   │   ├── hybrid.py            # RRF fusion of BM25 + vector
│   │   ├── rerank.py            # bge-reranker, lazy
│   │   └── filters.py           # project, role, date, tool_use
│   ├── ui/                      # Phase 2
│   │   ├── server.py            # FastAPI
│   │   ├── templates/
│   │   └── static/
│   ├── watch.py                 # watchdog daemon
│   └── telemetry.py             # opt-in
├── tests/
│   ├── conftest.py              # fixtures: anonymized JSONL samples
│   ├── fixtures/sessions/       # 5-10 redacted real sessions
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_repo.py
│   │   └── test_hybrid.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── eval/
│       ├── queries.jsonl        # 50 hand-labeled (query, relevant_chunk_id) pairs
│       └── run_eval.py          # Recall@10, MRR, nDCG@10
└── benchmarks/
    └── index_throughput.py
```

### A.4 Data Model

```python
# All frozen dataclasses (immutability)

@dataclass(frozen=True)
class Session:
    id: str                      # JSONL stem
    project: str                 # parent dir
    file_path: Path
    file_mtime: float
    file_sha256: str             # for incremental
    started_at: datetime
    ended_at: datetime
    message_count: int

@dataclass(frozen=True)
class Message:
    uuid: str                    # from JSONL
    session_id: str
    parent_uuid: str | None
    role: str                    # user|assistant
    timestamp: datetime
    text: str                    # extracted via existing extract_text()
    has_tool_use: bool
    tool_names: tuple[str, ...]

@dataclass(frozen=True)
class Chunk:
    id: str                      # sha256(session_id + turn_index)
    session_id: str
    turn_index: int              # 0-based turn (one user + following assistant)
    start_uuid: str
    end_uuid: str
    text: str                    # concatenated turn text, truncated to ~2k tokens
    token_count: int
    embedding: bytes             # f32 little-endian
```

**Chunking decision: per-turn.** A "turn" = one user message + the immediately following assistant response (including tool_use/tool_result blocks until next user message). Per-message loses Q-A pairing; sliding-window inflates index 3-5x. Per-turn averages ~1.2k tokens, fits bge-m3 comfortably.

**Incremental indexing:** SQLite table `files(path PK, mtime, sha256, last_indexed)`. On `recall index`: walk JSONL, skip if `(mtime, sha256)` unchanged; else delete chunks for that session and reingest. Cheap, idempotent, crash-safe.

**Filterable metadata** in normal SQL columns on `chunks`: `project`, `role_mix`, `started_at`, `has_tool_use`, `tool_names`. Filters become WHERE clauses before vector search.

### A.5 Search Design

Hybrid retrieval, **RRF fusion**, optional reranking:

1. Parse query → optional filters (`project:foo`, `since:7d`, `tool:bash`).
2. **BM25** via FTS5 on `chunks.text` → top 50.
3. **Vector** via `vec0` virtual table → top 50.
4. **Reciprocal Rank Fusion** (k=60) → merged top 30.
5. If `--rerank` (or auto when ≥10 candidates), bge-reranker-v2-m3 → top 10.
6. Return at two granularities: `recall search` defaults to chunk-level; `--sessions` rolls up by `session_id`.

Why hybrid: pure vector misses exact tokens (filenames, error strings, function names) that dominate dev queries; pure BM25 misses paraphrase. RRF beats weighted scoring — no per-query tuning.

### A.6 CLI UX

```
recall index                       # full or incremental scan
recall index --watch               # daemonize via watchdog
recall search "rrf fusion"         # default: chunks, top 10
recall search "auth bug" --project myapp --since 30d --sessions
recall show <session_id>           # render full session, paged
recall show <session_id> --turn 14
recall inject <chunk_id>           # copy turn text to clipboard
recall stats                       # counts, index size, last sync
recall config                      # show/edit ~/.config/claude-recall/config.toml
recall serve --port 7777           # Phase 2 web UI
recall export <session_id> --md    # markdown export
```

Example output:

```
$ recall search "sqlite vec hybrid" --since 14d
1. [2026-04-22 10:14] claude-recall  score=0.91
   Q: should we use sqlite-vec or lancedb for chunk storage…
   A: For <10M vectors sqlite-vec wins on ops simplicity. JOIN with FTS5…
   → recall show 9f2c…  |  recall inject c_4831
```

### A.7 Phased Delivery (6-week v1.0)

#### Week 1 — Skeleton + Ingest
- Migrate `parse.py` into `parsers/claude_code.py` with type-annotated `Message` / `Session`.
- uv project, src layout, ruff + black + mypy + pytest wired in CI.
- SQLite schema, `ChunkRepo` upsert, incremental file tracker.
- CLI: `recall index`, `recall stats`, `recall search` (BM25 only).
- **DoD:** `recall index` on real ~/.claude/projects completes <60s, idempotent rerun <5s, BM25 returns relevant hits, 70% line coverage.

#### Week 2 — Embeddings + Hybrid
- bge-m3 embedder, batched, on-disk cache keyed by chunk hash.
- sqlite-vec virtual table, RRF fusion.
- 50-query labeled eval set from user's own history; `python -m claude_recall.eval` reports Recall@10 and nDCG@10.
- **DoD:** Hybrid Recall@10 ≥ 0.75; full reindex of 15.7k messages <10 min on CPU; `recall search` p95 latency <300ms.

#### Week 3 — UX polish + Watch + Filters
- `recall show`, `recall inject` (pyperclip), `recall export --md`.
- Filter parsing (`project:`, `since:`, `tool:`, `role:`).
- `--watch` daemon via watchdog with WSL2 polling fallback.
- Rich-formatted output, --json mode.
- **DoD:** Every CLI command works; watch picks up new JSONL within 2s; coverage ≥80%.

#### Week 4 — v0.2 OSS Launch
- README with GIF, install via `uv tool install claude-recall` and `pipx install claude-recall`.
- Anonymized eval blog post + benchmark numbers.
- Show HN + r/ClaudeAI + Anthropic Discord post.
- Telemetry (opt-in: anonymous_id, command_name, duration_ms).
- **DoD:** Published to PyPI; README has install + 30s demo GIF; 50 GitHub stars or first 10 issue/PR signals.

#### Week 5 — Web UI (FastAPI + HTMX)
- `recall serve` boots local UI at 127.0.0.1:7777.
- Search box, filter sidebar, session viewer, syntax-highlighted code blocks.
- Auth-less by default; bind to localhost.
- **DoD:** UI matches CLI parity; Lighthouse a11y ≥90; loads <300ms cached.

#### Week 6 — Pro Foundations + v1.0
- License key gate (Lemon Squeezy / Stripe Checkout, offline JWT verification).
- Pro feature flag for: voyage-3-lite embedder, encrypted cloud sync stub (S3/R2 user-supplied bucket), multi-machine merge.
- Product Hunt launch with v1.0.
- **DoD:** Stripe webhook → license email tested end-to-end; Pro features gated; Product Hunt page live; pricing page live.

### A.8 Testing Strategy

- **pytest** with `unit`, `integration`, `eval` marks.
- Fixtures: 5–10 hand-redacted sessions in `tests/fixtures/sessions/`.
- Property-based tests on parser via Hypothesis.
- Coverage gate: 80% lines, 70% branches; enforced in CI.
- Eval harness produces markdown report committed to `benchmarks/` per release.

### A.9 Cross-Platform

- Primary: Linux (incl. WSL2), macOS. Both in CI matrix.
- Windows native: best-effort.
- WSL2: detect via `/proc/version` containing "microsoft"; force watchdog `PollingObserver`.

### A.10 Cross-Tool Roadmap

**v1.0 stays Claude Code only.** Cursor (SQLite) and Cline/Aider (varied) have very different storage. Add Cursor in v1.1 (week 8), Aider in v1.2. Frame externally as "deepest support for Claude Code first, broader tools next" — moat, not limitation.

---

## Half B: Monetization & GTM

### B.1 Free vs Pro Split

| Capability | Free (OSS, Apache-2.0) | Pro |
|---|---|---|
| Local indexing + hybrid search (CLI) | yes | yes |
| Watch daemon | yes | yes |
| Local web UI (`recall serve`) | yes | yes |
| Markdown export | yes | yes |
| **Encrypted cloud sync across machines** | — | yes |
| **Voyage / OpenAI / Anthropic embedder backends** | — | yes |
| **Cross-tool import (Cursor/Cline/Aider)** | — | yes (in v1.1+) |
| **Team workspace + shared search** | — | Team tier |
| **Hosted web UI (no install)** | — | Team tier |
| Priority support | — | Pro/Team |

Principle: local-only solo-dev experience is forever free and complete — drives stars and adoption. Pro starts the moment you cross a machine boundary, want managed embedding upgrade, or share with teammates.

### B.2 Pricing

- **Free** — local CLI + UI, forever.
- **Pro: $8 / month** or **$72 / year** (one license, up to 3 personal machines). Comparable: Raycast Pro $10, Warp $15, Cody Pro $9.
- **Team: $15 / user / month**, min 3 seats. Adds shared workspace and SSO-lite.
- **Lifetime early-bird (first 100 customers): $99 one-time.** Funds infra and seeds testimonials.

### B.3 Distribution (priority order)

1. **Build-in-public on X** (week 0 → ongoing). Daily: screenshot/GIF/benchmark. Tag @AnthropicAI, @simonw on milestones. Goal: 500 followers by launch.
2. **Show HN** at v0.2 (end of Week 4). Title: *"Show HN: claude-recall — semantic search over your Claude Code history (local-first, hybrid BM25+vector)"*. Lead with 20s GIF + eval table. Post 8am PT Tuesday.
3. **r/ClaudeAI, r/LocalLLaMA, r/commandline** — same week, 24h apart, each with different angle.
4. **Anthropic Discord + Claude Developers Discord** — short demo to maintainers.
5. **GitHub awesome lists** — PRs to `awesome-claude`, `awesome-cli`, `awesome-ai-agents`.
6. **Product Hunt** at v1.0 (end of Week 6). Tue/Wed 12:01am PT. Pre-line up 20 hunters/comments.
7. **Hacker News follow-up** — only after meaningful version bump (v1.1 with Cursor support).

### B.4 Launch Sequencing (relative to today, 2026-04-29)

- 2026-04-29 → 2026-05-04: Week 1 build, first build-in-public tweet by 2026-05-01.
- 2026-05-19: v0.2 release + Show HN.
- 2026-06-02: v1.0 release + Product Hunt + Pro launch.
- 2026-06-16: Cursor support v1.1, second X push.

### B.5 Metrics from Day 1

- GitHub stars, issues, contributors.
- PyPI weekly downloads (trend signal).
- Anonymous opt-in telemetry: weekly active CLI users, command mix, p95 latency, index size buckets.
- Conversion: pricing page → checkout → activated license.
- Eval Recall@10 over time (must not regress).
- North Star: **Weekly Active Indexers (WAI)** = unique anonymous_ids running `recall search` ≥3 times in 7 days. Target end of month 3: 500 WAI, 25 paying.

### B.6 Top 3 Risks

1. **Anthropic ships native history search.** *Mitigation:* (a) move fast — be obvious answer for 6 months; (b) ship cross-tool support in v1.1 — value prop becomes "one search across all your AI coding history," which Anthropic won't build; (c) own workflow layer (inject, export, team share).
2. **Privacy backlash on cloud Pro features.** *Mitigation:* end-to-end encryption with user-held key (libsodium secretstream), bring-your-own-bucket option, public threat model, OSS the sync client.
3. **Eval/quality regression as index grows.** *Mitigation:* labeled eval set committed to repo, runs in CI on every PR; >2 nDCG point regression blocks merge. Doubles as resume credibility artifact.

### B.7 Career / Resume Angle

Frame as: *"Production-quality retrieval system for AI coding session history: hybrid BM25+vector search over SQLite-vec, bge-m3 embeddings, evaluated on hand-labeled set with Recall@10 / nDCG / MRR reporting."* Hits AI infra, retrieval, agent tooling, rigorous eval.

Concrete artifacts:
- **Eval blog post** (Week 4): "How well does hybrid retrieval actually work on dev conversation data?" with labeled set, ablation table, latency vs quality Pareto curve.
- **Architecture deep-dive** (Week 5): "Why sqlite-vec + FTS5 in one file beats Chroma for 100k-vector dev tools."
- **Benchmark repo** with reproducible scripts.
- **README header**: install GIF, eval table, architecture diagram. README is the resume.

### B.8 Three Specific Actions in Week 1

1. **2026-05-01**: Register `claude-recall` on PyPI (placeholder 0.0.0), GitHub repo public Apache-2.0, X handle `@claudeRecall`. Post tweet #1: 15s GIF of existing `parse.py` substring search, caption *"shipping a real semantic version of this in 6 weeks — building in public"*.
2. **2026-05-02**: Hand-label 50 (query, relevant_chunk_id) pairs from your own history into `tests/fixtures/queries.jsonl`. Highest-leverage 2-hour task — converts project from "I think this is good" to "I can prove this is good".
3. **2026-05-04**: Write README first (README-driven). Include: 1-paragraph pitch, install command, 3 example commands with real output, planned eval table with TODO numbers, architecture diagram. Push before any vector code.

---

## The 20% That Captures 80% of the Value

Ship Week 1 + Week 2 only — Typer CLI with `recall index` and `recall search`, hybrid BM25 + bge-m3 in single SQLite file, on-disk embedding cache, incremental reindex via mtime/sha, labeled 50-query eval set in README with Recall@10 numbers. No watch daemon, no web UI, no Pro tier, no cross-tool, no telemetry.

Why this is the 80%:
- Solves real pain end-to-end on user's own machine.
- Labeled eval set + published numbers = credibility for Show HN and AI-infra resume.
- Everything else layered on top; if core retrieval is bad, nothing matters; if good, rest sells itself.

If forced to compress further: ship `recall search` with hybrid retrieval and eval table in README. That's the entire pitch.
