# Show HN draft — claudegrep v0.2

## Title (≤80 chars)
Show HN: claudegrep – grep, but semantic, for your Claude Code history

## Body

Every Claude Code session vanishes when you close the terminal. The JSONL files in `~/.claude/projects/` are right there, but `grep` is the only way to find anything in them. I wanted to type a vague half-remembered phrase and find the turn from three weeks ago where I figured out the right approach.

`claudegrep` indexes those JSONL files into one SQLite database (FTS5 + sqlite-vec, no separate vector store) and gives you a CLI:

```
$ recall index
$ recall search "sqlite vec hybrid" --rerank
$ recall show <session-id> --turn 7
$ recall inject <chunk-id>     # copies the turn text to your clipboard
$ recall watch                 # re-index on file change
```

Hybrid retrieval (BM25 + vector with RRF fusion) plus optional cross-encoder reranking. Filter inline: `project:foo since:7d role:user tool:Bash`.

I built a 51-query labeled eval set from my own developer history and reported the numbers in the README:

| Method | Recall@10 | MRR | nDCG@10 | p95 ms |
|---|---:|---:|---:|---:|
| BM25 (FTS5) | 0.216 | 0.125 | 0.148 | 2 |
| Vector (bge-small-zh, 512d) | 0.353 | 0.175 | 0.217 | 13 |
| Hybrid (RRF) | 0.392 | 0.175 | 0.228 | 16 |
| Hybrid + rerank (bge-reranker-base) | **0.471** | **0.230** | **0.289** | 214 |

+118% Recall@10 over BM25. Absolute numbers look modest because the eval queries are deliberately short (median 5 chars) and developer-specific — that's the realistic distribution for "I vaguely remember talking about this last month." Query expansion + bge-m3 are next.

Local-first by design: no network calls in the OSS build. Runs on a CPU laptop. Apache-2.0.

Install:

```
uv tool install claudegrep    # or pipx install claudegrep
recall index
```

Repo: https://github.com/lbbstarry/claudegrep
PyPI: https://pypi.org/project/claudegrep/
Eval methodology + ablation: see README and `benchmarks/eval_results.md`.

Curious what other Claude Code users actually search for — if you try it and the queries you wished worked didn't, I'd love to hear about them.

---

## Posting checklist
- [ ] Tuesday 8:00 AM PT
- [ ] First comment from author: thank-you + link to eval methodology
- [ ] Pre-prepped 30s GIF embedded in README (not in HN body)
- [ ] Reply within first 30 min to top comments
- [ ] Cross-post r/ClaudeAI 24h later, r/LocalLLaMA 48h later, r/commandline 72h later
