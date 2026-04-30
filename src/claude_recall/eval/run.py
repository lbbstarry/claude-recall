"""Eval harness: Recall@k, MRR, nDCG@k for BM25 / vector / hybrid."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from claude_recall.config import load_config
from claude_recall.embed.local import LocalEmbedder
from claude_recall.search import bm25, hybrid, vector
from claude_recall.search.bm25 import SearchHit
from claude_recall.search.rerank import Reranker
from claude_recall.store import db

DEFAULT_QUERIES = Path("tests/fixtures/queries.jsonl")
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
K = 10

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _load_queries(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append((rec["query"], rec["relevant_chunk_id"]))
    return out


def _metrics(hits: list[SearchHit], gold_id: str, k: int) -> tuple[float, float, float]:
    """Returns (recall@k, mrr, ndcg@k) for a single query (single relevant doc)."""
    ids = [h.chunk_id for h in hits[:k]]
    recall = 1.0 if gold_id in ids else 0.0
    if gold_id in ids:
        rank = ids.index(gold_id) + 1
        mrr = 1.0 / rank
        ndcg = 1.0 / math.log2(rank + 1)
    else:
        mrr = 0.0
        ndcg = 0.0
    return recall, mrr, ndcg


def _evaluate(
    name: str,
    queries: list[tuple[str, str]],
    search_fn: Callable[[str], list[SearchHit]],
    k: int = K,
) -> dict[str, float]:
    rs, ms, ns, lats = [], [], [], []
    for q, gold in queries:
        t0 = time.perf_counter()
        hits = search_fn(q)
        lats.append((time.perf_counter() - t0) * 1000)
        r, m, n = _metrics(hits, gold, k)
        rs.append(r)
        ms.append(m)
        ns.append(n)
    n_q = len(queries)
    p95 = sorted(lats)[int(0.95 * (n_q - 1))] if n_q else 0.0
    return {
        "method": name,
        f"recall@{k}": sum(rs) / n_q,
        "mrr": sum(ms) / n_q,
        f"ndcg@{k}": sum(ns) / n_q,
        "p95_ms": p95,
        "n": float(n_q),
    }


@app.command()
def main(
    queries_path: Path = typer.Option(DEFAULT_QUERIES, "--queries"),
    output: Path = typer.Option(Path("benchmarks/eval_results.md"), "--out"),
) -> None:
    cfg = load_config()
    queries = _load_queries(queries_path)
    console.print(f"[bold]eval set:[/bold] {len(queries)} queries from {queries_path}")

    embedder = LocalEmbedder(DEFAULT_MODEL)
    conn = db.connect(cfg.db_path, vector_dim=embedder.dim)

    reranker = Reranker()
    methods = {
        "BM25": lambda q: bm25.search(conn, q, limit=K),
        f"Vector ({embedder.name})": lambda q: vector.search(conn, embedder, q, limit=K),
        "Hybrid (RRF)": lambda q: hybrid.search(conn, embedder, q, limit=K),
        "Hybrid + rerank": lambda q: reranker.rerank(
            q, hybrid.search(conn, embedder, q, limit=30), top_k=K
        ),
    }
    rows = [_evaluate(name, queries, fn) for name, fn in methods.items()]

    table = Table(title=f"eval (n={len(queries)}, k={K})")
    table.add_column("method")
    table.add_column(f"recall@{K}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"nDCG@{K}", justify="right")
    table.add_column("p95 ms", justify="right")
    for r in rows:
        table.add_row(
            r["method"],
            f"{r[f'recall@{K}']:.3f}",
            f"{r['mrr']:.3f}",
            f"{r[f'ndcg@{K}']:.3f}",
            f"{r['p95_ms']:.1f}",
        )
    console.print(table)

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# claude-recall eval (n={len(queries)}, k={K})",
        "",
        f"Embedder: `{embedder.name}` (dim={embedder.dim})",
        "",
        f"| method | recall@{K} | MRR | nDCG@{K} | p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r[f'recall@{K}']:.3f} | {r['mrr']:.3f} | "
            f"{r[f'ndcg@{K}']:.3f} | {r['p95_ms']:.1f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"\n[green]wrote[/green] {output}")


if __name__ == "__main__":
    app()
