"""recall CLI."""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.table import Table

from claude_recall.config import load_config
from claude_recall.embed.cache import EmbedCache
from claude_recall.embed.local import LocalEmbedder
from claude_recall.ingest.pipeline import ingest_all
from claude_recall.search import bm25, hybrid, vector
from claude_recall.store import db, repo

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _open(*, with_vec: bool = True) -> tuple:
    cfg = load_config()
    if with_vec:
        embedder = LocalEmbedder(DEFAULT_MODEL)
        conn = db.connect(cfg.db_path, vector_dim=embedder.dim)
        return cfg, conn, embedder
    return cfg, db.connect(cfg.db_path), None


@app.command()
def index(no_embed: bool = typer.Option(False, "--no-embed", help="BM25 only")) -> None:
    """Scan ~/.claude/projects/ and index new/changed sessions."""
    cfg, conn, embedder = _open(with_vec=not no_embed)
    cache = EmbedCache(cfg.data_dir / "embed_cache.db") if embedder else None
    t0 = time.perf_counter()
    stats = ingest_all(conn, cfg.claude_projects_dir, embedder=embedder, cache=cache)
    dt = time.perf_counter() - t0
    console.print(
        f"[green]indexed[/green] files_seen={stats.files_seen} "
        f"skipped={stats.files_skipped} indexed={stats.files_indexed} "
        f"chunks={stats.chunks_written} embedded={stats.chunks_embedded} "
        f"in {dt:.2f}s"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n"),
    project: str | None = typer.Option(None, "--project", "-p"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="bm25 | vector | hybrid"),
) -> None:
    """Search indexed conversations."""
    cfg, conn, embedder = _open(with_vec=mode != "bm25")
    if mode == "bm25":
        hits = bm25.search(conn, query, limit=limit, project=project)
    elif mode == "vector":
        assert embedder is not None
        hits = vector.search(conn, embedder, query, limit=limit, project=project)
    else:
        assert embedder is not None
        hits = hybrid.search(conn, embedder, query, limit=limit, project=project)

    if not hits:
        console.print("[yellow]no hits[/yellow]")
        return
    for i, h in enumerate(hits, 1):
        snippet = h.text.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        console.print(
            f"[bold]{i}.[/bold] [dim]{h.started_at[:19]}[/dim]  "
            f"[cyan]{h.project[:32]}[/cyan]  score={h.score:.4f}"
        )
        console.print(f"   {snippet}")
        console.print(f"   [dim]chunk={h.chunk_id} session={h.session_id}[/dim]")


@app.command()
def stats() -> None:
    """Show index statistics."""
    cfg, conn, _ = _open(with_vec=False)
    s = repo.stats(conn)
    try:
        n_vec = conn.execute("SELECT COUNT(*) AS c FROM chunks_vec").fetchone()["c"]
    except Exception:
        n_vec = 0
    table = Table(show_header=False)
    table.add_row("files", str(s["files"]))
    table.add_row("chunks", str(s["chunks"]))
    table.add_row("vectors", str(n_vec))
    table.add_row("projects", str(s["projects"]))
    table.add_row("db_path", str(cfg.db_path))
    console.print(table)


if __name__ == "__main__":
    app()
