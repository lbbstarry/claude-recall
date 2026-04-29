"""recall CLI."""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.table import Table

from claude_recall.config import load_config
from claude_recall.ingest.pipeline import ingest_all
from claude_recall.search import bm25
from claude_recall.store import db, repo

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def index() -> None:
    """Scan ~/.claude/projects/ and index new/changed sessions."""
    cfg = load_config()
    conn = db.connect(cfg.db_path)
    t0 = time.perf_counter()
    stats = ingest_all(conn, cfg.claude_projects_dir)
    dt = time.perf_counter() - t0
    console.print(
        f"[green]indexed[/green] files_seen={stats.files_seen} "
        f"skipped={stats.files_skipped} indexed={stats.files_indexed} "
        f"chunks={stats.chunks_written} in {dt:.2f}s"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n"),
    project: str | None = typer.Option(None, "--project", "-p"),
) -> None:
    """Search indexed conversations (BM25)."""
    cfg = load_config()
    conn = db.connect(cfg.db_path)
    hits = bm25.search(conn, query, limit=limit, project=project)
    if not hits:
        console.print("[yellow]no hits[/yellow]")
        return
    for i, h in enumerate(hits, 1):
        snippet = h.text.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        console.print(
            f"[bold]{i}.[/bold] [dim]{h.started_at[:19]}[/dim]  "
            f"[cyan]{h.project[:32]}[/cyan]  score={h.score:.2f}"
        )
        console.print(f"   {snippet}")
        console.print(f"   [dim]chunk={h.chunk_id} session={h.session_id}[/dim]")


@app.command()
def stats() -> None:
    """Show index statistics."""
    cfg = load_config()
    conn = db.connect(cfg.db_path)
    s = repo.stats(conn)
    table = Table(show_header=False)
    table.add_row("files", str(s["files"]))
    table.add_row("chunks", str(s["chunks"]))
    table.add_row("projects", str(s["projects"]))
    table.add_row("db_path", str(cfg.db_path))
    console.print(table)


if __name__ == "__main__":
    app()
