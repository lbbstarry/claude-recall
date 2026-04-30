"""recall CLI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from claude_recall.config import load_config
from claude_recall.embed.cache import EmbedCache
from claude_recall.embed.local import LocalEmbedder
from claude_recall.ingest.pipeline import ingest_all
from claude_recall.search import bm25, hybrid, vector
from claude_recall.search.bm25 import SearchHit
from claude_recall.search.filters import parse as parse_filters
from claude_recall.search.rerank import Reranker
from claude_recall.store import db, repo
from claude_recall.store import session as session_store

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _open(*, with_vec: bool = True):
    cfg = load_config()
    if with_vec:
        embedder = LocalEmbedder(DEFAULT_MODEL)
        conn = db.connect(cfg.db_path, vector_dim=embedder.dim)
        return cfg, conn, embedder
    return cfg, db.connect(cfg.db_path), None


def _print_hits(hits: list[SearchHit]) -> None:
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
        console.print(f"   [dim]chunk={h.chunk_id[:12]} session={h.session_id[:12]}[/dim]")


@app.command()
def index(no_embed: bool = typer.Option(False, "--no-embed", help="BM25 only, skip embeddings")) -> None:
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
    query: str = typer.Argument(..., help="Query (supports project: since: role: tool: filters)"),
    limit: int = typer.Option(10, "--limit", "-n"),
    project: str | None = typer.Option(None, "--project", "-p"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="bm25 | vector | hybrid"),
    rerank: bool = typer.Option(False, "--rerank", help="Cross-encoder rerank top results"),
) -> None:
    """Search indexed conversations."""
    cleaned, filters = parse_filters(query)
    cfg, conn, embedder = _open(with_vec=mode != "bm25")

    fetch = limit * 3 if rerank else limit
    if mode == "bm25":
        hits = bm25.search(conn, cleaned, limit=fetch, filters=filters, project=project)
    elif mode == "vector":
        assert embedder is not None
        hits = vector.search(conn, embedder, cleaned, limit=fetch, filters=filters, project=project)
    else:
        assert embedder is not None
        hits = hybrid.search(
            conn, embedder, cleaned, limit=fetch, filters=filters, project=project
        )

    if rerank and hits:
        hits = Reranker(DEFAULT_RERANKER).rerank(cleaned, hits, top_k=limit)
    else:
        hits = hits[:limit]

    _print_hits(hits)


@app.command()
def show(
    session_id: str = typer.Argument(..., help="Session id (full or 8+ char prefix)"),
    turn: int | None = typer.Option(None, "--turn", "-t", help="Show only this turn index"),
) -> None:
    """Render a full session as markdown."""
    _, conn, _ = _open(with_vec=False)
    try:
        chunks = session_store.list_session_chunks(conn, session_id, turn=turn)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    if not chunks:
        console.print(f"[yellow]no chunks for session {session_id}[/yellow]")
        raise typer.Exit(1)
    md = session_store.render_session_markdown(chunks)
    console.print(Markdown(md))


@app.command()
def inject(
    chunk_id: str = typer.Argument(..., help="Chunk id (full or 8+ char prefix)"),
) -> None:
    """Copy a chunk's text to the system clipboard."""
    _, conn, _ = _open(with_vec=False)
    try:
        c = session_store.get_chunk(conn, chunk_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    if c is None:
        console.print(f"[red]chunk not found: {chunk_id}[/red]")
        raise typer.Exit(1)

    try:
        import pyperclip

        pyperclip.copy(c.text)
        console.print(
            f"[green]copied[/green] turn {c.turn_index} of session "
            f"{c.session_id[:12]}  ({len(c.text)} chars)"
        )
    except Exception:
        # Fallback: print to stdout so user can pipe it
        console.print(
            "[yellow]clipboard unavailable; printing to stdout (use `recall inject … | pbcopy` etc.)[/yellow]"
        )
        sys.stdout.write(c.text)


@app.command()
def export(
    session_id: str = typer.Argument(...),
    out: Path = typer.Option(Path("-"), "--out", "-o", help="Output path or '-' for stdout"),
) -> None:
    """Export a session as markdown."""
    _, conn, _ = _open(with_vec=False)
    try:
        chunks = session_store.list_session_chunks(conn, session_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    if not chunks:
        console.print(f"[red]no chunks for session {session_id}[/red]")
        raise typer.Exit(1)
    md = session_store.render_session_markdown(chunks)
    if str(out) == "-":
        sys.stdout.write(md)
    else:
        out.write_text(md, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}  ({len(chunks)} turns)")


@app.command()
def watch() -> None:
    """Watch ~/.claude/projects/ and re-index on change (debounced)."""
    from claude_recall.watch import is_wsl, watch_loop

    cfg = load_config()
    embedder = LocalEmbedder(DEFAULT_MODEL)
    conn = db.connect(cfg.db_path, vector_dim=embedder.dim)
    cache = EmbedCache(cfg.data_dir / "embed_cache.db")

    def reindex() -> None:
        t0 = time.perf_counter()
        stats = ingest_all(conn, cfg.claude_projects_dir, embedder=embedder, cache=cache)
        if stats.files_indexed:
            console.print(
                f"[green]∆[/green] indexed={stats.files_indexed} "
                f"chunks={stats.chunks_written} embedded={stats.chunks_embedded} "
                f"in {time.perf_counter() - t0:.1f}s"
            )

    mode = "polling (WSL2)" if is_wsl() else "inotify"
    console.print(f"[bold]watching[/bold] {cfg.claude_projects_dir}  ({mode})  Ctrl+C to stop")
    reindex()
    watch_loop(cfg.claude_projects_dir, reindex)


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
