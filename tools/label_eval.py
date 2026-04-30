"""Interactive labeling tool for the eval set.

Strategy: random-sample chunks from your real index. For each, you write 1-3
natural-language queries that *should* retrieve it. We append to
tests/fixtures/queries.jsonl as one line per (query, relevant_chunk_id).

Usage:
    uv run python tools/label_eval.py            # random sampling
    uv run python tools/label_eval.py --target 50

Commands during labeling:
    <enter a query>      record (query → current chunk_id)
    .                    save & next chunk
    s                    skip this chunk
    q                    save & quit
    f                    show full chunk text (default shows preview)
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from claude_recall.config import load_config
from claude_recall.store import db

OUTPUT = Path(__file__).parent.parent / "tests" / "fixtures" / "queries.jsonl"

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _existing_count() -> int:
    if not OUTPUT.exists():
        return 0
    return sum(1 for _ in OUTPUT.open(encoding="utf-8") if _.strip())


def _existing_chunk_ids() -> set[str]:
    if not OUTPUT.exists():
        return set()
    out = set()
    for line in OUTPUT.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.add(json.loads(line)["relevant_chunk_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def _sample_chunks(conn: sqlite3.Connection, n: int, exclude: set[str]) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT id, project, started_at, role_mix, text, token_count
        FROM chunks
        WHERE token_count >= 40 AND token_count <= 600
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n * 3,),  # oversample so excludes don't starve us
    ).fetchall()
    out = []
    for r in rows:
        if r["id"] in exclude:
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out


def _show(row: sqlite3.Row, full: bool) -> None:
    text = row["text"]
    if not full and len(text) > 600:
        text = text[:600] + "\n…[truncated; press 'f' for full]"
    title = f"{row['project']}  ·  {row['started_at'][:19]}  ·  {row['role_mix']}  ·  ~{row['token_count']} tok"
    console.print(Panel(text, title=title, title_align="left"))
    console.print(f"[dim]chunk_id = {row['id']}[/dim]")


def _append(query: str, chunk_id: str) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "relevant_chunk_id": chunk_id}, ensure_ascii=False) + "\n")


@app.command()
def main(
    target: int = typer.Option(50, "--target", "-t", help="Stop when total reaches this."),
    pool: int = typer.Option(40, "--pool", "-p", help="How many candidate chunks to sample."),
) -> None:
    cfg = load_config()
    conn = db.connect(cfg.db_path)

    have = _existing_count()
    excl = _existing_chunk_ids()
    console.print(f"[bold]eval set:[/bold] {have}/{target} labeled  ({OUTPUT})")
    if have >= target:
        console.print("[green]target reached — exit[/green]")
        return

    rows = _sample_chunks(conn, pool, excl)
    console.print(f"sampled {len(rows)} candidate chunks\n")
    full = False

    for row in rows:
        if have >= target:
            break
        console.rule(f"chunk {have + 1}/{target}")
        _show(row, full=full)
        full = False
        recorded_for_this = 0
        while True:
            try:
                line = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]bye[/yellow]")
                return
            if line == "":
                continue
            if line == ".":
                if recorded_for_this == 0:
                    console.print("[dim](no queries recorded; moving on)[/dim]")
                break
            if line == "s":
                console.print("[dim]skipped[/dim]")
                break
            if line == "q":
                console.print(f"[green]saved {have} total[/green]")
                return
            if line == "f":
                _show(row, full=True)
                continue
            _append(line, row["id"])
            recorded_for_this += 1
            have += 1
            console.print(f"[green]✓ recorded ({have}/{target})[/green]")

    console.print(f"\n[bold]done[/bold]  total labeled: {have}/{target}")
    if have < target:
        console.print("rerun the command for another batch")


if __name__ == "__main__":
    app()
