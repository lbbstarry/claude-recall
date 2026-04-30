"""Filesystem watcher: re-index when JSONL files change."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


class _Debouncer(FileSystemEventHandler):
    def __init__(self, on_change: callable, debounce_sec: float = 1.0) -> None:
        self._on_change = on_change
        self._debounce = debounce_sec
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._on_change)
            self._timer.daemon = True
            self._timer.start()

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(".jsonl"):
            self._schedule()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(".jsonl"):
            self._schedule()


def watch_loop(projects_dir: Path, on_change: callable) -> None:
    handler = _Debouncer(on_change)
    observer = PollingObserver() if is_wsl() else Observer()
    observer.schedule(handler, str(projects_dir), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
