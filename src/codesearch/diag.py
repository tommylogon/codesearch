"""Smart, high-verbosity timing and stall diagnostics for codesearch.

Writes rotating UTF-8 logs with millisecond timestamps, PID/thread tags, stage
timings, and an automatic stall watchdog (`watch_slow`) that dumps live Python
stack traces while an operation is still blocked. Never writes to stdout so
stdio transports (such as FastMCP) stay uncorrupted.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import config

_ROOT_NAME = "codesearch"
_LOCK = threading.RLock()
_CURRENT_LOG_PATH: Path | None = None
_FILE_HANDLER: RotatingFileHandler | None = None
_STDERR_HANDLER: logging.Handler | None = None

_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(process)d:%(threadName)s] %(levelname)-5s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_level(raw: str | None) -> int:
    val = (raw or os.environ.get("CODESEARCH_LOG_LEVEL", "DEBUG")).strip().upper()
    return getattr(logging, val, logging.DEBUG)


def default_log_path(component: str = "codesearch", log_dir: Path | str | None = None) -> Path:
    """Resolve the log file path cleanly without failing on read-only workspaces."""
    override = os.environ.get("CODESEARCH_LOG_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    base = Path(log_dir) if log_dir else config.DATA_DIR
    return base / f"{component}.log"


def get_logger(subname: str = "") -> logging.Logger:
    """Return a logger inside the `codesearch.*` hierarchy."""
    name = f"{_ROOT_NAME}.{subname}" if subname else _ROOT_NAME
    return logging.getLogger(name)


def current_log_path() -> Path | None:
    return _CURRENT_LOG_PATH


def configure(
    component: str = "codesearch",
    *,
    log_path: Path | str | None = None,
    log_dir: Path | str | None = None,
    level: str | None = None,
    max_bytes: int = 8 * 1024 * 1024,
    backup_count: int = 4,
) -> logging.Logger:
    """Attach (or retarget) a rotating file handler cleanly on the codesearch root logger."""
    global _CURRENT_LOG_PATH, _FILE_HANDLER, _STDERR_HANDLER
    root = logging.getLogger(_ROOT_NAME)
    lvl = _parse_level(level)

    with _LOCK:
        root.setLevel(lvl)
        root.propagate = False
        target = Path(log_path).expanduser() if log_path else default_log_path(component, log_dir=log_dir)

        if _FILE_HANDLER is None or _CURRENT_LOG_PATH != target:
            if _FILE_HANDLER is not None:
                root.removeHandler(_FILE_HANDLER)
                try:
                    _FILE_HANDLER.close()
                except Exception:
                    pass
                _FILE_HANDLER = None

            handler: RotatingFileHandler | None = None
            for candidate in (
                target,
                Path(tempfile.gettempdir()) / "codesearch_logs" / target.name,
            ):
                try:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    handler = RotatingFileHandler(
                        str(candidate),
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                        encoding="utf-8",
                        delay=False,
                    )
                    target = candidate
                    break
                except OSError:
                    continue

            if handler is not None:
                handler.setLevel(lvl)
                handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
                root.addHandler(handler)
                _FILE_HANDLER = handler
                _CURRENT_LOG_PATH = target
            elif not root.handlers:
                root.addHandler(logging.NullHandler())
                _CURRENT_LOG_PATH = None
        else:
            _FILE_HANDLER.setLevel(lvl)

        want_stderr = os.environ.get("CODESEARCH_LOG_STDERR", "").strip().lower() in ("1", "true", "yes")
        if want_stderr and _STDERR_HANDLER is None:
            sh = logging.StreamHandler(sys.stderr)
            sh.setLevel(lvl)
            sh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            root.addHandler(sh)
            _STDERR_HANDLER = sh
        elif not want_stderr and _STDERR_HANDLER is not None:
            root.removeHandler(_STDERR_HANDLER)
            _STDERR_HANDLER = None

    return get_logger(component if component != _ROOT_NAME else "")


def flush() -> None:
    """Flush buffered log lines immediately so crashes/kills keep recent diagnostics."""
    with _LOCK:
        if _FILE_HANDLER is not None:
            try:
                _FILE_HANDLER.flush()
            except Exception:
                pass


def format_timings(timings: dict[str, Any]) -> str:
    """Render a compact key=value summary of stage timings."""
    parts: list[str] = []
    for key, val in timings.items():
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.1f}")
        elif isinstance(val, (dict, list, tuple)):
            parts.append(f"{key}={val!r}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)


@contextmanager
def watch_slow(
    label: str,
    *,
    logger: logging.Logger | None = None,
    warn_after: float | None = None,
    repeat_every: float = 15.0,
):
    """Dump the blocked thread's stack frame to the log if an operation stalls past `warn_after`."""
    threshold = config.SLOW_WATCHDOG_SECS if warn_after is None else float(warn_after)
    if threshold <= 0:
        yield
        return

    log = logger or get_logger("watchdog")
    tid = threading.get_ident()
    tname = threading.current_thread().name
    started = time.perf_counter()
    done = threading.Event()
    timer_lock = threading.Lock()
    state: dict[str, threading.Timer | None] = {"timer": None}

    def _fire(next_wait: float) -> None:
        if done.is_set():
            return
        elapsed = time.perf_counter() - started
        frame = sys._current_frames().get(tid)
        stack_txt = "unavailable"
        if frame is not None:
            frames = traceback.extract_stack(frame)
            tail = frames[-8:]
            stack_txt = " <- ".join(
                f"{Path(f.filename).name}:{f.lineno}({f.name})" for f in reversed(tail)
            )
        log.warning(
            "STALL_WATCHDOG op=%s elapsed=%.2fs thread=%s(%s) stack=%s",
            label,
            elapsed,
            tname,
            tid,
            stack_txt,
        )
        flush()
        with timer_lock:
            if not done.is_set() and repeat_every > 0:
                nxt = threading.Timer(repeat_every, _fire, args=(repeat_every,))
                nxt.daemon = True
                nxt.name = f"watchdog-{label[:16]}"
                state["timer"] = nxt
                nxt.start()

    first = threading.Timer(threshold, _fire, args=(repeat_every,))
    first.daemon = True
    first.name = f"watchdog-{label[:16]}"
    state["timer"] = first
    first.start()
    try:
        yield
    finally:
        done.set()
        with timer_lock:
            if state["timer"] is not None:
                state["timer"].cancel()


def log_startup_banner(
    logger: logging.Logger,
    *,
    component: str,
    repo_root: Path | str,
    db_path: Path | str | None,
    conn=None,
) -> None:
    """Log workspace/DB/index/endpoint metadata once on startup or root switch."""
    root = Path(repo_root)
    dbp = Path(db_path) if db_path else None
    db_mb = 0.0
    wal_mb = 0.0
    if dbp and dbp.exists():
        db_mb = dbp.stat().st_size / (1024 * 1024)
        walp = dbp.with_name(dbp.name + "-wal")
        if walp.exists():
            wal_mb = walp.stat().st_size / (1024 * 1024)

    has_parent_idx = None
    has_vec_model_idx = None
    files_cnt = None
    jmode = None
    if conn is not None:
        try:
            idx_names = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            }
            has_parent_idx = "idx_nodes_parent" in idx_names
            has_vec_model_idx = "idx_vec_cache_model" in idx_names
            files_cnt = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            jmode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        except Exception as exc:
            logger.warning("STARTUP_INSPECT_ERR component=%s err=%s", component, exc)

    logger.info(
        "STARTUP component=%s pid=%d log=%s root=%s db=%s db_mb=%.1f wal_mb=%.1f "
        "files=%s journal=%s idx_parent=%s idx_vec_model=%s embed_ep=%s embed_model=%s "
        "query_timeout=(%.1f,%.1f) img_ep=%s img_model=%s",
        component,
        os.getpid(),
        _CURRENT_LOG_PATH,
        root,
        dbp,
        db_mb,
        wal_mb,
        files_cnt,
        jmode,
        has_parent_idx,
        has_vec_model_idx,
        config.EMBED_ENDPOINT,
        config.EMBED_MODEL,
        config.EMBED_CONNECT_TIMEOUT,
        config.EMBED_QUERY_TIMEOUT,
        config.IMAGE_EMBED_ENDPOINT,
        config.IMAGE_EMBED_MODEL,
    )
    flush()


def tail_lines(path: Path | None = None, limit: int = 80) -> list[str]:
    """Read the last `limit` lines from the active or specified log file safely."""
    target = path or _CURRENT_LOG_PATH
    if target is None or not target.is_file():
        return []
    limit = max(1, min(int(limit), 500))
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    return lines[-limit:]


def shutdown() -> None:
    """Close and detach active file/stderr handlers cleanly (safe for Windows cleanup)."""
    global _CURRENT_LOG_PATH, _FILE_HANDLER, _STDERR_HANDLER
    root = logging.getLogger(_ROOT_NAME)
    with _LOCK:
        if _FILE_HANDLER is not None:
            root.removeHandler(_FILE_HANDLER)
            try:
                _FILE_HANDLER.close()
            except Exception:
                pass
            _FILE_HANDLER = None
        if _STDERR_HANDLER is not None:
            root.removeHandler(_STDERR_HANDLER)
            _STDERR_HANDLER = None
        _CURRENT_LOG_PATH = None
