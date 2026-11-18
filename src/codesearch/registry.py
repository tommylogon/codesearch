"""Multi-index registry for codesearch MCP and CLI tools.

Resolves explicit `index` identifiers (short names or indexed repository paths)
from:
  1. The local registry JSON (`~/.codesearch/indexes.json` or `CODESEARCH_REGISTRY`)
  2. `CODESEARCH_INDEXES` env var (JSON object or `name=path;...` list)
  3. Explicit CLI `--index name=path` overrides
  4. Active `CODESEARCH_ROOT` / `cwd` (added to the available map, never a silent default)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import config

_log = logging.getLogger("codesearch.registry")
_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class IndexEntry:
    """One named codesearch index pointing at a repository root."""

    id: str
    path: Path
    description: str = ""
    source: str = "registry"

    @property
    def data_dir(self) -> Path:
        return self.path / ".codesearch"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "codesearch.db"

    @property
    def is_ready(self) -> bool:
        return self.db_path.is_file()


class IndexResolutionError(ValueError):
    """Raised when an explicit index selection is missing, unknown, or unindexed."""

    def __init__(self, message: str, available: dict[str, IndexEntry] | None = None):
        super().__init__(message)
        self.available: dict[str, IndexEntry] = available or {}


def slugify(name: str) -> str:
    """Normalize an index label or directory name into a lowercase identifier."""
    cleaned = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-._")
    return cleaned or "index"


def registry_path(override: Path | str | None = None) -> Path:
    """Return the machine-local JSON registry file path."""
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.environ.get("CODESEARCH_REGISTRY", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".codesearch" / "indexes.json").resolve()


def _coerce_entry(raw_id: str, raw_val: object, source: str) -> IndexEntry | None:
    if isinstance(raw_val, str):
        raw_path = raw_val
        desc = ""
    elif isinstance(raw_val, dict):
        raw_path = str(raw_val.get("path") or raw_val.get("root") or "").strip()
        desc = str(raw_val.get("description") or "").strip()
    else:
        return None
    if not raw_path:
        return None
    key = slugify(raw_id)
    resolved = Path(raw_path).expanduser().resolve()
    return IndexEntry(id=key, path=resolved, description=desc, source=source)


def _merge_mapping(target: dict[str, IndexEntry], data: object, source: str) -> None:
    if not isinstance(data, dict):
        return
    mapping = data.get("indexes") if isinstance(data.get("indexes"), dict) else data
    if not isinstance(mapping, dict):
        return
    for k, v in mapping.items():
        entry = _coerce_entry(str(k), v, source=source)
        if entry is not None:
            target[entry.id] = entry


def parse_index_specs(raw: str | Iterable[str], source: str = "env") -> dict[str, IndexEntry]:
    """Parse JSON or `name=path` specifications into IndexEntry records."""
    out: dict[str, IndexEntry] = {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return out
        if text.startswith("{"):
            try:
                _merge_mapping(out, json.loads(text), source=source)
                return out
            except json.JSONDecodeError as exc:
                _log.warning("REGISTRY_ENV_BAD_JSON err=%s", exc)
                return out
        items: list[str] = [p for p in re.split(r"[;\r\n]+", text) if p.strip()]
    else:
        items = [str(p).strip() for p in raw if str(p).strip()]

    for item in items:
        if "=" in item:
            name, path_str = item.split("=", 1)
            entry = _coerce_entry(name, path_str.strip(), source=source)
        else:
            p = Path(item).expanduser().resolve()
            entry = IndexEntry(id=slugify(p.name), path=p, source=source)
        if entry is not None:
            out[entry.id] = entry
    return out


def load_file_registry(reg_path: Path | str | None = None) -> dict[str, IndexEntry]:
    """Load persistent index definitions from the JSON registry file."""
    path = registry_path(reg_path)
    out: dict[str, IndexEntry] = {}
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("REGISTRY_READ_ERR path=%s err=%s", path, exc)
        return out
    _merge_mapping(out, data, source="file")
    return out


def save_file_registry(entries: dict[str, IndexEntry], reg_path: Path | str | None = None) -> Path:
    """Atomically write the index registry JSON file."""
    path = registry_path(reg_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexes": {
            key: {
                "path": str(entry.path),
                **({"description": entry.description} if entry.description else {}),
            }
            for key, entry in sorted(entries.items())
        }
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _unique_key(base: str, entries: dict[str, IndexEntry]) -> str:
    key = slugify(base)
    if key not in entries:
        return key
    idx = 2
    while f"{key}-{idx}" in entries:
        idx += 1
    return f"{key}-{idx}"


def _add_path_if_missing(
    entries: dict[str, IndexEntry],
    root: Path,
    *,
    source: str,
    require_db: bool,
) -> None:
    resolved = root.expanduser().resolve()
    if any(e.path == resolved for e in entries.values()):
        return
    if require_db and not (resolved / ".codesearch" / "codesearch.db").is_file():
        return
    key = _unique_key(resolved.name, entries)
    entries[key] = IndexEntry(id=key, path=resolved, source=source)


def all_entries(
    *,
    extra_specs: Iterable[str] | None = None,
    reg_path: Path | str | None = None,
    include_env_root: bool = True,
    include_cwd: bool = True,
) -> dict[str, IndexEntry]:
    """Combine file registry, env overrides, CLI specs, and active roots."""
    entries = load_file_registry(reg_path)

    env_indexes = os.environ.get("CODESEARCH_INDEXES", "").strip()
    if env_indexes:
        entries.update(parse_index_specs(env_indexes, source="env"))

    if extra_specs:
        entries.update(parse_index_specs(extra_specs, source="cli"))

    if include_env_root:
        env_root = os.environ.get("CODESEARCH_ROOT", "").strip()
        if env_root:
            _add_path_if_missing(entries, Path(env_root), source="env_root", require_db=False)

    if include_cwd:
        _add_path_if_missing(entries, Path.cwd(), source="cwd", require_db=True)

    return entries


def resolve_index(
    index: str | None,
    *,
    extra_specs: Iterable[str] | None = None,
    reg_path: Path | str | None = None,
    include_env_root: bool = True,
    include_cwd: bool = True,
) -> IndexEntry:
    """Resolve a required explicit `index` (alias or indexed repo directory)."""
    entries = all_entries(
        extra_specs=extra_specs,
        reg_path=reg_path,
        include_env_root=include_env_root,
        include_cwd=include_cwd,
    )
    raw = index.strip() if isinstance(index, str) else ""
    if not raw:
        raise IndexResolutionError(
            "Parameter 'index' is required. Call list_indexes() to see registered indexes "
            "or pass an indexed repository directory path.",
            entries,
        )

    low = raw.lower()
    if low in entries:
        return entries[low]
    slug = slugify(raw)
    if slug in entries:
        return entries[slug]

    looks_like_path = ("/" in raw) or ("\\" in raw) or (":" in raw) or Path(raw).expanduser().exists()
    if looks_like_path:
        cand = Path(raw).expanduser().resolve()
        if cand.is_file() and cand.name == "codesearch.db" and cand.parent.name == ".codesearch":
            cand = cand.parent.parent
        for entry in entries.values():
            if entry.path == cand:
                return entry
        if (cand / ".codesearch" / "codesearch.db").is_file():
            return IndexEntry(id=slugify(cand.name), path=cand, source="path")

    raise IndexResolutionError(
        f"Unknown index {raw!r}. Call list_indexes() to see available indexes.",
        entries,
    )


def upsert_entry(
    path: Path | str,
    *,
    name: str | None = None,
    description: str = "",
    reg_path: Path | str | None = None,
) -> IndexEntry:
    """Add or update an index entry in the persistent JSON registry."""
    resolved = Path(path).expanduser().resolve()
    entries = load_file_registry(reg_path)
    existing = next((e for e in entries.values() if e.path == resolved), None)
    if name:
        key = slugify(name)
        if existing is not None and existing.id != key:
            entries.pop(existing.id, None)
        desc = description or (existing.description if existing else "")
    elif existing is not None:
        key = existing.id
        desc = description or existing.description
    else:
        key = _unique_key(resolved.name, entries)
        desc = description
    entry = IndexEntry(id=key, path=resolved, description=desc, source="file")
    entries[key] = entry
    save_file_registry(entries, reg_path)
    return entry


def remove_entry(name: str, *, reg_path: Path | str | None = None) -> bool:
    """Remove an index entry by id (or path) from the persistent JSON registry."""
    entries = load_file_registry(reg_path)
    key = slugify(name)
    removed = False
    if key in entries:
        entries.pop(key)
        removed = True
    else:
        cand = Path(name).expanduser().resolve()
        for k, entry in list(entries.items()):
            if entry.path == cand:
                entries.pop(k)
                removed = True
    if removed:
        save_file_registry(entries, reg_path)
    return removed


def scan_roots(
    roots: Iterable[Path | str],
    *,
    max_depth: int = 2,
    reg_path: Path | str | None = None,
) -> list[IndexEntry]:
    """Discover `.codesearch/codesearch.db` folders up to `max_depth` and register them."""
    entries = load_file_registry(reg_path)
    added: list[IndexEntry] = []

    def _walk(base: Path, depth: int) -> None:
        if (base / ".codesearch" / "codesearch.db").is_file():
            resolved = base.resolve()
            if not any(e.path == resolved for e in entries.values()):
                key = _unique_key(resolved.name, entries)
                entry = IndexEntry(id=key, path=resolved, source="file")
                entries[key] = entry
                added.append(entry)
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in base.iterdir() if p.is_dir() and p.name not in config.IGNORE_DIRS)
        except OSError:
            return
        for child in children:
            _walk(child, depth + 1)

    for root in roots:
        p = Path(root).expanduser()
        if p.is_dir():
            _walk(p.resolve(), 0)

    if added:
        save_file_registry(entries, reg_path)
    return added


def index_summary(entries: dict[str, IndexEntry]) -> list[dict]:
    """Return lightweight filesystem metadata for every entry without opening SQLite."""
    items: list[dict] = []
    for entry in sorted(entries.values(), key=lambda e: e.id):
        dbp = entry.db_path
        ready = dbp.is_file()
        db_mb = round(dbp.stat().st_size / (1024 * 1024), 1) if ready else 0.0
        packs = (entry.data_dir / "vecpacks").is_dir()
        item: dict[str, object] = {
            "id": entry.id,
            "path": str(entry.path),
            "ready": ready,
            "db_mb": db_mb,
            "vecpacks": packs,
            "source": entry.source,
        }
        if entry.description:
            item["description"] = entry.description
        items.append(item)
    return items
