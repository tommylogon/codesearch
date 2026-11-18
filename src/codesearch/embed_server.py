"""Direct llama.cpp (llama-server) launcher for high-throughput parallel embeddings."""
from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import config

DEFAULT_PRESET = "embeddinggemma-300m"
DEFAULT_PORT = 1236


@dataclass(frozen=True)
class LlamaModelSpec:
    gguf_path: Path
    model_id: str
    parallel: int = 8
    slot_ctx: int = 2048
    batch: int = 4096
    ubatch: int = 2048
    max_chars: int = 3200
    reranking: bool = False


_PRESETS: dict[str, dict] = {
    "embeddinggemma-300m": {
        "model_id": "text-embedding-embeddinggemma-300m",
        "gguf_names": ("embeddinggemma-300M-Q8_0.gguf", "embeddinggemma-300M-F16.gguf"),
        "parallel": 16,
        "slot_ctx": 2048,
        "batch": 4096,
        "ubatch": 2048,
        "max_chars": 3200,
    },
    "embeddinggemma-2": {
        "model_id": "embeddinggemma-2",
        "gguf_names": ("embeddinggemma-2-Q8_0.gguf",),
        "parallel": 8,
        "slot_ctx": 2048,
        "batch": 4096,
        "ubatch": 2048,
        "max_chars": 3200,
    },
    "qwen3-0.6b": {
        "model_id": "text-embedding-qwen3-embedding-0.6b",
        "gguf_names": ("Qwen3-Embedding-0.6B-Q8_0.gguf", "Qwen3-Embedding-0.6B-Q4_K_M.gguf"),
        "parallel": 8,
        "slot_ctx": 2048,
        "batch": 4096,
        "ubatch": 2048,
        "max_chars": 4000,
    },
    "nomic-embed-text-v1.5": {
        "model_id": "text-embedding-nomic-embed-text-v1.5@f32",
        "gguf_names": ("nomic-embed-text-v1.5.f32.gguf", "nomic-embed-text-v1.5.Q8_0.gguf"),
        "parallel": 8,
        "slot_ctx": 2048,
        "batch": 4096,
        "ubatch": 2048,
        "max_chars": 3200,
    },
    "qwen3-reranker-0.6b": {
        "model_id": "Qwen3-Reranker-0.6B",
        "gguf_names": (
            "Qwen3-Reranker-0.6B-Q8_0.gguf",
            "Qwen3-Reranker-0.6B-Q4_K_M.gguf",
            "Qwen3-Reranker-0.6B-F16.gguf",
        ),
        "parallel": 4,
        "slot_ctx": 2048,
        "batch": 2048,
        "ubatch": 2048,
        "max_chars": 2000,
        "reranking": True,
    },
}

_PRESET_ALIASES = {
    "text-embedding-embeddinggemma-300m": "embeddinggemma-300m",
    "gemma-300m": "embeddinggemma-300m",
    "text-embedding-qwen3-embedding-0.6b": "qwen3-0.6b",
    "qwen3": "qwen3-0.6b",
    "nomic": "nomic-embed-text-v1.5",
    "qwen3-reranker": "qwen3-reranker-0.6b",
    "reranker": "qwen3-reranker-0.6b",
}


def resolve_llama_server_bin(explicit: str | Path | None = None) -> Path:
    """Locate the compiled llama-server binary."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if config.LLAMA_SERVER_BIN:
        candidates.append(Path(config.LLAMA_SERVER_BIN).expanduser())
    which = shutil.which("llama-server")
    if which:
        candidates.append(Path(which))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"llama-server executable not found (checked {config.LLAMA_SERVER_BIN!r} and PATH); "
        f"set CODESEARCH_LLAMA_SERVER to your built binary"
    )


def _find_gguf(filenames: tuple[str, ...]) -> Path | None:
    for root_str in config.MODEL_SEARCH_DIRS:
        root = Path(root_str).expanduser()
        if not root.exists():
            continue
        for name in filenames:
            direct = root / name
            if direct.is_file():
                return direct
            for match in root.rglob(name):
                if match.is_file():
                    return match
    return None


def resolve_gguf_spec(spec: str | None = None, *, model_id_override: str | None = None) -> LlamaModelSpec:
    """Resolve a preset name or explicit .gguf path into a concrete LlamaModelSpec."""
    raw = (spec or config.EMBED_GGUF or DEFAULT_PRESET).strip()
    key = _PRESET_ALIASES.get(raw.lower(), raw.lower())
    if key in _PRESETS:
        preset = _PRESETS[key]
        gguf = _find_gguf(preset["gguf_names"])
        if gguf is None:
            raise FileNotFoundError(
                f"GGUF file for preset {key!r} ({preset['gguf_names']}) not found under {config.MODEL_SEARCH_DIRS}"
            )
        return LlamaModelSpec(
            gguf_path=gguf,
            model_id=model_id_override or preset["model_id"],
            parallel=preset["parallel"],
            slot_ctx=preset["slot_ctx"],
            batch=preset["batch"],
            ubatch=preset["ubatch"],
            max_chars=preset["max_chars"],
            reranking=bool(preset.get("reranking", False)),
        )

    path = Path(raw).expanduser()
    if path.is_file():
        return LlamaModelSpec(
            gguf_path=path.resolve(),
            model_id=model_id_override or path.stem,
        )
    raise FileNotFoundError(
        f"Unknown preset or missing GGUF file {raw!r}; known presets: {', '.join(sorted(_PRESETS))}"
    )


def build_llama_command(
    spec: LlamaModelSpec,
    *,
    server_bin: Path,
    port: int,
    host: str = "127.0.0.1",
    gpu_layers: int = 99,
    parallel: int | None = None,
    slot_ctx: int | None = None,
    batch: int | None = None,
    ubatch: int | None = None,
    flash_attn: str = "on",
) -> list[str]:
    """Build the llama-server CLI command for multi-slot embedding serving."""
    slots = parallel if parallel is not None else spec.parallel
    ctx_per_slot = slot_ctx if slot_ctx is not None else spec.slot_ctx
    total_ctx = max(ctx_per_slot, ctx_per_slot * max(1, slots))
    b = batch if batch is not None else spec.batch
    ub = ubatch if ubatch is not None else spec.ubatch
    return [
        str(server_bin),
        "-m", str(spec.gguf_path),
        "--alias", spec.model_id,
        "--reranking" if spec.reranking else "--embedding",
        "-ngl", str(gpu_layers),
        "-np", str(slots),
        "-c", str(total_ctx),
        "-b", str(b),
        "-ub", str(ub),
        "-fa", flash_attn,
        "--host", host,
        "--port", str(port),
    ]


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _tail_text(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


class LlamaEmbedServer:
    """Managed llama-server subprocess that configures codesearch while active."""

    def __init__(
        self,
        spec: LlamaModelSpec,
        *,
        server_bin: Path | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        gpu_layers: int = 99,
        parallel: int | None = None,
        slot_ctx: int | None = None,
        batch: int | None = None,
        ubatch: int | None = None,
        req_batch: int = 32,
        workers: int | None = None,
        startup_timeout: float = 45.0,
    ):
        self.spec = spec
        self.server_bin = server_bin or resolve_llama_server_bin()
        self.host = host
        self.port = port if port is not None else _free_port(host)
        self.gpu_layers = gpu_layers
        self.parallel = parallel if parallel is not None else spec.parallel
        self.slot_ctx = slot_ctx if slot_ctx is not None else spec.slot_ctx
        self.batch = batch if batch is not None else spec.batch
        self.ubatch = ubatch if ubatch is not None else spec.ubatch
        self.req_batch = req_batch
        self.workers = workers if workers is not None else self.parallel
        self.startup_timeout = startup_timeout
        self.endpoint = f"http://{self.host}:{self.port}/v1"
        self.proc: subprocess.Popen | None = None
        self._log_file = None
        self._log_path: Path | None = None
        self._saved_config: dict | None = None

    def start(self) -> "LlamaEmbedServer":
        cmd = build_llama_command(
            self.spec,
            server_bin=self.server_bin,
            port=self.port,
            host=self.host,
            gpu_layers=self.gpu_layers,
            parallel=self.parallel,
            slot_ctx=self.slot_ctx,
            batch=self.batch,
            ubatch=self.ubatch,
        )
        fd, raw_path = tempfile.mkstemp(prefix="codesearch_llama_", suffix=".log")
        self._log_path = Path(raw_path)
        self._log_file = open(fd, "w", encoding="utf-8", errors="replace", closefd=True)
        self.proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT)

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = _tail_text(self._log_path)
                self.stop()
                raise RuntimeError(f"llama-server exited early ({self.proc.returncode}):\n{tail}")
            try:
                r = requests.get(f"{self.endpoint}/models", timeout=1.0)
                if r.status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.25)

        tail = _tail_text(self._log_path)
        self.stop()
        raise TimeoutError(f"llama-server did not become ready within {self.startup_timeout}s:\n{tail}")

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
            self.proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        if self._log_path is not None:
            try:
                self._log_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._log_path = None

    def __enter__(self) -> "LlamaEmbedServer":
        self._saved_config = {
            "EMBED_ENDPOINT": config.EMBED_ENDPOINT,
            "EMBED_MODEL": config.EMBED_MODEL,
            "EMBED_WORKERS": config.EMBED_WORKERS,
            "EMBED_BATCH": config.EMBED_BATCH,
            "EMBED_MAX_CHARS": config.EMBED_MAX_CHARS,
        }
        self.start()
        config.EMBED_ENDPOINT = self.endpoint
        config.EMBED_MODEL = self.spec.model_id
        config.EMBED_WORKERS = self.workers
        config.EMBED_BATCH = self.req_batch
        config.EMBED_MAX_CHARS = self.spec.max_chars
        return self

    def __exit__(self, *_exc) -> None:
        try:
            self.stop()
        finally:
            if self._saved_config is not None:
                for key, val in self._saved_config.items():
                    setattr(config, key, val)
                self._saved_config = None


def serve_forever(
    model_or_gguf: str | None = None,
    *,
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    gpu_layers: int = 99,
    parallel: int | None = None,
    slot_ctx: int | None = None,
    batch: int | None = None,
    ubatch: int | None = None,
) -> int:
    """Start llama-server for embeddings and keep running until interrupted."""
    spec = resolve_gguf_spec(model_or_gguf)
    srv = LlamaEmbedServer(
        spec,
        host=host,
        port=port,
        gpu_layers=gpu_layers,
        parallel=parallel,
        slot_ctx=slot_ctx,
        batch=batch,
        ubatch=ubatch,
    )
    srv.start()
    print(
        f"[embed-server] serving {spec.model_id} from {spec.gguf_path.name} "
        f"at {srv.endpoint} (slots={srv.parallel}, slot_ctx={srv.slot_ctx}, ub={srv.ubatch})",
        flush=True,
    )
    try:
        while srv.proc is not None and srv.proc.poll() is None:
            time.sleep(0.5)
        return srv.proc.returncode if srv.proc is not None and srv.proc.returncode is not None else 0
    except KeyboardInterrupt:
        return 0
    finally:
        srv.stop()
