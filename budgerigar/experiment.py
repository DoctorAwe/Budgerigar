from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def collect_run_metadata(
    manifest: str | Path | None = None,
    extra: dict | None = None,
    repository: str | Path | None = None,
) -> dict:
    repository_path = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
    git_prefix = ["git", "-c", f"safe.directory={repository_path}"]
    git_commit = _command([*git_prefix, "rev-parse", "HEAD"], cwd=repository_path)
    git_status = _command([*git_prefix, "status", "--porcelain"], cwd=repository_path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "colab": "COLAB_RELEASE_TAG" in os.environ,
        "colab_release": os.environ.get("COLAB_RELEASE_TAG"),
        "git_commit": git_commit,
        "git_dirty": None if git_status is None else bool(git_status),
        "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    }
    try:
        import torch
        metadata.update({"torch": torch.__version__, "cuda": torch.version.cuda})
    except ImportError:
        metadata.update({"torch": None, "cuda": None})
    if manifest is not None:
        metadata["manifest"] = str(Path(manifest).resolve())
        metadata["manifest_sha256"] = file_sha256(manifest)
    if extra:
        metadata.update(extra)
    return metadata


def _command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, cwd=cwd,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_run_metadata(
    destination: str | Path,
    manifest: str | Path | None = None,
    extra: dict | None = None,
    repository: str | Path | None = None,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            collect_run_metadata(manifest, extra, repository), ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return destination
