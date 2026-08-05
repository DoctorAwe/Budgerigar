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


def collect_run_metadata(manifest: str | Path | None = None) -> dict:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "colab": "COLAB_RELEASE_TAG" in os.environ,
        "colab_release": os.environ.get("COLAB_RELEASE_TAG"),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
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
    return metadata


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_run_metadata(destination: str | Path, manifest: str | Path | None = None) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(collect_run_metadata(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination

