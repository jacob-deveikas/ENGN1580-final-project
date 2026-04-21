from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def prepare_run_dir(base: str | Path = "runs", label: str = "run", mode: str = "auto") -> Optional[Path]:
    if mode is None:
        mode = "auto"
    mode = str(mode)
    if mode.lower() in {"", "none", "off", "no"}:
        return None
    if mode.lower() == "auto":
        root = Path(base)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slug(label)
        for idx in range(1, 1000):
            path = root / f"{stamp}_{slug}_{idx:03d}"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=False)
                return path
        raise RuntimeError("could not allocate run directory")
    path = Path(mode)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_optional_path(path: str | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def artifact_path(run_dir: Optional[Path], filename: str, fallback: str | None = None) -> Optional[str]:
    if run_dir is not None:
        return str((run_dir / filename).resolve())
    if fallback is None:
        return None
    return str(Path(fallback).expanduser().resolve())


def write_json(path: str | Path, obj: Dict[str, Any]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return str(out)


def append_text(path: str | Path, text: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(text)
    return str(out)


def write_text(path: str | Path, text: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(text)
    return str(out)


def command_manifest(run_dir: Optional[Path], argv: list[str], extra: Dict[str, Any] | None = None) -> Optional[str]:
    if run_dir is None:
        return None
    obj: Dict[str, Any] = {
        "argv": argv,
        "cwd": os.getcwd(),
        "python": sys.executable,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        obj.update(extra)
    return write_json(run_dir / "manifest.json", obj)
