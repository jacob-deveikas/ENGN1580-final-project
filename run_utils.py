from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _slug(text: str) -> str:
    text = str(text).strip().lower()
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
    path = Path(mode).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(run_dir: Optional[Path], filename: str, fallback: str | None = None) -> Optional[str]:
    if run_dir is not None:
        return str((run_dir / filename).resolve())
    if fallback is None:
        return None
    return str(Path(fallback).expanduser().resolve())


def write_json(path: str | Path, obj: Dict[str, Any]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    def default(o: Any):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        return str(o)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=default)
    return str(out)


def write_text(path: str | Path, text: str) -> str:
    out = Path(path).expanduser()
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


def latest_run_dir(base: str | Path = "runs", contains: str | None = None) -> Optional[Path]:
    root = Path(base)
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if contains:
        dirs = [p for p in dirs if contains in p.name]
    if not dirs:
        return None
    return sorted(dirs)[-1]
