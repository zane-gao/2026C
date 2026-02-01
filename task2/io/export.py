from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import csv
import json
import hashlib

import numpy as np


def hash_dict(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_npz(path: str | Path, **arrays: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def save_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
