from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_csv(path: str):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for load_csv") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_csv(p, dtype=str)
    return df
