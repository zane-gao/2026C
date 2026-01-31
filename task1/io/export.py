from __future__ import annotations

from pathlib import Path
import json


def save_idata(idata, path: str):
    try:
        import arviz as az  # type: ignore
    except Exception as exc:
        raise RuntimeError("arviz required to save idata") from exc
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, p)


def save_json(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
