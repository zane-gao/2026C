from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SOCIAL_TOTAL_CELEB = "celebrity_total_followers_wikidata"
SOCIAL_TOTAL_PARTNER = "partner_total_followers_wikidata"

PROXY_P_CELEB = "P_cele"
PROXY_P_PARTNER = "P_partner"

KEY_SEASON = "season"
KEY_CELEB = "celebrity_name"
KEY_PARTNER = "ballroom_partner"


def _normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_name(value: str) -> str:
    return _normalize_name(value)


def _parse_number(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() == "N/A":
        return None
    text = text.replace(",", "")
    try:
        num = float(text)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _read_text_with_fallback(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_rows_csv(path: Path) -> List[Dict[str, str]]:
    text = _read_text_with_fallback(path)
    lines = text.splitlines()
    reader = csv.DictReader(lines)
    return [dict(row) for row in reader]


def _read_rows_xlsx(path: Path) -> List[Dict[str, str]]:
    try:
        import openpyxl  # type: ignore
    except Exception as exc:
        raise RuntimeError("openpyxl is required to read .xlsx files") from exc

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        return []
    headers = [str(h).strip() if h is not None else "" for h in header]
    rows: List[Dict[str, str]] = []
    for row in rows_iter:
        rec: Dict[str, str] = {}
        for i, key in enumerate(headers):
            if key == "":
                continue
            rec[key] = row[i] if i < len(row) else None
        rows.append(rec)
    return rows


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return _read_rows_xlsx(path)
    return _read_rows_csv(path)


def _has_any_column(row: Dict[str, str], columns: Iterable[str]) -> bool:
    row_keys = set(row.keys())
    return any(col in row_keys for col in columns)


def load_social_proxy_map(path: str) -> Dict[Tuple[int, str, str], Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return {}
    rows = _read_rows(p)
    if not rows:
        return {}

    first = rows[0]
    has_proxy = _has_any_column(first, [PROXY_P_CELEB, PROXY_P_PARTNER])
    has_raw = _has_any_column(first, [SOCIAL_TOTAL_CELEB, SOCIAL_TOTAL_PARTNER])
    if not has_proxy and not has_raw:
        return {}

    out: Dict[Tuple[int, str, str], Dict[str, object]] = {}
    for row in rows:
        try:
            season = int(str(row.get(KEY_SEASON, "")).strip() or 0)
        except Exception:
            season = 0
        if season <= 0:
            continue
        celeb_raw = str(row.get(KEY_CELEB, "")).strip()
        partner_raw = str(row.get(KEY_PARTNER, "")).strip()
        celeb = _normalize_name(celeb_raw)
        partner = _normalize_name(partner_raw)
        if celeb == "" and partner == "":
            continue

        raw_cele = _parse_number(row.get(SOCIAL_TOTAL_CELEB)) if has_raw else None
        raw_partner = _parse_number(row.get(SOCIAL_TOTAL_PARTNER)) if has_raw else None
        p_cele = _parse_number(row.get(PROXY_P_CELEB)) if has_proxy else None
        p_partner = _parse_number(row.get(PROXY_P_PARTNER)) if has_proxy else None

        if p_cele is None and raw_cele is not None and raw_cele > 0:
            p_cele = math.log1p(raw_cele)
        if p_partner is None and raw_partner is not None and raw_partner > 0:
            p_partner = math.log1p(raw_partner)

        missing_cele = raw_cele is None or raw_cele <= 0
        missing_partner = raw_partner is None or raw_partner <= 0

        key = (season, celeb, partner)
        rec = out.get(key)
        if rec is None:
            out[key] = {
                KEY_SEASON: season,
                KEY_CELEB: celeb_raw,
                KEY_PARTNER: partner_raw,
                SOCIAL_TOTAL_CELEB: raw_cele,
                SOCIAL_TOTAL_PARTNER: raw_partner,
                PROXY_P_CELEB: p_cele,
                PROXY_P_PARTNER: p_partner,
                "missing_cele_total": missing_cele,
                "missing_partner_total": missing_partner,
            }
            continue

        if rec.get(PROXY_P_CELEB) is None and p_cele is not None:
            rec[PROXY_P_CELEB] = p_cele
        if rec.get(PROXY_P_PARTNER) is None and p_partner is not None:
            rec[PROXY_P_PARTNER] = p_partner
        if rec.get(SOCIAL_TOTAL_CELEB) is None and raw_cele is not None:
            rec[SOCIAL_TOTAL_CELEB] = raw_cele
            rec["missing_cele_total"] = missing_cele
        if rec.get(SOCIAL_TOTAL_PARTNER) is None and raw_partner is not None:
            rec[SOCIAL_TOTAL_PARTNER] = raw_partner
            rec["missing_partner_total"] = missing_partner

    return out


def save_social_proxy_csv(proxy_map: Dict[Tuple[int, str, str], Dict[str, object]], out_path: Path) -> None:
    rows = list(proxy_map.values())
    rows.sort(key=lambda r: (r.get(KEY_SEASON, 0), r.get(KEY_CELEB, ""), r.get(KEY_PARTNER, "")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        KEY_SEASON,
        KEY_CELEB,
        KEY_PARTNER,
        SOCIAL_TOTAL_CELEB,
        SOCIAL_TOTAL_PARTNER,
        PROXY_P_CELEB,
        PROXY_P_PARTNER,
        "missing_cele_total",
        "missing_partner_total",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
