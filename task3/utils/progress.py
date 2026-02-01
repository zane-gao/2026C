from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def iter_with_progress(iterable: Iterable[T], total: int | None = None, label: str = "") -> Iterable[T]:
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=label) if label else tqdm(iterable, total=total)
    except Exception:
        return _basic_progress(iterable, total, label)


def _basic_progress(iterable: Iterable[T], total: int | None, label: str) -> Iterator[T]:
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
    if total is None or total <= 0:
        for item in iterable:
            yield item
        return

    step = max(1, total // 50)
    prefix = f"{label} " if label else ""
    for idx, item in enumerate(iterable, 1):
        if idx == 1 or idx == total or idx % step == 0:
            pct = (idx / total) * 100.0
            bar_len = 20
            fill = int(bar_len * idx / total)
            bar = "#" * fill + "-" * (bar_len - fill)
            print(f"\r{prefix}[{bar}] {idx}/{total} {pct:5.1f}% ", end="", flush=True)
        yield item
    print()
