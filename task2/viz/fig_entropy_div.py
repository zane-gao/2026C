from __future__ import annotations

from pathlib import Path


def plot(out_path: str | Path, data=None) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib required for plotting") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Entropy and Divergence")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if data is not None and hasattr(data, "__len__"):
        try:
            ax.plot(list(range(len(data))), list(data))
        except Exception:
            pass
    ax.grid(True, alpha=0.3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
