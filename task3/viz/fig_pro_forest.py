from __future__ import annotations

from typing import Optional


def plot(path, data, channel: str = "J") -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    df = data.get("pro_effects") if isinstance(data, dict) else None
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    df = df.sort_values("median", ascending=True)
    y = range(len(df))
    fig, ax = plt.subplots(figsize=(6, max(4, len(df) * 0.2)))
    ax.errorbar(df["median"], y, xerr=[df["median"] - df["q05"], df["q95"] - df["median"]], fmt="o")
    ax.set_yticks(list(y))
    if "pro_name" in df.columns:
        labels = df["pro_name"].fillna("").astype(str).tolist()
    elif "pro_id" in df.columns:
        labels = df["pro_id"].astype(str).tolist()
    else:
        labels = [str(idx) for idx in range(len(df))]
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Pro Effects ({channel})")
    ax.set_xlabel("Effect")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
