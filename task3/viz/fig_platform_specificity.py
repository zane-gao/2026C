from __future__ import annotations


def plot(path, data) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    df = data.get("platform_effects") if isinstance(data, dict) else None
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["platform"], df["estimate"], color="#55A868")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Effect")
    ax.set_title("Platform Specificity")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
