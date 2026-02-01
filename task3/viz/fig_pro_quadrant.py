from __future__ import annotations


def plot(path, data) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    df = data.get("pro_quadrant") if isinstance(data, dict) else None
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["u_pro_J"], df["u_pro_F"], alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("u_pro_J")
    ax.set_ylabel("u_pro_F")
    ax.set_title("Pro Quadrant")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
