from __future__ import annotations

import numpy as np


def plot(path, data) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    var_j = data.get("var_j", {}) if isinstance(data, dict) else {}
    var_f = data.get("var_f", {}) if isinstance(data, dict) else {}

    labels = ["season", "pro", "celeb", "resid"]
    vals_j = np.array([var_j.get(k, 0.0) for k in labels])
    vals_f = np.array([var_f.get(k, 0.0) for k in labels])
    
    # 检查是否有数据
    has_j = vals_j.sum() > 0
    has_f = vals_f.sum() > 0
    
    # 归一化为百分比（方差占比）
    if has_j and vals_j.sum() > 0:
        vals_j_pct = vals_j / vals_j.sum() * 100
    else:
        vals_j_pct = vals_j
    
    if has_f and vals_f.sum() > 0:
        vals_f_pct = vals_f / vals_f.sum() * 100
    else:
        vals_f_pct = vals_f

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：原始方差值（对数刻度如果差异太大）
    ax1 = axes[0]
    x = np.arange(len(labels))
    width = 0.35
    
    if has_j:
        bars1 = ax1.bar(x - width/2, vals_j, width, label='Judges (M1)', color='#3498db', edgecolor='white')
    if has_f:
        bars2 = ax1.bar(x + width/2, vals_f, width, label='Fans (M2)', color='#e74c3c', edgecolor='white')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Variance", fontsize=11)
    ax1.set_title("Variance Components (Absolute)", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 如果数值差异很大，使用对数刻度
    if has_j and has_f:
        max_val = max(vals_j.max(), vals_f.max())
        min_val = min(vals_j[vals_j > 0].min() if (vals_j > 0).any() else 1,
                      vals_f[vals_f > 0].min() if (vals_f > 0).any() else 1)
        if max_val / (min_val + 1e-10) > 100:
            ax1.set_yscale('log')
            ax1.set_ylabel("Variance (log scale)", fontsize=11)
    
    # 右图：百分比占比
    ax2 = axes[1]
    
    if has_j:
        bars1 = ax2.bar(x - width/2, vals_j_pct, width, label='Judges (M1)', color='#3498db', edgecolor='white')
    if has_f:
        bars2 = ax2.bar(x + width/2, vals_f_pct, width, label='Fans (M2)', color='#e74c3c', edgecolor='white')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title("Variance Decomposition (Proportion)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 100)
    
    # 添加百分比标签
    def add_labels(ax, bars, vals):
        for bar, val in zip(bars, vals):
            if val > 5:  # 只显示大于5%的标签
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    if has_j:
        add_labels(ax2, bars1, vals_j_pct)
    if has_f:
        add_labels(ax2, bars2, vals_f_pct)
    
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor='white')
    plt.close(fig)
