from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def plot(path, data: Dict, feature_names: Optional[list] = None, top_k: int = 15) -> None:
    """
    绘制特征重要性图（基于模型系数的伪 SHAP 风格图）.
    
    如果提供了 fixed_effects DataFrame，将绘制系数重要性条形图。
    否则绘制占位图。
    
    Args:
        path: 保存路径
        data: 数据字典，可包含:
            - fixed_effects: DataFrame with 'term' and 'estimate' columns
            - X: 特征矩阵 [N, D]
            - feature_importance: 预计算的特征重要性
        feature_names: 特征名列表
        top_k: 显示前 k 个重要特征
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 尝试从 fixed_effects 提取特征重要性
    if "fixed_effects" in data and data["fixed_effects"] is not None:
        df = data["fixed_effects"]
        if "term" in df.columns and "estimate" in df.columns:
            # 计算绝对值重要性
            df = df.copy()
            df["abs_importance"] = np.abs(df["estimate"])
            df = df.sort_values("abs_importance", ascending=True).tail(top_k)
            
            # 绘制水平条形图
            colors = ['#ff6b6b' if v < 0 else '#4ecdc4' for v in df["estimate"]]
            bars = ax.barh(range(len(df)), df["estimate"], color=colors, edgecolor='white', linewidth=0.5)
            
            # 设置 y 轴标签
            terms = df["term"].tolist()
            # 简化过长的特征名
            terms = [t[:30] + "..." if len(t) > 30 else t for t in terms]
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(terms, fontsize=9)
            
            ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
            ax.set_xlabel("Coefficient (Effect Size)", fontsize=11)
            ax.set_title(f"Top {len(df)} Feature Effects (M1 Judge Model)", fontsize=12, fontweight='bold')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4ecdc4', label='Positive Effect'),
                Patch(facecolor='#ff6b6b', label='Negative Effect')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
    elif "feature_importance" in data and data["feature_importance"] is not None:
        # 使用预计算的特征重要性
        importance = data["feature_importance"]
        names = feature_names if feature_names else [f"Feature {i}" for i in range(len(importance))]
        
        # 排序并取 top_k
        sorted_idx = np.argsort(np.abs(importance))[-top_k:]
        importance = importance[sorted_idx]
        names = [names[i] for i in sorted_idx]
        
        colors = ['#ff6b6b' if v < 0 else '#4ecdc4' for v in importance]
        ax.barh(range(len(importance)), importance, color=colors)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_title(f"Top {len(importance)} Feature Importance", fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    else:
        # 占位图
        ax.text(0.5, 0.5, "SHAP Analysis\n(No data available)", 
                ha="center", va="center", fontsize=14, color='gray')
        ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor='white')
    plt.close(fig)

