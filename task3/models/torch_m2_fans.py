"""
Task3 M2 模型的 PyTorch 实现.
Fan share 差异模型: y_k ~ X*beta + u_pro + v_celeb + w_season + eps

支持批量处理多个后验样本 k，使用 CUDA 并行加速.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.tensors import TensorPack, get_device
from ..utils.progress import iter_with_progress


class BatchedMixedLM(nn.Module):
    """
    批量处理多个 k 的线性混合效应模型.
    
    通过批量处理所有 k 的数据实现并行化:
    - 共享固定效应 beta
    - 每个 k 有独立的随机效应 (可选)
    
    模型:
        y_k = X @ beta + u_pro[k] + v_celeb[k] + w_season + eps
    """
    
    def __init__(
        self,
        n_features: int,
        n_pro: int,
        n_celeb: int,
        n_season: int,
        n_k: int,
        device: torch.device = None,
        shared_beta: bool = True,
        prior_std: float = 10.0,
    ):
        super().__init__()
        
        self.device = device or get_device("auto")
        self.n_features = n_features
        self.n_pro = n_pro
        self.n_celeb = n_celeb
        self.n_season = n_season
        self.n_k = n_k
        self.shared_beta = shared_beta
        
        # 固定效应
        if shared_beta:
            self.beta = nn.Parameter(torch.zeros(n_features))
        else:
            self.beta = nn.Parameter(torch.zeros(n_k, n_features))
        
        # 随机效应 - 每个 k 独立
        self.u_pro = nn.Parameter(torch.zeros(n_k, n_pro))
        self.v_celeb = nn.Parameter(torch.zeros(n_k, n_celeb))
        self.w_season = nn.Parameter(torch.zeros(n_season))  # 赛季共享
        
        # 方差分量
        self.log_sigma_pro = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_celeb = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_season = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_resid = nn.Parameter(torch.tensor(0.0))
        
        self.prior_std = prior_std
        self.to(self.device)
    
    @property
    def sigma_pro(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_pro)
    
    @property
    def sigma_celeb(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_celeb)
    
    @property
    def sigma_season(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_season)
    
    @property
    def sigma_resid(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_resid)
    
    def forward(
        self,
        X: torch.Tensor,
        pro_id: torch.Tensor,
        celeb_id: torch.Tensor,
        season: torch.Tensor,
        k_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播.
        
        Args:
            X: [N, D] 或 [K, N, D] 设计矩阵
            pro_id: [N] 专业舞者 ID
            celeb_id: [N] 名人 ID
            season: [N] 赛季
            k_idx: [K] 或 None, k 索引
        
        Returns:
            y_pred: [K, N] 预测值
        """
        if X.dim() == 2:
            # X: [N, D]
            if self.shared_beta:
                # beta: [D]
                fixed = X @ self.beta  # [N]
                fixed = fixed.unsqueeze(0).expand(self.n_k, -1)  # [K, N]
            else:
                # beta: [K, D]
                fixed = torch.einsum("nd,kd->kn", X, self.beta)  # [K, N]
        else:
            # X: [K, N, D] - 不同 k 有不同 X
            if self.shared_beta:
                fixed = torch.einsum("knd,d->kn", X, self.beta)
            else:
                fixed = torch.einsum("knd,kd->kn", X, self.beta)
        
        # 随机效应
        # u_pro: [K, n_pro], pro_id: [N] -> [K, N]
        u_effect = self.u_pro[:, pro_id]  # [K, N]
        v_effect = self.v_celeb[:, celeb_id]  # [K, N]
        w_effect = self.w_season[season]  # [N]
        
        y_pred = fixed + u_effect + v_effect + w_effect.unsqueeze(0)
        
        return y_pred
    
    def loss(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        pro_id: torch.Tensor,
        celeb_id: torch.Tensor,
        season: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算批量损失.
        
        Args:
            y: [K, N] 目标值
            X: [N, D] 设计矩阵
            pro_id: [N]
            celeb_id: [N]
            season: [N]
            mask: [K, N] 有效掩码
        
        Returns:
            loss: 标量
        """
        y_pred = self.forward(X, pro_id, celeb_id, season)  # [K, N]
        resid = y - y_pred
        
        if mask is not None:
            resid = resid * mask.float()
            n_valid = mask.sum()
        else:
            n_valid = y.numel()
        
        # 似然
        ll = -0.5 * torch.sum(resid ** 2) / (self.sigma_resid ** 2 + 1e-8)
        ll = ll - n_valid * torch.log(self.sigma_resid + 1e-8)
        
        # 先验
        lp = -0.5 * torch.sum(self.beta ** 2) / (self.prior_std ** 2)
        lp = lp - 0.5 * torch.sum(self.u_pro ** 2) / (self.sigma_pro ** 2 + 1e-8)
        lp = lp - 0.5 * torch.sum(self.v_celeb ** 2) / (self.sigma_celeb ** 2 + 1e-8)
        lp = lp - 0.5 * torch.sum(self.w_season ** 2) / (self.sigma_season ** 2 + 1e-8)
        
        return -(ll + lp)
    
    def get_fixed_effects(self) -> np.ndarray:
        """获取固定效应."""
        return self.beta.detach().cpu().numpy()
    
    def get_random_effects(self) -> Dict[str, np.ndarray]:
        """获取随机效应."""
        return {
            "u_pro": self.u_pro.detach().cpu().numpy(),  # [K, n_pro]
            "v_celeb": self.v_celeb.detach().cpu().numpy(),  # [K, n_celeb]
            "w_season": self.w_season.detach().cpu().numpy(),  # [n_season]
        }
    
    def get_var_components(self) -> Dict[str, float]:
        """获取方差分量估计."""
        return {
            "pro": float(self.sigma_pro.detach().cpu().numpy() ** 2),
            "celeb": float(self.sigma_celeb.detach().cpu().numpy() ** 2),
            "season": float(self.sigma_season.detach().cpu().numpy() ** 2),
            "resid": float(self.sigma_resid.detach().cpu().numpy() ** 2),
        }


def fit_m2_torch(
    tensor_pack: TensorPack,
    y_matrix: torch.Tensor,
    k_index: List[int],
    lr: float = 0.01,
    max_epochs: int = 500,
    patience: int = 20,
    verbose: bool = True,
    use_amp: bool = True,
    batch_k: int = 50,
) -> Tuple[BatchedMixedLM, Dict]:
    """
    使用 PyTorch 拟合 M2 模型 (批量处理所有 k).
    
    Args:
        tensor_pack: TensorPack 数据
        y_matrix: [K, N] Y 矩阵 (fan share 差异)
        k_index: k 索引列表
        lr: 学习率
        max_epochs: 最大迭代次数
        patience: 早停 patience
        verbose: 是否打印进度
        use_amp: 是否使用混合精度
        batch_k: 每批处理的 k 数量
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    device = tensor_pack.device
    
    # 筛选有效数据
    mask = tensor_pack.is_active & tensor_pack.is_valid & torch.isfinite(tensor_pack.y)
    X = tensor_pack.X[mask]
    pro_id = tensor_pack.pro_id[mask]
    celeb_id = tensor_pack.celeb_id[mask]
    season = tensor_pack.season[mask]
    
    # Y 矩阵
    y = y_matrix[:, mask.cpu().numpy() if hasattr(mask, 'cpu') else mask]
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32, device=device)
    else:
        y = y.to(device)
    
    # 处理 NaN
    y_mask = torch.isfinite(y)
    y = torch.where(y_mask, y, torch.zeros_like(y))
    
    n_k = len(k_index)
    
    # 创建模型
    model = BatchedMixedLM(
        n_features=X.shape[1],
        n_pro=tensor_pack.n_pro,
        n_celeb=tensor_pack.n_celeb,
        n_season=tensor_pack.n_season,
        n_k=n_k,
        device=device,
        shared_beta=True,
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    
    # AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == "cuda" else None
    
    # 训练
    best_loss = float("inf")
    patience_counter = 0
    history = {"loss": []}
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = model.loss(y, X, pro_id, celeb_id, season, y_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.loss(y, X, pro_id, celeb_id, season, y_mask)
            loss.backward()
            optimizer.step()
        
        loss_val = float(loss.detach().cpu().numpy())
        history["loss"].append(loss_val)
        scheduler.step(loss_val)
        
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"M2 early stopping at epoch {epoch + 1}")
            break
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"M2 Epoch {epoch + 1}/{max_epochs}, Loss: {loss_val:.4f}")
    
    return model, history


def extract_m2_results(
    model: BatchedMixedLM,
    tensor_pack: TensorPack,
    y_matrix: torch.Tensor,
    k_index: List[int],
    panel,
    feature_names: List[str],
) -> Dict:
    """
    从训练好的模型提取 M2 结果.
    
    Args:
        model: 训练好的 BatchedMixedLM 模型
        tensor_pack: TensorPack 数据
        y_matrix: [K, N] Y 矩阵
        k_index: k 索引列表
        panel: 原始 panel DataFrame
        feature_names: 特征名列表
    
    Returns:
        结果字典
    """
    import pandas as pd
    
    model.eval()
    
    # 固定效应
    beta = model.get_fixed_effects()
    if beta.ndim == 1:
        # 共享 beta
        fixed_rows = []
        for i, name in enumerate(feature_names):
            fixed_rows.append({
                "term": name,
                "median": float(beta[i]),
                "q05": float(beta[i]),
                "q95": float(beta[i]),
            })
    else:
        # 每个 k 独立 beta
        fixed_rows = []
        for i, name in enumerate(feature_names):
            vals = beta[:, i]
            fixed_rows.append({
                "term": name,
                "median": float(np.median(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "q95": float(np.quantile(vals, 0.95)),
            })
    fixed_df = pd.DataFrame(fixed_rows)
    
    # 随机效应
    re = model.get_random_effects()
    
    # Pro 随机效应汇总
    u_pro = re["u_pro"]  # [K, n_pro]
    pro_name_map = panel.groupby("pro_id")["pro_name"].first().to_dict()
    pro_id_list = sorted(panel["pro_id"].unique())
    pro_rows = []
    for i, pid in enumerate(pro_id_list):
        if i < u_pro.shape[1]:
            vals = u_pro[:, i]
            pro_rows.append({
                "pro_id": int(pid),
                "pro_name": pro_name_map.get(pid, ""),
                "median": float(np.median(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "q95": float(np.quantile(vals, 0.95)),
            })
    pro_df = pd.DataFrame(pro_rows)
    
    # Celeb 随机效应汇总
    v_celeb = re["v_celeb"]  # [K, n_celeb]
    celeb_name_map = panel.groupby("celeb_id")["celeb_name"].first().to_dict()
    celeb_id_list = sorted(panel["celeb_id"].unique())
    celeb_rows = []
    for i, cid in enumerate(celeb_id_list):
        if i < v_celeb.shape[1]:
            vals = v_celeb[:, i]
            celeb_rows.append({
                "celeb_id": int(cid),
                "celeb_name": celeb_name_map.get(cid, ""),
                "median": float(np.median(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "q95": float(np.quantile(vals, 0.95)),
            })
    celeb_df = pd.DataFrame(celeb_rows)
    
    # 计算 metrics
    device = tensor_pack.device
    mask = tensor_pack.is_active & tensor_pack.is_valid & torch.isfinite(tensor_pack.y)
    X = tensor_pack.X[mask]
    pro_id = tensor_pack.pro_id[mask]
    celeb_id = tensor_pack.celeb_id[mask]
    season = tensor_pack.season[mask]
    
    y = y_matrix[:, mask.cpu().numpy() if hasattr(mask, 'cpu') else mask]
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32, device=device)
    else:
        y = y.to(device)
    
    with torch.no_grad():
        y_pred = model.forward(X, pro_id, celeb_id, season)
    
    y_true = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 每个 k 的 R2
    r2_list = []
    for k in range(y_true.shape[0]):
        y_k = y_true[k]
        y_pred_k = y_pred_np[k]
        valid = np.isfinite(y_k)
        if valid.sum() > 0:
            ss_res = np.sum((y_k[valid] - y_pred_k[valid]) ** 2)
            ss_tot = np.sum((y_k[valid] - y_k[valid].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2_list.append(r2)
    
    metrics = {
        "r2_median": float(np.median(r2_list)) if r2_list else float("nan"),
        "r2_q05": float(np.quantile(r2_list, 0.05)) if r2_list else float("nan"),
        "r2_q95": float(np.quantile(r2_list, 0.95)) if r2_list else float("nan"),
        "n_k": len(k_index),
    }
    
    # 方差分量
    var_components = model.get_var_components()
    
    return {
        "fixed_summary": fixed_df,
        "random_pro_summary": pro_df,
        "random_celeb_summary": celeb_df,
        "metrics": metrics,
        "var_components": var_components,
    }
