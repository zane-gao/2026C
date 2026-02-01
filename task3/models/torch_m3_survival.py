"""
Task3 M3 模型的 PyTorch 实现.
生存分析模型 (Logistic 回归): eliminated ~ X*beta + J_z + y_k

支持批量处理多个后验样本 k，使用 CUDA 并行加速.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.tensors import TensorPack, get_device
from ..utils.progress import iter_with_progress


class BatchedLogisticRegression(nn.Module):
    """
    批量 Logistic 回归模型.
    
    支持同时拟合多个 k 的数据:
        logit(p) = X @ beta + eta_J * J_z + eta_F * y_k
    
    所有 k 共享 beta 和 eta_J，只有 eta_F 可选独立.
    """
    
    def __init__(
        self,
        n_features: int,
        n_k: int = 1,
        device: torch.device = None,
        prior_std: float = 10.0,
    ):
        super().__init__()
        
        self.device = device or get_device("auto")
        self.n_features = n_features
        self.n_k = n_k
        
        # 固定效应
        self.beta = nn.Parameter(torch.zeros(n_features))
        
        # J_z 系数 (shared)
        self.eta_J = nn.Parameter(torch.tensor(0.0))
        
        # y (fan share) 系数
        self.eta_F = nn.Parameter(torch.tensor(0.0))
        
        self.prior_std = prior_std
        self.to(self.device)
    
    def forward(
        self,
        X: torch.Tensor,
        J_z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 logit.
        
        Args:
            X: [N, D] 设计矩阵
            J_z: [N] 标准化评委分数
            y: [K, N] fan share 或差异
        
        Returns:
            logit: [K, N] logit 值
        """
        # 固定效应
        fixed = X @ self.beta  # [N]
        
        # J_z 效应
        j_effect = self.eta_J * J_z  # [N]
        
        # y 效应
        f_effect = self.eta_F * y  # [K, N]
        
        # 组合
        logit = fixed.unsqueeze(0) + j_effect.unsqueeze(0) + f_effect
        
        return logit
    
    def predict_proba(
        self,
        X: torch.Tensor,
        J_z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """计算概率."""
        logit = self.forward(X, J_z, y)
        return torch.sigmoid(logit)
    
    def loss(
        self,
        eliminated: torch.Tensor,
        X: torch.Tensor,
        J_z: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算二元交叉熵损失.
        
        Args:
            eliminated: [N] 淘汰标志 (0/1)
            X: [N, D]
            J_z: [N]
            y: [K, N]
            mask: [K, N] 有效掩码
        
        Returns:
            loss: 标量
        """
        logit = self.forward(X, J_z, y)  # [K, N]
        
        # 扩展 eliminated 到 [K, N]
        target = eliminated.unsqueeze(0).expand(logit.shape[0], -1).float()
        
        # 计算 BCE
        if mask is not None:
            # 只计算有效位置
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
            bce = (bce * mask.float()).sum() / (mask.sum() + 1e-8)
        else:
            bce = F.binary_cross_entropy_with_logits(logit, target)
        
        # L2 正则
        l2 = 0.5 * torch.sum(self.beta ** 2) / (self.prior_std ** 2)
        
        return bce + l2


def fit_m3_torch(
    tensor_pack: TensorPack,
    y_matrix: torch.Tensor,
    k_index: List[int],
    lr: float = 0.01,
    max_epochs: int = 500,
    patience: int = 20,
    verbose: bool = True,
    use_amp: bool = True,
) -> Tuple[BatchedLogisticRegression, Dict]:
    """
    使用 PyTorch 拟合 M3 模型.
    
    Args:
        tensor_pack: TensorPack 数据
        y_matrix: [K, N] Y 矩阵 (fan share)
        k_index: k 索引列表
        lr: 学习率
        max_epochs: 最大迭代次数
        patience: 早停 patience
        verbose: 是否打印进度
        use_amp: 是否使用混合精度
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    device = tensor_pack.device
    
    # 筛选有效数据
    mask = tensor_pack.is_active & tensor_pack.is_valid & torch.isfinite(tensor_pack.y)
    X = tensor_pack.X[mask]
    J_z = tensor_pack.y[mask]  # J_z 在 tensor_pack.y 中
    eliminated = tensor_pack.eliminated[mask].float()
    
    # Y 矩阵
    if isinstance(y_matrix, np.ndarray):
        y = torch.tensor(y_matrix, dtype=torch.float32, device=device)
    else:
        y = y_matrix.to(device)
    
    # 处理维度
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    # 筛选
    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
    y = y[:, mask_np]
    
    # 处理 NaN
    y_mask = torch.isfinite(y)
    y = torch.where(y_mask, y, torch.zeros_like(y))
    
    n_k = len(k_index)
    
    # 创建模型
    model = BatchedLogisticRegression(
        n_features=X.shape[1],
        n_k=n_k,
        device=device,
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
                loss = model.loss(eliminated, X, J_z, y, y_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.loss(eliminated, X, J_z, y, y_mask)
            loss.backward()
            optimizer.step()
        
        loss_val = float(loss.detach().cpu().numpy())
        history["loss"].append(loss_val)
        scheduler.step(loss_val)
        
        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"M3 early stopping at epoch {epoch + 1}")
            break
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"M3 Epoch {epoch + 1}/{max_epochs}, Loss: {loss_val:.4f}")
    
    return model, history


def compute_auc_brier(
    model: BatchedLogisticRegression,
    tensor_pack: TensorPack,
    y_matrix: torch.Tensor,
) -> Dict[str, float]:
    """
    计算 AUC 和 Brier score.
    
    Args:
        model: 训练好的模型
        tensor_pack: 数据
        y_matrix: [K, N] Y 矩阵
    
    Returns:
        metrics 字典
    """
    from sklearn.metrics import roc_auc_score
    
    device = tensor_pack.device
    model.eval()
    
    # 筛选有效数据
    mask = tensor_pack.is_active & tensor_pack.is_valid & torch.isfinite(tensor_pack.y)
    X = tensor_pack.X[mask]
    J_z = tensor_pack.y[mask]
    eliminated = tensor_pack.eliminated[mask]
    
    # Y 矩阵
    if isinstance(y_matrix, np.ndarray):
        y = torch.tensor(y_matrix, dtype=torch.float32, device=device)
    else:
        y = y_matrix.to(device)
    
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
    y = y[:, mask_np]
    y_mask = torch.isfinite(y)
    y = torch.where(y_mask, y, torch.zeros_like(y))
    
    with torch.no_grad():
        prob = model.predict_proba(X, J_z, y)  # [K, N]
    
    prob_np = prob.cpu().numpy()
    elim_np = eliminated.cpu().numpy()
    
    # 每个 k 的 AUC 和 Brier
    auc_list = []
    brier_list = []
    
    for k in range(prob_np.shape[0]):
        p_k = prob_np[k]
        valid = y_mask[k].cpu().numpy()
        if valid.sum() > 0 and elim_np[valid].sum() > 0 and elim_np[valid].sum() < valid.sum():
            try:
                auc = roc_auc_score(elim_np[valid], p_k[valid])
                auc_list.append(auc)
            except:
                pass
            brier = np.mean((elim_np[valid] - p_k[valid]) ** 2)
            brier_list.append(brier)
    
    return {
        "auc": float(np.median(auc_list)) if auc_list else float("nan"),
        "auc_q05": float(np.quantile(auc_list, 0.05)) if auc_list else float("nan"),
        "auc_q95": float(np.quantile(auc_list, 0.95)) if auc_list else float("nan"),
        "brier": float(np.median(brier_list)) if brier_list else float("nan"),
        "brier_q05": float(np.quantile(brier_list, 0.05)) if brier_list else float("nan"),
        "brier_q95": float(np.quantile(brier_list, 0.95)) if brier_list else float("nan"),
    }


def extract_m3_results(
    model: BatchedLogisticRegression,
    tensor_pack: TensorPack,
    y_matrix: torch.Tensor,
    feature_names: List[str],
) -> Dict:
    """
    从训练好的模型提取 M3 结果.
    
    Args:
        model: 训练好的模型
        tensor_pack: 数据
        y_matrix: [K, N] Y 矩阵
        feature_names: 特征名列表
    
    Returns:
        结果字典
    """
    import pandas as pd
    
    model.eval()
    
    # 固定效应
    beta = model.beta.detach().cpu().numpy()
    fixed_rows = []
    for i, name in enumerate(feature_names):
        fixed_rows.append({
            "term": name,
            "estimate": float(beta[i]),
        })
    fixed_rows.append({"term": "J_z", "estimate": float(model.eta_J.detach().cpu().numpy())})
    fixed_rows.append({"term": "y (fan)", "estimate": float(model.eta_F.detach().cpu().numpy())})
    fixed_df = pd.DataFrame(fixed_rows)
    
    # Metrics
    metrics = compute_auc_brier(model, tensor_pack, y_matrix)
    metrics["eta_J"] = float(model.eta_J.detach().cpu().numpy())
    metrics["eta_F"] = float(model.eta_F.detach().cpu().numpy())
    
    return {
        "fixed": fixed_df,
        "metrics": metrics,
    }
