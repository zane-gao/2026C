"""
Task3 M1 模型的 PyTorch 实现.
线性混合效应模型: J_z ~ X*beta + u_pro + v_celeb + w_season + eps

使用变分推断实现，支持 CUDA 加速.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.tensors import TensorPack, get_device


class MixedLMTorch(nn.Module):
    """
    PyTorch 实现的线性混合效应模型.
    
    模型:
        y = X @ beta + u[pro_id] + v[celeb_id] + w[season] + eps
        
        u ~ N(0, sigma_pro^2)
        v ~ N(0, sigma_celeb^2)
        w ~ N(0, sigma_season^2)
        eps ~ N(0, sigma_resid^2)
    
    使用 MAP 估计或变分推断.
    """
    
    def __init__(
        self,
        n_features: int,
        n_pro: int,
        n_celeb: int,
        n_season: int,
        device: torch.device = None,
        prior_std: float = 10.0,
    ):
        super().__init__()
        
        self.device = device or get_device("auto")
        self.n_features = n_features
        self.n_pro = n_pro
        self.n_celeb = n_celeb
        self.n_season = n_season
        
        # 固定效应参数
        self.beta = nn.Parameter(torch.zeros(n_features))
        
        # 随机效应
        self.u_pro = nn.Parameter(torch.zeros(n_pro))  # 专业舞者效应
        self.v_celeb = nn.Parameter(torch.zeros(n_celeb))  # 名人效应
        self.w_season = nn.Parameter(torch.zeros(n_season))  # 赛季效应
        
        # 方差分量 (log scale for positivity)
        self.log_sigma_pro = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_celeb = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_season = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_resid = nn.Parameter(torch.tensor(0.0))
        
        # 先验
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
    ) -> torch.Tensor:
        """
        前向传播，计算预测值.
        
        Args:
            X: [N, D] 设计矩阵
            pro_id: [N] 专业舞者 ID
            celeb_id: [N] 名人 ID
            season: [N] 赛季
        
        Returns:
            y_pred: [N] 预测值
        """
        # 固定效应
        y_pred = X @ self.beta
        
        # 随机效应
        y_pred = y_pred + self.u_pro[pro_id]
        y_pred = y_pred + self.v_celeb[celeb_id]
        y_pred = y_pred + self.w_season[season]
        
        return y_pred
    
    def log_likelihood(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        pro_id: torch.Tensor,
        celeb_id: torch.Tensor,
        season: torch.Tensor,
    ) -> torch.Tensor:
        """计算对数似然."""
        y_pred = self.forward(X, pro_id, celeb_id, season)
        resid = y - y_pred
        
        # 高斯似然
        ll = -0.5 * torch.sum(resid ** 2) / (self.sigma_resid ** 2)
        ll = ll - y.shape[0] * torch.log(self.sigma_resid)
        ll = ll - 0.5 * y.shape[0] * math.log(2 * math.pi)
        
        return ll
    
    def log_prior(self) -> torch.Tensor:
        """计算对数先验."""
        lp = torch.tensor(0.0, device=self.device)
        
        # 固定效应先验 N(0, prior_std^2)
        lp = lp - 0.5 * torch.sum(self.beta ** 2) / (self.prior_std ** 2)
        
        # 随机效应先验
        lp = lp - 0.5 * torch.sum(self.u_pro ** 2) / (self.sigma_pro ** 2 + 1e-8)
        lp = lp - self.n_pro * torch.log(self.sigma_pro + 1e-8)
        
        lp = lp - 0.5 * torch.sum(self.v_celeb ** 2) / (self.sigma_celeb ** 2 + 1e-8)
        lp = lp - self.n_celeb * torch.log(self.sigma_celeb + 1e-8)
        
        lp = lp - 0.5 * torch.sum(self.w_season ** 2) / (self.sigma_season ** 2 + 1e-8)
        lp = lp - self.n_season * torch.log(self.sigma_season + 1e-8)
        
        return lp
    
    def loss(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        pro_id: torch.Tensor,
        celeb_id: torch.Tensor,
        season: torch.Tensor,
    ) -> torch.Tensor:
        """计算负对数后验 (loss)."""
        return -(self.log_likelihood(y, X, pro_id, celeb_id, season) + self.log_prior())
    
    def get_random_effects(self) -> Dict[str, np.ndarray]:
        """获取随机效应估计."""
        return {
            "u_pro": self.u_pro.detach().cpu().numpy(),
            "v_celeb": self.v_celeb.detach().cpu().numpy(),
            "w_season": self.w_season.detach().cpu().numpy(),
        }
    
    def get_var_components(self) -> Dict[str, float]:
        """获取方差分量估计."""
        return {
            "pro": float(self.sigma_pro.detach().cpu().numpy() ** 2),
            "celeb": float(self.sigma_celeb.detach().cpu().numpy() ** 2),
            "season": float(self.sigma_season.detach().cpu().numpy() ** 2),
            "resid": float(self.sigma_resid.detach().cpu().numpy() ** 2),
        }


def fit_m1_torch(
    tensor_pack: TensorPack,
    lr: float = 0.01,
    max_epochs: int = 500,
    patience: int = 20,
    verbose: bool = True,
    use_amp: bool = True,
) -> Tuple[MixedLMTorch, Dict]:
    """
    使用 PyTorch 拟合 M1 模型.
    
    Args:
        tensor_pack: TensorPack 数据
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
    y = tensor_pack.y[mask]
    pro_id = tensor_pack.pro_id[mask]
    celeb_id = tensor_pack.celeb_id[mask]
    season = tensor_pack.season[mask]
    
    # 创建模型
    model = MixedLMTorch(
        n_features=X.shape[1],
        n_pro=tensor_pack.n_pro,
        n_celeb=tensor_pack.n_celeb,
        n_season=tensor_pack.n_season,
        device=device,
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == "cuda" else None
    
    # 训练
    best_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "lr": []}
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = model.loss(y, X, pro_id, celeb_id, season)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.loss(y, X, pro_id, celeb_id, season)
            loss.backward()
            optimizer.step()
        
        loss_val = float(loss.detach().cpu().numpy())
        history["loss"].append(loss_val)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        scheduler.step(loss_val)
        
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss_val:.4f}")
    
    return model, history


def extract_m1_results(
    model: MixedLMTorch,
    tensor_pack: TensorPack,
    panel,
    feature_names: List[str],
) -> Dict:
    """
    从训练好的模型提取 M1 结果.
    
    Args:
        model: 训练好的 MixedLMTorch 模型
        tensor_pack: TensorPack 数据
        panel: 原始 panel DataFrame
        feature_names: 特征名列表
    
    Returns:
        结果字典，包含 fixed, random_pro, random_celeb, random_season, metrics, var_components
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
            "std_err": float("nan"),  # 可通过 Hessian 估计
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        })
    fixed_df = pd.DataFrame(fixed_rows)
    
    # 随机效应
    re = model.get_random_effects()
    
    # Pro 随机效应
    pro_name_map = panel.groupby("pro_id")["pro_name"].first().to_dict()
    pro_id_list = sorted(panel["pro_id"].unique())
    pro_rows = []
    for i, pid in enumerate(pro_id_list):
        if i < len(re["u_pro"]):
            pro_rows.append({
                "pro_id": int(pid),
                "pro_name": pro_name_map.get(pid, ""),
                "u_pro": float(re["u_pro"][i]),
            })
    pro_df = pd.DataFrame(pro_rows)
    
    # Celeb 随机效应
    celeb_name_map = panel.groupby("celeb_id")["celeb_name"].first().to_dict()
    celeb_id_list = sorted(panel["celeb_id"].unique())
    celeb_rows = []
    for i, cid in enumerate(celeb_id_list):
        if i < len(re["v_celeb"]):
            celeb_rows.append({
                "celeb_id": int(cid),
                "celeb_name": celeb_name_map.get(cid, ""),
                "v_celeb": float(re["v_celeb"][i]),
            })
    celeb_df = pd.DataFrame(celeb_rows)
    
    # Season 随机效应
    season_list = sorted(panel["season"].unique())
    season_rows = []
    for i, s in enumerate(season_list):
        if i < len(re["w_season"]):
            season_rows.append({
                "season": int(s),
                "u_season": float(re["w_season"][i]),
            })
    season_df = pd.DataFrame(season_rows)
    
    # 方差分量
    var_components = model.get_var_components()
    
    # 计算 metrics
    mask = tensor_pack.is_active & tensor_pack.is_valid & torch.isfinite(tensor_pack.y)
    with torch.no_grad():
        y_pred = model.forward(
            tensor_pack.X[mask],
            tensor_pack.pro_id[mask],
            tensor_pack.celeb_id[mask],
            tensor_pack.season[mask],
        )
    
    y_true = tensor_pack.y[mask].cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    ss_res = np.sum((y_true - y_pred_np) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    rmse = np.sqrt(np.mean((y_true - y_pred_np) ** 2))
    
    metrics = {
        "r2": float(r2),
        "rmse": float(rmse),
        "n_obs": int(mask.sum().cpu().numpy()),
    }
    
    return {
        "fixed": fixed_df,
        "random_pro": pro_df,
        "random_celeb": celeb_df,
        "random_season": season_df,
        "var_components": var_components,
        "metrics": metrics,
    }
