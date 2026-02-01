"""
Task3 PyTorch 数据张量模块.
将 pandas DataFrame panel 转换为 PyTorch 张量格式，支持 CUDA 加速.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..types import FeatureSpec


@dataclass
class TensorPack:
    """存储所有用于 PyTorch 模型的张量."""
    # 设计矩阵 [N_obs, D]
    X: torch.Tensor
    # 响应变量
    y: torch.Tensor  # [N_obs] - J_z 或 fan share
    # 分组索引
    pro_id: torch.Tensor  # [N_obs] - 专业舞者 ID
    celeb_id: torch.Tensor  # [N_obs] - 名人 ID
    season: torch.Tensor  # [N_obs] - 赛季
    week: torch.Tensor  # [N_obs] - 周数
    # 掩码
    is_active: torch.Tensor  # [N_obs]
    is_valid: torch.Tensor  # [N_obs]
    eliminated: torch.Tensor  # [N_obs] - 淘汰标志
    # 索引映射
    obs_idx: torch.Tensor  # [N_obs] - 原始行索引
    # 元数据
    n_pro: int
    n_celeb: int
    n_season: int
    n_week: int
    feature_names: List[str]
    device: torch.device

    def to(self, device: torch.device) -> "TensorPack":
        """移动所有张量到指定设备."""
        return TensorPack(
            X=self.X.to(device),
            y=self.y.to(device),
            pro_id=self.pro_id.to(device),
            celeb_id=self.celeb_id.to(device),
            season=self.season.to(device),
            week=self.week.to(device),
            is_active=self.is_active.to(device),
            is_valid=self.is_valid.to(device),
            eliminated=self.eliminated.to(device),
            obs_idx=self.obs_idx.to(device),
            n_pro=self.n_pro,
            n_celeb=self.n_celeb,
            n_season=self.n_season,
            n_week=self.n_week,
            feature_names=self.feature_names,
            device=device,
        )


def get_device(device_str: str = "auto") -> torch.device:
    """获取计算设备."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _encode_categorical(series, unknown_token: str = "Unknown") -> Tuple[torch.Tensor, Dict[str, int]]:
    """将分类变量编码为整数索引."""
    unique_vals = series.fillna(unknown_token).unique().tolist()
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    encoded = series.fillna(unknown_token).map(val_to_idx).values
    return torch.tensor(encoded, dtype=torch.long), val_to_idx


def _one_hot_encode(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot 编码."""
    return F.one_hot(indices, num_classes=num_classes).float()


def build_design_matrix(
    panel,
    feature_spec: FeatureSpec,
    week_effect: str = "linear",
    device: torch.device = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    从 panel DataFrame 构建设计矩阵.
    
    Args:
        panel: pandas DataFrame
        feature_spec: 特征规格
        week_effect: 'linear' 或 'categorical'
        device: 计算设备
    
    Returns:
        X: [N, D] 设计矩阵
        feature_names: 特征名列表
    """
    import pandas as pd
    
    if device is None:
        device = torch.device("cpu")
    
    features = []
    feature_names = []
    
    # 数值特征
    for col in feature_spec.base_numeric:
        if col in panel.columns:
            vals = pd.to_numeric(panel[col], errors="coerce").fillna(0.0).values
            features.append(torch.tensor(vals, dtype=torch.float32).unsqueeze(1))
            feature_names.append(col)
    
    # 社交特征
    for col in feature_spec.social_cols:
        if col in panel.columns:
            vals = pd.to_numeric(panel[col], errors="coerce").fillna(0.0).values
            features.append(torch.tensor(vals, dtype=torch.float32).unsqueeze(1))
            feature_names.append(col)
    
    # 平台特征
    for col in feature_spec.platform_cols:
        if col in panel.columns:
            vals = pd.to_numeric(panel[col], errors="coerce").fillna(0.0).values
            features.append(torch.tensor(vals, dtype=torch.float32).unsqueeze(1))
            feature_names.append(col)
    
    # 缺失标志
    for col in feature_spec.missing_cols:
        if col in panel.columns:
            vals = panel[col].astype(float).values
            features.append(torch.tensor(vals, dtype=torch.float32).unsqueeze(1))
            feature_names.append(col)
    
    # 分类特征 (one-hot)
    for col in feature_spec.base_categorical:
        if col in panel.columns:
            encoded, mapping = _encode_categorical(panel[col])
            one_hot = _one_hot_encode(encoded, len(mapping))
            features.append(one_hot)
            for val in mapping.keys():
                feature_names.append(f"{col}_{val}")
    
    # 周效应
    if week_effect == "linear":
        week_vals = panel["week"].values.astype(float)
        features.append(torch.tensor(week_vals, dtype=torch.float32).unsqueeze(1))
        feature_names.append("week")
    elif week_effect == "categorical":
        week_encoded, week_mapping = _encode_categorical(panel["week"].astype(str))
        week_one_hot = _one_hot_encode(week_encoded, len(week_mapping))
        features.append(week_one_hot)
        for val in week_mapping.keys():
            feature_names.append(f"week_{val}")
    
    # 合并所有特征
    if features:
        X = torch.cat(features, dim=1).to(device)
    else:
        X = torch.zeros((len(panel), 1), dtype=torch.float32, device=device)
        feature_names = ["intercept"]
    
    return X, feature_names


def panel_to_tensors(
    panel,
    feature_spec: FeatureSpec,
    response_col: str = "J_z",
    week_effect: str = "linear",
    device: str = "auto",
) -> TensorPack:
    """
    将 panel DataFrame 转换为 TensorPack.
    
    Args:
        panel: pandas DataFrame
        feature_spec: 特征规格
        response_col: 响应变量列名
        week_effect: 'linear' 或 'categorical'
        device: 计算设备 ('cuda', 'cpu', 'auto')
    
    Returns:
        TensorPack 对象
    """
    import pandas as pd
    
    device = get_device(device)
    
    # 构建设计矩阵
    X, feature_names = build_design_matrix(panel, feature_spec, week_effect, device)
    
    # 响应变量
    y = pd.to_numeric(panel[response_col], errors="coerce").fillna(0.0).values
    y = torch.tensor(y, dtype=torch.float32, device=device)
    
    # 编码分组变量
    pro_encoded, pro_map = _encode_categorical(panel["pro_id"].astype(str))
    celeb_encoded, celeb_map = _encode_categorical(panel["celeb_id"].astype(str))
    season_encoded, season_map = _encode_categorical(panel["season"].astype(str))
    
    # 周数
    week_vals = panel["week"].values.astype(np.int64)
    week = torch.tensor(week_vals, dtype=torch.long, device=device)
    
    # 掩码
    is_active = panel["is_active"].values if "is_active" in panel.columns else np.ones(len(panel), dtype=bool)
    is_valid = panel["is_valid_week"].values if "is_valid_week" in panel.columns else np.ones(len(panel), dtype=bool)
    eliminated = panel["eliminated_this_week"].values if "eliminated_this_week" in panel.columns else np.zeros(len(panel), dtype=bool)
    
    return TensorPack(
        X=X,
        y=y,
        pro_id=pro_encoded.to(device),
        celeb_id=celeb_encoded.to(device),
        season=season_encoded.to(device),
        week=week,
        is_active=torch.tensor(is_active, dtype=torch.bool, device=device),
        is_valid=torch.tensor(is_valid, dtype=torch.bool, device=device),
        eliminated=torch.tensor(eliminated, dtype=torch.bool, device=device),
        obs_idx=torch.arange(len(panel), device=device),
        n_pro=len(pro_map),
        n_celeb=len(celeb_map),
        n_season=len(season_map),
        n_week=int(panel["week"].max()) + 1,
        feature_names=feature_names,
        device=device,
    )


def build_y_matrix_torch(
    theta: torch.Tensor,
    ref_idx: torch.Tensor,
    s_idx: torch.Tensor,
    t_idx: torch.Tensor,
    i_idx: torch.Tensor,
    k_index: List[int],
) -> torch.Tensor:
    """
    构建 M2 的 Y 矩阵 (fan share 差异).
    
    Args:
        theta: [K, S, T, N] 潜在能力张量
        ref_idx: [S, T] 参考选手索引
        s_idx: [N_obs] 赛季索引
        t_idx: [N_obs] 周索引
        i_idx: [N_obs] 选手索引
        k_index: 后验样本索引列表
    
    Returns:
        y_matrix: [len(k_index), N_obs] Y 矩阵
    """
    device = theta.device
    n_obs = s_idx.shape[0]
    n_k = len(k_index)
    
    y_matrix = torch.zeros((n_k, n_obs), dtype=torch.float32, device=device)
    
    for kk, k in enumerate(k_index):
        # 获取参考选手的 theta
        ref = ref_idx[s_idx, t_idx]
        # 计算差异
        y_matrix[kk] = theta[k, s_idx, t_idx, i_idx] - theta[k, s_idx, t_idx, ref]
    
    return y_matrix


class Task3Dataset(torch.utils.data.Dataset):
    """Task3 PyTorch Dataset 类."""
    
    def __init__(self, tensor_pack: TensorPack, mask: Optional[torch.Tensor] = None):
        """
        Args:
            tensor_pack: TensorPack 对象
            mask: 可选的布尔掩码，用于选择子集
        """
        self.data = tensor_pack
        if mask is not None:
            self.indices = torch.where(mask)[0]
        else:
            self.indices = torch.arange(tensor_pack.X.shape[0])
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {
            "X": self.data.X[i],
            "y": self.data.y[i],
            "pro_id": self.data.pro_id[i],
            "celeb_id": self.data.celeb_id[i],
            "season": self.data.season[i],
            "week": self.data.week[i],
            "eliminated": self.data.eliminated[i],
        }


def create_dataloader(
    tensor_pack: TensorPack,
    batch_size: int = 256,
    shuffle: bool = True,
    mask: Optional[torch.Tensor] = None,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    创建 DataLoader.
    
    Args:
        tensor_pack: TensorPack 对象
        batch_size: 批大小
        shuffle: 是否打乱
        mask: 可选的布尔掩码
        num_workers: 工作进程数
    
    Returns:
        DataLoader
    """
    dataset = Task3Dataset(tensor_pack, mask)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=tensor_pack.device.type == "cuda",
    )
