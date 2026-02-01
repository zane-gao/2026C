from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class PathsConfig:
    data_csv: str = "code/raw_data/2026_MCM_Problem_C_Data.csv"
    social_data: str = "code/raw_data/2026美赛C题补充数据集！.xlsx"
    enriched_path: Optional[str] = None
    task1_artifact: str = "code/task1/outputs/run_torch"
    output_root: str = "outputs/task3"


@dataclass
class PanelConfig:
    max_week: int = 11
    use_valid_mask: bool = True
    exclude_post_elim_zero: bool = True
    ref_strategy: str = "max_mean_F"


@dataclass
class FeatureConfig:
    include_social: bool = True
    include_platform: bool = False
    missing_as_zero: bool = False
    winsorize_q: Optional[float] = None
    week_effect: str = "linear"  # linear or categorical


@dataclass
class RuntimeConfig:
    seed: int = 42
    num_workers: int = 0
    k_subsample: int = 50
    m3_k_repeats: int = 0


@dataclass
class TorchConfig:
    """PyTorch/CUDA 加速配置."""
    enabled: bool = True
    device: str = "cuda"  # cuda, cpu, auto
    batch_size: int = 256
    lr: float = 0.005  # 更保守的学习率
    max_epochs: int = 500
    patience: int = 50  # 早停 patience (增大避免M2早停)
    n_samples: int = 100  # 后验采样数
    mixed_precision: bool = True  # 使用 AMP 加速


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    torch: TorchConfig = field(default_factory=TorchConfig)

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if p.suffix.lower() in (".yml", ".yaml"):
            if yaml is None:
                raise RuntimeError("pyyaml not installed")
            p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config(path: Optional[str] = None) -> Config:
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8-sig")
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    cfg = Config()
    if isinstance(data, dict):
        if "paths" in data:
            cfg.paths = PathsConfig(**data["paths"])
        if "panel" in data:
            cfg.panel = PanelConfig(**data["panel"])
        if "features" in data:
            cfg.features = FeatureConfig(**data["features"])
        if "runtime" in data:
            cfg.runtime = RuntimeConfig(**data["runtime"])
        if "torch" in data:
            cfg.torch = TorchConfig(**data["torch"])
    return cfg
