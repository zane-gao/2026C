from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class PathsConfig:
    data_csv: str = r"d:\杂七杂八\美赛\2026美赛\2026c\题目和资料\2026_MCM_Problem_C_Data.csv"
    output_dir: str = r"d:\杂七杂八\美赛\2026美赛\2026c\outputs"


@dataclass
class RuleConfig:
    # (start_season, end_season, rule_name)
    season_rules: Tuple[Tuple[int, int, str], ...] = (
        (1, 2, "rank"),
        (3, 27, "percent"),
        (28, 34, "bottom2_save"),
    )
    bottom2_base: str = "rank"  # "rank" or "percent"
    loglik_mode: str = "full"  # "full" or "fast"
    fast_model: bool = False  # If True, use simplified model (no random walk) for faster compilation


@dataclass
class HyperParams:
    epsilon: float = 0.005
    delta: float = 0.01
    tau: float = 0.2
    kappa: float = 20.0
    kappa_r: float = 10.0
    alpha: float = 0.01
    kappa_b: float = 20.0
    eta1: float = 1.0
    eta2: float = 0.5
    lambda_ent: float = 0.0


@dataclass
class DataConfig:
    max_week: int = 11
    withdrawal_policy: str = "exclude_only_withdrawn"  # or "invalidate_week"
    multi_elim_policy: str = "topk"  # or "invalidate_week"


@dataclass
class InferenceConfig:
    backend: str = "pymc"  # "pymc" or "numpyro" (placeholder)
    method: str = "vi"  # "vi" or "nuts"
    progressbar: bool = True
    progress_interval: int = 1000
    log_file: str | None = None
    log_interval_sec: int = 60
    pytensor_flags: str | None = None
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.9
    vi_iters: int = 20000
    seed: int = 42


@dataclass
class TorchConfig:
    """PyTorch training configuration."""
    epochs: int = 10000
    lr: float = 0.01
    lr_min: float = 1e-5
    warmup_epochs: int = 100
    patience: int = 500
    min_delta: float = 1.0
    grad_clip: float = 10.0
    n_samples: int = 1
    log_interval: int = 50
    save_interval: int = 500


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    hyper: HyperParams = field(default_factory=HyperParams)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
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
    # tolerate UTF-8 BOM in config files
    text = p.read_text(encoding="utf-8-sig")
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    cfg = Config()
    if "paths" in data:
        cfg.paths = PathsConfig(**data["paths"])
    if "rules" in data:
        cfg.rules = RuleConfig(**data["rules"])
    if "hyper" in data:
        cfg.hyper = HyperParams(**data["hyper"])
    if "data" in data:
        cfg.data = DataConfig(**data["data"])
    if "inference" in data:
        cfg.inference = InferenceConfig(**data["inference"])
    if "torch" in data:
        cfg.torch = TorchConfig(**data["torch"])
    return cfg
