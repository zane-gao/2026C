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
    data_csv: str = "2026_MCM_Problem_C_Data.csv"
    task1_artifact: str = "code/task1/outputs/results"
    output_dir: str = "outputs"


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    num_workers: int = 0


@dataclass
class SimulationConfig:
    K: int = 200
    modes: List[str] = field(default_factory=lambda: ["A"])
    mechanisms: List[str] = field(default_factory=lambda: ["P", "R", "JS"])
    max_week: int = 11
    tie_mode: str = "average"
    topk_jaccard: int = 3
    topk_final: int = 3
    controversy_q: float = 0.2
    soft_percent: bool = False
    soft_rank: bool = False


@dataclass
class SkillConfig:
    model: str = "linear"
    sigma_J: float = 1.0
    J_max_policy: str = "max_obs"  # "max_obs" or "p95_obs"
    std_match: bool = True
    lambda_min: float = 0.1
    lambda_max: float = 6.0
    lambda_iters: int = 30


@dataclass
class JudgesSaveConfig:
    bottom2_base: str = "rank"  # "rank" or "percent"
    eta1: float = 1.0
    eta2: float = 0.5
    deterministic: bool = False
    strength_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])


@dataclass
class MixturePolicyConfig:
    w_grid: List[float] = field(default_factory=lambda: [round(x * 0.1, 2) for x in range(11)])
    kappa: float = 20.0
    kappa_r: float = 10.0


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    judges_save: JudgesSaveConfig = field(default_factory=JudgesSaveConfig)
    mixture: MixturePolicyConfig = field(default_factory=MixturePolicyConfig)

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
    if "paths" in data:
        cfg.paths = PathsConfig(**data["paths"])
    if "runtime" in data:
        cfg.runtime = RuntimeConfig(**data["runtime"])
    if "simulation" in data:
        cfg.simulation = SimulationConfig(**data["simulation"])
    if "skill" in data:
        cfg.skill = SkillConfig(**data["skill"])
    if "judges_save" in data:
        cfg.judges_save = JudgesSaveConfig(**data["judges_save"])
    if "mixture" in data:
        cfg.mixture = MixturePolicyConfig(**data["mixture"])
    return cfg
