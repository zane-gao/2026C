import argparse
import copy
from pathlib import Path

import sys
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task2.config import load_config
from task2.io.export import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Task2 sensitivity runner (config generator)")
    parser.add_argument("--config", type=str, required=False, help="Base config path")
    parser.add_argument("--out", type=str, default="outputs/task2/sensitivity", help="Output dir")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config(None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    for tie_mode in ["average", "random"]:
        c = copy.deepcopy(cfg)
        c.simulation.tie_mode = tie_mode
        variants.append((f"tie_{tie_mode}", c))

    for scale in cfg.judges_save.strength_grid:
        c = copy.deepcopy(cfg)
        c.judges_save.eta1 *= scale
        c.judges_save.eta2 *= scale
        variants.append((f"js_scale_{scale}", c))

    for name, c in variants:
        path = out_dir / f"config_{name}.json"
        save_json(path, c.to_dict())

    print(f"Generated {len(variants)} config variants in {out_dir}")


if __name__ == "__main__":
    main()
