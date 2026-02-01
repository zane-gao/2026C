import argparse
import os
from pathlib import Path
from typing import Optional

import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task2.io.social_proxy import load_social_proxy_map, save_social_proxy_csv


def _find_data_file() -> Optional[Path]:
    candidates = [
        "2026美赛C题补充数据集！.xlsx",
        "enriched.csv",
        "2026_MCM_Problem_C_Data.csv",
    ]
    root = Path.cwd()
    for name in candidates:
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return Path(dirpath) / name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate social_proxy.csv from enriched data")
    parser.add_argument("--data", type=str, default=None, help="Enriched data path (xlsx/csv)")
    parser.add_argument("--out", type=str, default="outputs/task1/results/social_proxy.csv", help="Output csv path")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else _find_data_file()
    if data_path is None or not data_path.exists():
        raise FileNotFoundError("Enriched data file not found. Provide --data.")

    proxy_map = load_social_proxy_map(str(data_path))
    if not proxy_map:
        raise RuntimeError("Social proxy columns not found in data file.")

    out_path = Path(args.out)
    save_social_proxy_csv(proxy_map, out_path)
    print(f"Saved social proxy to {out_path}")


if __name__ == "__main__":
    main()
