# Task2 反事实模拟与赛制对比

本目录实现 Task2 的完整工程流程代码：在固定"技能轨迹 S 与偏好轨迹 theta"的前提下，通过共随机数模拟（CRN）比较不同赛制（P / R / JS / W_MIX）的结果，计算指标、因果推断与可视化。

---

## 1. 项目结构

```
code/
  scripts/
    task2_fit_skill.py          # 入口：拟合/校准技能轨迹 S 与 score scale
    task2_run_simulation.py     # 入口：反事实平行宇宙模拟（Mode A/B + P/R/JS/W_MIX）
    task2_postprocess.py        # 入口：计算指标 + 因果推断 + 出图 Fig.1-10
    task2_sensitivity.py        # 入口：敏感性分析配置（tie / JS 强度）
  task2/
    config.py                   # 配置 dataclass + load/save
    types.py                    # 枚举/数据结构
    io/
      dataset.py                # 读取题目 CSV 并建立 season/week/contestant 索引
      task1_artifact.py         # 读取 Task1 Artifact（theta/mu/gamma/epsilon 等）
      export.py                 # 统一导出 npz/json/csv
    data/
      indexing.py               # 索引/淘汰表/活跃表
      calibrate.py              # J_max 与 lambda 校准（STD 匹配）
    skill/
      fit_linear.py             # 线性技能拟合 (a,b,alpha)
      fit_rw.py                 # 可选：随机游走参数估计
      generate.py               # 生成 S/tildeJ/epsilon 样本
    preference/
      mode_a.py                 # Frozen-Preference
      mode_b.py                 # Coupled-Preference
    mechanisms/
      percent.py                # P 规则
      rank.py                   # R 规则
      judges_save.py            # JS 规则
      mixture.py                # W_MIX 规则
      tie.py                    # tie-breaking
    engine/
      crn.py                    # Common Random Numbers
      simulate.py               # 主循环仿真模拟 + 状态更新
    eval/
      metrics_core.py           # Merit/Pop/Div/Regret/Cap 等指标
      causal.py                 # ITE/ATE
      controversy.py            # 争议性选取
    viz/
      fig_pipeline.py           # Fig.1
      fig_parallel_universe.py  # Fig.2
      fig_rank_bar.py           # Fig.3
      fig_survival_ite_ate.py   # Fig.4
      fig_margin.py             # Fig.5
      fig_bias_box.py           # Fig.6
      fig_js_scan.py            # Fig.7
      fig_radar_pareto.py       # Fig.8
      fig_entropy_div.py        # Fig.9
      fig_weight_sensitivity.py # Fig.10
```

---

## 2. 环境依赖

- Python 3.10+（3.12 也可）
- numpy, matplotlib
- 本实现不依赖 pandas，避免环境冲突

---

## 3. 数据与输入

- 原始数据：`题目和资料/2026_MCM_Problem_C_Data.csv`
- Task1 输出（Artifact）：默认路径 `code/task1/outputs/results`
  - 至少需包含 `theta`（或 `F` 可反推）以及 `Valid` / `A_init` 等
  - 若缺少 `mu/gamma/epsilon`，Mode B 将自动降级为 Mode A

如 Task1 输出路径不同，请修改：`code/task2/config/task2_full.json`。

---

## 4. 配置说明（主要字段）

配置文件：`code/task2/config/task2_full.json`

- `paths`
  - `data_csv`：题目 CSV 路径
  - `task1_artifact`：Task1 输出路径
  - `output_dir`：统一输出目录
- `simulation`
  - `K`：平行宇宙数量
  - `modes`：`["A","B"]`
  - `mechanisms`：`["P","R","JS","W_MIX"]`
  - `max_week`：最大仿真周
  - `tie_mode`：`average` / `random`
  - `topk_jaccard` / `topk_final`：一致性检验的 TopK
- `skill`
  - `J_max_policy`：`max_obs` / `p95_obs`
  - `std_match`：是否做 STD 匹配
- `judges_save`
  - `bottom2_base`：`rank` / `percent`
  - `eta1`, `eta2`：Save 强度
- `mixture`
  - `w_grid`：混合权重扫描
  - `kappa`, `kappa_r`：soft 排序温度

---

## 5. 使用流程（推荐）

### 步骤 1：拟合技能轨迹

```bash
python code/scripts/task2_fit_skill.py --config code/task2/config/task2_full.json
```

输出：`outputs/task2/skill_cache/skill_params_*.npz`

---

### 步骤 2：反事实仿真模拟

```bash
python code/scripts/task2_run_simulation.py --config code/task2/config/task2_full.json
```

输出：`outputs/task2/run_YYYYMMDD_HHMMSS/` 下的

- `trajectories_{mode}_{mechanism}.npz`
- `week_stats_{mode}_{mechanism}.csv`
- `config.json`

---

### 步骤 3：计算指标与出图

```bash
python code/scripts/task2_postprocess.py --run outputs/task2/run_20260201_112324
```

输出：

- `task2_report.json`
- `season_summary.csv`
- `ite_ate.csv`
- `fig01_*.png` ~ `fig10_*.png`

---

## 6. 敏感性分析（可选）

可生成多组配置（tie / JS 强度）：

```bash
python code/scripts/task2_sensitivity.py --config code/task2/config/task2_full.json --out outputs/task2/sensitivity
```

之后对生成的配置批量运行模拟。

python code/scripts/task2_postprocess.py --run outputs/task2/run_20260201_112324 --social "2026美赛C题最新补充数据+预处理后数据！！！适合模型检验+提高结果准确度！/2026美赛C题补充数据集！.xlsx"

---

## 7. 输出说明

- `trajectories_*.npz`
  - `winner`, `final_rank`, `elim_matrix`, `active_mask`
  - `S_bar`, `F_bar`, `margin`
  - JS 规则还包含 `bottom2_mask`, `saved_mask`
- `task2_report.json`：全局指标（Merit/Pop/Div/Regret/Cap/IR 等）
- `season_summary.csv`：分赛季摘要（存活率、TopK 标识、Spearman、Div）
- `ite_ate.csv`：ITE/ATE + 置信信息

---

## 8. 常见问题

1) **Mode B 无法运行？**

   - 若 Task1 输出没有 `mu/gamma/epsilon`，将自动走 Mode A 逻辑替代。
2) **图片是空白？**

   - 当前 `viz/` 提供的是占位图，保证流程的结构对接。
   - 请在 `code/task2/viz/*.py` 中替换为真实绘图。
3) **Conda 环境乱码（GBK 编码）？**

   - 本流程不依赖 conda，直接 `python` 运行即可。

---

## 9. 快速测试（小规模参数）

- `K=200`
- `modes=["A"]`
- `mechanisms=["P","R","JS"]`

验证通过后再扩展到全量实验。

---

## 10. 扩展建议

- 在 `task2_postprocess.py` 中加入 Pareto 最优/权重敏感性分析
- 在 `viz/` 中实现更多的绘图格式
- 在 `skill/fit_rw.py` 中加入随机游走技能轨迹
