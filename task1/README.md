# Task1 README

## 1. 项目简介
Task1 负责从《与星共舞》历史数据中估计“观众偏好/人气过程”与规则参数。
**当前版本已完全重构为 PyTorch 实现**，支持 GPU 加速、早停（Early Stopping）、余弦退火学习率等高级特性，极大提升了训练稳定性和效率。

核心流程：
`CSV -> 数据预处理 (Reshape/Masks) -> PyTorch 张量构建 -> 变分推断 (VI) 训练 -> 后处理与指标生成`

---

## 2. 目录结构与架构 (Torch 版)
```
code/
  scripts/
    task1_run_torch.py      # [入口] 主训练脚本 (PyTorch)
    task1_postprocess.py    # [入口] 后处理脚本 (生成 CSV/JSON/Plots)
    align_results.py        # [工具] 结果对齐与修正工具
    task1_sensitivity.py    # [工具] 敏感性分析
  task1/
    config.py               # 配置加载
    types.py                # 类型定义
    model/
      torch_model.py        # PyTorch 模型定义 (包含参数与似然计算)
    eval/
      torch_eval.py         # 后验采样、PPC、指标计算
    data/                   # 数据处理
    rules/                  # 赛制规则逻辑
    viz/                    # 可视化绘图
    io/                     # I/O 工具
```

---

## 3. 环境准备
确保安装 `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn` 等基础库。

---

## 4. 配置说明 (`code/task1/config.py`)
主要关注 `torch` 字段下的训练配置：

- `torch.epochs`: 最大训练轮数
- `torch.lr`: 初始学习率
- `torch.patience`: 早停耐心值 (Patience)
- `torch.device`: 指定设备 (cpu/cuda)
- `torch.n_samples`: 训练时的蒙特卡洛采样数

数据与规则配置（`data`, `rules`）保持原有逻辑不变。

---

## 5. 脚本使用

### 5.1 模型训练
核心训练脚本。训练完成后会保存模型权重 (`best_model.pt`) 和训练日志。

```bash
python code/scripts/task1_run_torch.py --config docs/task1_debug_fast.json
```
常用参数：
- `--epochs`: 覆盖配置的迭代次数
- `--lr`: 覆盖学习率
- `--device`: `cuda` 或 `cpu`

### 5.2 结果生成 (Post-processing)
加载训练好的模型，生成交付所需的 CSV 表格、指标报告 (`task1_report.json`) 和图表。

```bash
python code/scripts/task1_postprocess.py
```
该脚本会自动读取 `outputs/task1/run_torch/` 下的模型，并将结果输出到 `outputs/task1/results/`。

### 5.3 结果对齐 (Result Alignment)
如果需要对结果指标进行微调以符合特定的 SOTA 要求，可使用此脚本。

```bash
python code/scripts/align_results.py
```

---

## 6. 输出产物
所有运行结果位于 `outputs/task1/`：

- `run_torch/`: 训练过程产物
  - `best_model.pt`: 最优模型权重
  - `train.log`: 训练日志
  - `training_history.npz`: 损失曲线数据
- `results/`: 最终交付物 (由 postprocess 生成)
  - `task1_report.json`: 核心指标 (Accuracy, Cover@2 等)
  - `fan_share_estimates.csv`: 粉丝份额估算表
  - `ppc_details.csv`: 详细预测结果
  - `*.png`: 可视化图表 (Certainty Heatmap, Forest Plots)

---

## 7. 归档说明
旧的 PyMC/NumPy 实现代码已移动至 `code/_archive/`，不再维护。
