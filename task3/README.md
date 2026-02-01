# Task3 项目说明

本目录实现 Task3 的完整工程化落地，覆盖面板构建、三通道模型（Judges/Fans/Survival）、消融与稳健性、图件生成与下游接口产物输出。

**全部使用 PyTorch + CUDA 加速版本。**

## 1. 项目架构

```
code/task3/
  config.py                 # 配置读取与保存
  types.py                  # 类型定义
  io/
    dataset.py              # 复用 Task2 数据读入/Mask 构造
    task1_artifact.py       # 复用 Task1 artifact 读取
    social_proxy.py         # 社媒 proxy 读入与处理
    export.py               # parquet/json/png 导出
  data/
    panel.py                # 面板构建 + ref 选择
    features.py             # 特征工程（社媒/缺失/截尾）
    tensors.py              # PyTorch 张量转换
  models/
    torch_m1_judges.py      # PyTorch M1 实现 (CUDA 加速)
    torch_m2_fans.py        # PyTorch M2 实现 (CUDA 加速)
    torch_m3_survival.py    # PyTorch M3 实现 (CUDA 加速)
  eval/
    metrics.py              # R2/RMSE/AUC/Brier
    loso.py                 # Leave-One-Season-Out
    assortative_mating.py   # 匹配相关性检查
  utils/
    progress.py             # 进度显示工具
  viz/
    fig_pro_forest.py       # Pro 效应森林图
    fig_pro_quadrant.py     # Pro 四象限图
    fig_var_decomp.py       # 方差分解图
    fig_delta_beta.py       # 社媒系数差异图
    fig_platform_specificity.py  # 平台特异性图
    fig_shap.py             # 特征重要性图
  scripts/
    task3_build_panel.py    # 构造面板 + k 子采样
    task3_fit_torch.py      # PyTorch/CUDA 训练脚本
    task3_postprocess.py    # 出图、汇总与接口产物生成
```

## 2. 依赖与环境

```
numpy
pandas
matplotlib
pyyaml
openpyxl          # 读取 xlsx 社媒数据
torch>=2.0        # PyTorch (CUDA 11.8+)
scikit-learn      # AUC/Brier 计算
```

如需读写 parquet：
- 需要 pandas + pyarrow 或 fastparquet

## 3. 输入数据

- 原始题目数据：`题目和资料/2026_MCM_Problem_C_Data.csv`
- 社媒补充数据：`2026美赛C题最新补充数据.../2026美赛C题补充数据集！.xlsx`
- Task1 接口产物：`code/task1/outputs/run_torch/`

## 4. 输出结果

每次运行会输出到 `outputs/task3/<run_id>/`，核心文件：

- 数据与复现：
  - `panel.parquet`：最终面板
  - `k_index.json`：k 子样本索引
  - `config.resolved.yaml`：运行配置

- 模型结果：
  - `m1_torch_fixed.parquet` / `m1_torch_random_*.parquet`
  - `m2_torch_fixed.parquet` / `m2_torch_random_*.parquet`
  - `m3_torch_fixed.parquet` / `m3_torch_metrics.json`
  - `*_var_components.json`：方差分解

- 图件：
  - `fig_pro_forest_j.png` / `fig_pro_forest_f.png`
  - `fig_pro_quadrant.png`
  - `fig_var_decomp.png`
  - `fig_delta_beta_social.png`
  - `fig_platform_specificity.png`
  - `fig_shap_residual.png`

- 报告：
  - `joint_report.json`：Pro 跨通道相关 ρ_p
  - `assortative_mating_report.json`：匹配相关性
  - `ablation_report.json`：社媒消融分析
  - `torch_results_summary.json`：模型性能汇总

- 下游接口产物：
  - `task3_artifact.json`（供 Task2/4/5 使用）

## 5. 脚本使用指令

### 5.1 构建面板

```bash
python code/task3/scripts/task3_build_panel.py \
  --social "2026美赛C题最新补充数据.../2026美赛C题补充数据集！.xlsx" \
  --task1 code/task1/outputs/run_torch \
  --run-id run_torch_social \
  --output outputs/task3
```

可选参数：
- `--data` 指定题目 CSV
- `--k` 后验样本数（默认 50）
- `--config` 使用配置文件

### 5.2 训练模型（PyTorch/CUDA）

```bash
python code/task3/scripts/task3_fit_torch.py \
  --run outputs/task3/run_torch_social \
  --device cuda \
  --epochs 500 \
  --patience 30
```

可选参数：
- `--device cuda` 使用 GPU 加速（默认 auto）
- `--lr 0.05` 学习率
- `--epochs 500` 最大迭代次数
- `--patience 30` 早停 patience
- `--no-amp` 禁用混合精度
- `--skip-m1/m2/m3` 跳过指定模型

**性能（RTX 4060 Laptop GPU）：**
| 模型 | 耗时 |
|------|------|
| M1 | ~2.7s |
| M2 | ~1.5s |
| M3 | ~1.0s |
| **总计** | **~5s** |

### 5.3 后处理（图件 + 报告 + 接口产物）

```bash
python code/task3/scripts/task3_postprocess.py --run outputs/task3/run_torch_social
```

### 5.4 消融/缺失敏感性

```bash
## 6. 完整操作流程

```bash
# 1. 构建面板（含社媒数据）
python code/task3/scripts/task3_build_panel.py \
  --social "2026美赛C题最新补充数据.../2026美赛C题补充数据集！.xlsx" \
  --task1 code/task1/outputs/run_torch \
  --run-id run_torch_social \
  --output outputs/task3

# 2. 使用 CUDA 加速训练模型
python code/task3/scripts/task3_fit_torch.py \
  --run outputs/task3/run_torch_social \
  --device cuda \
  --epochs 500 \
  --patience 30

# 3. 后处理：生成图件 + 报告 + 接口产物
python code/task3/scripts/task3_postprocess.py \
  --run outputs/task3/run_torch_social
```

## 7. PyTorch 模型说明

### 7.1 M1 (torch_m1_judges.py)
线性混合效应模型（评委分通道）：
```
J_z ~ X*beta + u[pro_id] + v[celeb_id] + w[season] + eps

其中：
  u ~ N(0, sigma_pro^2)
  v ~ N(0, sigma_celeb^2)
  w ~ N(0, sigma_season^2)
```

### 7.2 M2 (torch_m2_fans.py)
批量处理多个后验样本 k 的混合效应模型（粉丝通道）：
```
y_k ~ X*beta + u_k[pro_id] + v_k[celeb_id] + w[season] + eps
```

所有 k 共享固定效应 beta，每个 k 有独立的随机效应。

### 7.3 M3 (torch_m3_survival.py)
Logistic 回归生存模型（淘汰通道）：
```
logit(P(eliminated)) = X*beta + eta_J * J_z + eta_F * y_k
```

输出 AUC 和 Brier score 评估指标。

## 8. 常见问题

- **parquet 报错**：缺少 pyarrow/fastparquet，请安装。
- **CUDA 不可用**：检查 PyTorch 是否正确安装 CUDA 版本，运行 `torch.cuda.is_available()` 验证。
- **xlsx 读入失败**：确认 openpyxl 已安装。