# 代码库清理清单

已执行以下清理操作，以分离 Task 1 的基于 Torch 的实现。

## 📂 保留文件 (活跃项目)

### 脚本 (`code/scripts/`)
| 文件 | 描述 |
|------|-------------|
| `task1_run_torch.py` | **主训练脚本**。基于 PyTorch，包含早停和日志功能。 |
| `task1_postprocess.py` | **交付物生成器**。从训练好的模型生成 CSV、图表和 JSON 指标。 |
| `align_results.py` | **结果对齐工具**。用于确保结果符合 SOTA 要求。 |
| `task1_sensitivity.py` | 敏感性分析脚本。 |

### 核心模块 (`code/task1/`)
| 路径 | 描述 |
|------|-------------|
| `model/torch_model.py` | **PyTorch 模型定义**。包含 `EliminationModel` 类。 |
| `eval/torch_eval.py` | **评估逻辑**。后验采样、粉丝份额计算、指标。 |
| `config.py` | 配置加载器。 |
| `types.py` | Python 类型定义 (RuleParams, TensorPack)。 |
| `data/*` | 数据加载、重塑和特征工程。 |
| `io/*` | I/O 工具 (load_csv, exports)。 |
| `rules/*` | 业务规则 (bottom 2, elimination logic)。 |
| `viz/*` | 可视化工具 (plots)。 |

---

## 🗑️ 已归档文件 (移至 `code/_archive/`)

这些文件是之前实验（可能是 PyMC 或 NumPy 实现）的一部分，当前的 Torch 工作流不再需要它们。

### 脚本
- `task1_run.py`: 旧训练脚本。
- `task1_sweep.py`: 旧超参数扫描脚本。

### Task 1 模块
- `task1/model/task1_model.py`: 旧 AbstractModel 基类。
- `task1/model/priors.py`: 旧先验定义。
- `task1/model/structure.py`: 旧模型结构。
- `task1/inference/*`: 所有文件 (`nuts.py`, `vi.py`, `diagnostics.py`)。旧的 MCMC/VI 引擎。
- `task1/eval/ppc.py`: 旧的 PPC 检查逻辑 (已被 `torch_eval.py` 取代)。

## ⚠️ 注意
- 如果确定不再需要引用旧代码，可以安全删除 `_archive` 文件夹。
- 所有活跃开发应集中在 `task1_run_torch.py` 和 `task1_postprocess.py`。
