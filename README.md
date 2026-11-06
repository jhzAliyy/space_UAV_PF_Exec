# space4_real · 无人机群通信中继（势场学习版）

> 场定方向、神经网络执行器定逐维增益。规模轴 n3 → n12 → n30 → n60 → n100 → n200，
> 场景 flow / burst，三种子。n3~n100 全链闭环；n200 全档按用户令永久终止。

本文件是**一级文档**，只做导航与概览。研究细节在二级文档（`docs/`），
每个实验格子的身份与实际数据在其目录下的三级文档（`说明.md`）。

## 三层文档体系

```
README.md（一级，本文件）
  └── docs/*.md（二级）
        └── artifacts/experiments/<格子>/说明.md（三级）
```

## 二级文档导航

| 文档 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 项目结构、目录约定、归档纪律、逐格档案索引 |
| [docs/01_数学.md](docs/01_数学.md) | 均衡、正则性与收敛的理论部分 |
| [docs/02_势场建模.md](docs/02_势场建模.md) | 势场构造与执行器接口 |
| [docs/03_实验.md](docs/03_实验.md) | **实验矩阵状态唯一源**：口径、全格状态、结果数字、判定 |
| [docs/04_基线.md](docs/04_基线.md) | 外部基线的方法细节 |
| [docs/05_总览.md](docs/05_总览.md) | 全局总览 |
| [docs/06_代码结构.md](docs/06_代码结构.md) | 代码分层与模块职责 |
| [docs/07_猜想.md](docs/07_猜想.md) | 架构猜想与验证记录 |
| [docs/08_理论核验.md](docs/08_理论核验.md) | 理论篇 D1–D3 条件在 n3~n100 的数值核验 |
| [docs/未完成实验.md](docs/未完成实验.md) | 在跑事项与巡检规程 |
| [docs/操作记录.md](docs/操作记录.md) | 全部操作史（批次、事故、纠正） |
| [docs/致命错误文档.md](docs/致命错误文档.md) | 致命错误与铁律清单 |

## 矩阵概览

| 块 | 规模 | 格数 | 需训练 |
|---|---|---|---|
| 主表（势场学习版 + 基线体系：调参版/定参版/外部基线） | n3~n200 × flow/burst | 70 | 10 |
| 场族对照（真纯势场 / 调优势场） | n12~n200 × flow/burst | 20 | 0 |
| 外部基线（ddpga / dqnm） | n12~n200 × flow/burst | 20 | 20 |
| 原生场景（四场景 × 原方法 + 客场） | cheng7 / rizvi3 / shen3 / jiang5 | 8 | 6 |
| 真实全局量退化线（含两组对照） | n12~n200 × flow/burst | 30 | 10 |
| **合计** | | **148** | **46** |

三种子覆盖（n200 除外）：**126 格全部训练与评估齐备**——主表 30、真实线 30、
外部基线 48、客场 18。逐格清点与结果见 [docs/03_实验.md](docs/03_实验.md) §6。

**核心结论**：主场景对垒 22 终态格 21 赢 1 输（主判据公平指数 J），实验排序为
学习版 > 调参版 > 其余基线；基线体系由调参版、定参版与外部基线共同构成，外部
基线经适配性改造与调优，主方法全规模同一套超参、无调优裸跑。客场四场景三胜
一不败、零败零死机；真实退化线 59/60 档死机为零、覆盖 ≥0.988。

## 快速入口

| 目的 | 去哪 |
|---|---|
| 查某格训练轮数、评估数字、来源 | `artifacts/experiments/<格子>/说明.md`（三级文档） |
| 查矩阵整体状态与判定 | [docs/03_实验.md](docs/03_实验.md) |
| 查某格原始数据 | `artifacts/experiments/<格子>/ckpt/`、`eval/<格子>.csv` |
| 跑训练 | `cd code && python run.py --algo learned --scene flow --n 12 --total-rollouts 1200` |
| 跑评估 | `cd code && python evaluate.py ../artifacts/lists/<清单>.txt out.csv` |
| 查操作史 / 事故 | [docs/操作记录.md](docs/操作记录.md)、[docs/致命错误文档.md](docs/致命错误文档.md) |

## 目录速览

```
code/        唯一入口 run.py / evaluate.py 与全部算法实现
docs/        二级文档（上表）
artifacts/experiments/<格子>/   ckpt/ + eval/ + 说明.md（三级文档）
artifacts/lists/               评估清单
artifacts/eval/                早期汇总评估表（历史来源，逐格数据已拆入各格 eval/）
results/curves/                评估逐场曲线
env_data/    信道表
deps/        MuJoCo-drones-gym、lightsfm
paper_work/  论文层（理论篇 / 工程篇 / 叙事 / 方案）
ref/         文献库
ext/         外部对照实现
ops/         云端通道与运维脚本
bin/         垃圾件区（散文件、错位件、空目录容器；只收纳不删除）
```

## 命名规范（格子目录）

| 类型 | 规则 | 例 |
|---|---|---|
| 训练格 | `{场景}[_n{规模}]_{方法}_tau5_s{种子}_{信道}_srand` | `flow_n30_learned_tau5_s1_umi_srand` |
| 客场 | `native_{方法}_{场景}[_s{种子}]` | `native_learned_cheng7_s2` |
| 免训格 | `{方法}_{场景}_n{规模}`（无种子维） | `pfg_flow_n12` |
| 规模缺省 | n12 不写 `_n12`（与 n3 同表，`layout_n=max(12,n)`） | `burst_learned_tau5_s0_umi_srand` |

## 归档纪律

- 三件套：`ckpt/`（latest.pt、progress.json、train_log.csv）+ `eval/<格子>.csv` + `说明.md`。
- **三件齐备且 03 文档状态同步，该格才算闭环**；文件到位不算完成。
- 三级文档的「所属」字段回指 [docs/03_实验.md](docs/03_实验.md)，形成闭环。
