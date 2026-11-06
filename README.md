# space4_real · 无人机群通信中继（势场学习版）

> **本仓库只保留代码、设施与使用说明。**
> 信道表、训练权重、评估 CSV、实验产物、论文与文献不在本仓库；它们通过
> GitHub Releases 或本地实验档案分发（见 §4）。

## 1. 项目是什么

无人机作为转发中继：用户 → 无人机 → 地面宿主基站 TBS。项目联合决定每架
无人机的带宽、接入优先级、功率与平面位置，目标是把机间负载锚在目标值
`u*`，同时保持覆盖与吞吐。

核心设计：**场定方向，执行器定速度。**

```
a = clip( Λ · η · (X* − X), ±U )
```

- `X*`：场闭式目标（带宽锚定、π 对偶次梯度、功率标称、Lloyd 质心）；
- `η`：学习执行器输出的五维正增益，逐机逐维，只改速度、不改方向；
- `Λ = diag(0.2, 0.2, 1, 1, 1)`；`U` 为物理单步界。

理论篇的 D1–D3 条件核验见 [docs/08_理论核验.md](docs/08_理论核验.md)。

## 2. 目录

```
code/                       统一入口、环境、场、执行器、训练器、外部基线
  run.py                    训练/算法统一入口
  evaluate.py               统一评估
  field.py                  方向决定器（闭式目标）
  executors/                执行器注册表（max/tuned/net/fparam）
  algorithms/               算法注册表
  trainer.py                TD3 训练管线
  uav_env.py                环境本体（场景/人群/信道/关联/负载/奖励）
  channel_env.py            环境构造唯一入口
  heuristics.py             免训/定参基线
  ddpga.py dqnm.py mfdqn.py tmarl.py   外部基线
  crowd_sfm.cpp fastlink3.c           C 核源码
  shim/                      C 核编译所需头文件
deps/                       编译依赖（lightsfm 头文件、MuJoCo-gym 源码/资产）
docs/                       使用说明（00 环境、06 代码结构、08 理论核验）
artifacts/theory_audit/     理论核验脚本与三级说明（结果不入库）
.github/workflows/          训练/评估/核验工作流
```

数据与产物（不入库）：`env_data/`、`artifacts/experiments/`、`results/`、
`paper_work/`、`ref/`、`bin/`、`tmp/`。

## 3. 环境

Python 3.11（项目使用 `py311` 环境；torch 已预装，**禁止**执行任何
pip/conda install/uninstall/upgrade torch 操作）。

```bash
conda activate py311
cd code
```

C 核编译（Linux）：

```bash
cd code
g++ -O3 -fopenmp -shared -fPIC -I shim -I ../deps/lightsfm/include \
    crowd_sfm.cpp -o crowd_sfm.so
gcc -O3 -funroll-loops -fopenmp -shared -fPIC fastlink3.c -o fastlink3.so -lm
```

Windows 下对应产物为 `crowd_sfm.dll` / `fastlink3.dll`（仓库不提交二进制）。

常用环境变量：

| 变量 | 含义 |
|---|---|
| `PF_CHANNEL=umi` | 主实验信道档（表拟合微站） |
| `PF_SHADOW_DB=6` | 固定阴影 σ（评估复现） |
| `PF_DXY_MAX=0.5` | 位置维单步上限（m/步，与场景一致） |
| `PF_REAL_OBS=1` | 真实全局量退化（0.5 s 更新 + 10 m 噪声） |

## 4. 数据与权重（GitHub Releases）

| 资产 | 内容 |
|---|---|
| `chan-tables` | `chan_12/30/60/100_uma_2ghz.npz` 信道表 |
| `real-code` | 历史代码覆盖包（当前仓库代码已更新，不再需要覆盖） |
| `real_ckpts.tar.gz` | 真实信息退化线训练权重 |

```bash
mkdir -p env_data/channel
wget https://github.com/jhzAliyy/space4-run/releases/download/chan-tables/chan_12_uma_2ghz.npz \
     -O env_data/channel/chan_12_uma_2ghz.npz
```

训练权重/评估 CSV 按实验格归档，不在本仓库。

## 5. 运行

```bash
cd code

# 训练（从零唯一模式）
python run.py --algo learned --scene flow --n 12 --total-rollouts 1200

# 评估（清单行：algo tau seed scene t_comm anchor chan n [ckpt_task] [preset]）
python evaluate.py ../docs/example_eval.txt out.csv

# 理论核验（n3~n100，单任务）
python ../artifacts/theory_audit/code/theory_audit.py --n 12 --scene flow --seed 0
```

训练/评估/核验在 GitHub Actions 上通过 `workflow_dispatch` 运行；工作流
清单与用途见 [.github/workflows/README.md](.github/workflows/README.md)。

## 6. 文档

| 文档 | 内容 |
|---|---|
| [docs/00_环境与构建.md](docs/00_环境与构建.md) | 环境、编译、数据下载、命令 |
| [docs/06_代码结构.md](docs/06_代码结构.md) | 代码分层、单一实现点、产物路径 |
| [docs/08_理论核验.md](docs/08_理论核验.md) | D1–D3 条件在 n3~n100 的数值核验 |
| [artifacts/theory_audit/说明.md](artifacts/theory_audit/说明.md) | 核验三级说明（回链 08） |

完整研究文档（01–05、07、操作记录、未完成实验、致命错误文档）与论文层
保留在本地研究档案，不随仓库分发。

## 7. 归档纪律

- 仓库只提交源代码、编译依赖、工作流与使用说明；
- 权重、评估 CSV、曲线、信道表、论文与文献一律不入库；
- 训练中断即作废，从零重跑；无续跑代码路径。
