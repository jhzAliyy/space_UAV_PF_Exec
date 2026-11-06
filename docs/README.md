> ← [项目总览](../README.md)

# space4_real · 项目结构

无人机群通信中继：**势场学习版**（场出方向 + 神经网络执行器出逐维增益）对比外部
同类与相邻问题基线。规模轴 n3 → n12 → n30 → n60 → n100 → n200，
两场景 flow/burst，三种子。n3~n100 已全链闭环：终态判定 22 格
21 赢 1 输（主判据公平指数 J，var_u 口径同为 21/22；本文全部 22 格
J≥0.89、20 格 ≥0.99；判定明细 artifacts/eval_summary/win_verdict.md；
对外表述 paper_work/工程.md §7）。客场四场景三胜一不败、零败零死机；
真实信息退化线十格主方法全稳（n3~n60 死机全 0、n100 近零、覆盖 ≥0.999）。

## 基本设计

**场固定方向，执行器决定逐维增益。**

- 场（field.py）：从滞后 0.5 s 的需求快照闭式算出最优点，输出包含目标
  X*、方向 s（标量维符号 ±1，位置维平面单位矢量）与物理步幅。
  零参数、零力度。
- 执行器（executors/，拔插）：拿到场输出包后处理再输出动作。
  - `max`：模长恒 1（真纯势场——速度恒最大，没有任何其他东西）
  - `net`：神经网络出五维正增益 η∈[0.50,2.01]，动作 = clip(η⊙(X*−X), ±物理
    步幅)，带宽/价格维另乘松弛 0.2；起步 η≡1（输出层零初始化 = 纯势场）
  - `tuned`：2026-10 调参律遗迹（对照专用，常数显式化）
- 训练只有从零模式（无续跑代码路径）。
- 全部超参走命令行，一个参数换算法/执行器/口径。

## 目录

```
code/
  run.py            统一入口（唯一命令行，13 臂注册表）
  evaluate.py       统一评估（场族同一条链；基线体系同列表分派）
  field.py          方向决定器（唯一闭式分支；FieldReading 输出包）
  executors/        执行器注册表（base/max/tuned/net，一行注册即插）
  algorithms/       算法注册表（field/tuned/learned=场+执行器预绑；
                    ddpga/dqnm/mfdqn/tmarl=外部基线；
                    rand/fixed/greedy/rr/shen/jiang=定参/启发式基线；
                    共同构成基线体系）
  trainer.py        TD3 管线（连续步流 256、探索噪声 0.3、梯度裁剪、
                    双向带早停、isfinite 护栏、drain_stats）
  uav_env.py        环境（物理/奖励/关联；STEP_* 为环境物理属性）
  config.py         环境与奖励参数（执行器零参数；调优遗迹两常数在尾注）
  channel_env.py    环境构造唯一入口
  heuristics.py     定参/启发式基线控制器
  pf_channel.py     解析信道标定（UMa/UMi 3GPP 36.777 回归）
  dqnm.py ddpga.py mfdqn.py tmarl.py    外部基线四臂（旧实现逐位沿用）
  pfmath.py         观测映射数学
  fastlink3.c / crowd_sfm.cpp / crowd_sfm.dll / fastlink3.dll
env_data/channel/   信道表（n12~n200 与原生场景）
deps/               MuJoCo-drones-gym、lightsfm
paper_work/         论文层（理论篇/工程篇/叙事/方案，直接复制）
ref/                文献库（PDF、36.777 标准原件、提取文本）
ext/                外部对照实现（uav_bs_ctrl：drqn / madrqn）
ops/                云端通道与运维脚本（连接凭据在本目录，不入文档）
artifacts/
  experiments/<格子>/     实验格：ckpt/ + eval/ + 说明.md
  lists/                评估清单
  logs/                   运行日志
results/curves/     评估逐场曲线（评估器写）
docs/               01 数学 · 02 势场建模 · 03 实验（矩阵状态唯一源）·
                    04 基线 · 05 总览 · 06 代码结构 · 07 猜想 ·
                    未完成实验（在跑与巡检）· 操作记录 · 致命错误文档
```

## 命令（全部臂同一套）

```
python run.py --algo learned --scene flow --n 12 --total-rollouts 1200
python run.py --algo field   --scene burst --n 30          # 免训
python run.py --algo tuned   --scene flow --n 12           # 免训
python run.py --algo ddpga   --scene flow --n 30 --total-rollouts 1200
python evaluate.py artifacts/lists/<x>.txt out.csv
```

环境变量口径：`PF_CHANNEL=umi PF_DXY_MAX=0.5`；真实全局量退化
（0.5 s 过期 + 10 m 位置噪声）训练评估同开关 `PF_REAL_OBS=1`。
基线原生场景按其物理口径（如 shen3 需 `PF_DXY_MAX=5.0`）。

评估 ckpt 读取顺序：`artifacts/experiments/<task>/ckpt/`（归档格，优先）
→ `code/ckpt/<task>/`（训练暂存兜底）。

## 观测与动作（神经网络执行器）

- 观测 22 维/机：12 维环境观测（÷OBS_NORM）+ 5 维场方向 + 5 维饱和
  残距 clip(|x*−x|/U, 0, 1)。残距只进观测不进动作。
- 输出 5 维/机：η = exp(0.7·tanh(logit)) ∈ [0.50,2.01]^5（逐维正增益）。
- 合成：a = clip(Λη⊙(X*−X), ±U)；Λ=diag(0.2,0.2,1,1,1)（带宽/价格松弛），
  U=(5.0,0.5,0.1,0.5,0.5) 物理单步界兜底。
- 起步 η≡1 = 纯势场（输出层零权重零偏置，热启动）。
- critic 逐机（奖励是逐机 (N,)，环境事实），输入 27 维 = 22 观测 + 5 增益
  （η 归一化）。

## 归档纪律

- 目录结构：`artifacts/experiments/<格子>/`，内含
  - `ckpt/`：`latest.pt`、`progress.json`、`train_log.csv`（snaps 不入归档）
  - `eval/<格子>.csv`：三档 2000/3000/4000
  - `说明.md`：三级文档（11 字段：算法/规模/场景/口径/训练/评估/来源/归档/状态/时间/所属）
- 命名规范：
  - 训练格 `{场景}[_n{规模}]_{方法}_tau5_s{种子}_{信道}_srand`（n12 不写 `_n12`）
  - 客场 `native_{方法}_{场景}[_s{种子}]`
  - 免训格 `{方法}_{场景}_n{规模}`（无种子维）
- **三件齐备且 03 文档状态同步，该格才算闭环**；文件到位不算完成。
- 早期汇总评估表（`artifacts/eval/`）保留为历史来源，逐格数据已按 `ckpt_task`
  字段拆入各格 `eval/`；`artifacts/eval_summary/` 存判定与汇总。
- 退出正式归档的物件移入项目根 `bin/`，只收纳不删除，见 `bin/说明.md`。
- 旧命名迁移格（`ddpga_*` / `dqnm_*` 四个 n100 目录）保留为历史来源，其数据已
  复制到对应规范格。
- 旧项目学习执行器产物一律隔离不转移；本目录 learned 格全部为本树
  从零新训（无续跑模式），基线体系（外部基线/定参/调参）按归档批次转移。

## 逐格档案索引（215 格）

> 每个格子一行；链接指向该格三级文档 `说明.md`。路径相对本文件所在 `docs/`。


### 主表·势场学习版（30 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `burst_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n3_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n3_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n3_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n3_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n3_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n3_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_learned_tau5_s0_umi_srand/说明.md) | — | 1 份 |
| `flow_n30_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n3_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n3_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n3_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n3_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n3_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n3_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_learned_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_learned_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_learned_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_learned_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_learned_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_learned_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |

### 真实全局量退化线（30 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `burst_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n3_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n3_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n3_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n3_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n3_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n3_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n3_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n3_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n3_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n3_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n3_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n3_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_learned_real_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_learned_real_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_learned_real_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_learned_real_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_learned_real_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_learned_real_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |

### 相邻 SOTA（主场景）（48 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `burst_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n12_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n12_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n12_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n12_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n12_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n12_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n60_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n60_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n100_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n12_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n12_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n12_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n12_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n12_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n12_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_ddpga_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_ddpga_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_ddpga_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_ddpga_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_ddpga_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_ddpga_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_dqnm_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_dqnm_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_dqnm_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_dqnm_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n60_dqnm_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n60_dqnm_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |

### SOTA·旧命名迁移格（4 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `ddpga_burst_n100` | [说明.md](../artifacts/experiments/ddpga_burst_n100/说明.md) | — | 1 份 |
| `ddpga_flow_n100` | [说明.md](../artifacts/experiments/ddpga_flow_n100/说明.md) | — | 1 份 |
| `dqnm_burst_n100` | [说明.md](../artifacts/experiments/dqnm_burst_n100/说明.md) | — | 1 份 |
| `dqnm_flow_n100` | [说明.md](../artifacts/experiments/dqnm_flow_n100/说明.md) | — | 1 份 |

### 客场（原生场景）（18 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `native_ddpga_cheng7` | [说明.md](../artifacts/experiments/native_ddpga_cheng7/说明.md) | 有 | 1 份 |
| `native_ddpga_cheng7_s1` | [说明.md](../artifacts/experiments/native_ddpga_cheng7_s1/说明.md) | 有 | 1 份 |
| `native_ddpga_cheng7_s2` | [说明.md](../artifacts/experiments/native_ddpga_cheng7_s2/说明.md) | 有 | 1 份 |
| `native_dqnm_rizvi3` | [说明.md](../artifacts/experiments/native_dqnm_rizvi3/说明.md) | 有 | 1 份 |
| `native_dqnm_rizvi3_s1` | [说明.md](../artifacts/experiments/native_dqnm_rizvi3_s1/说明.md) | 有 | 1 份 |
| `native_dqnm_rizvi3_s2` | [说明.md](../artifacts/experiments/native_dqnm_rizvi3_s2/说明.md) | 有 | 1 份 |
| `native_learned_cheng7` | [说明.md](../artifacts/experiments/native_learned_cheng7/说明.md) | 有 | 1 份 |
| `native_learned_cheng7_s1` | [说明.md](../artifacts/experiments/native_learned_cheng7_s1/说明.md) | 有 | 1 份 |
| `native_learned_cheng7_s2` | [说明.md](../artifacts/experiments/native_learned_cheng7_s2/说明.md) | 有 | 1 份 |
| `native_learned_jiang5` | [说明.md](../artifacts/experiments/native_learned_jiang5/说明.md) | 有 | 1 份 |
| `native_learned_jiang5_s1` | [说明.md](../artifacts/experiments/native_learned_jiang5_s1/说明.md) | 有 | 1 份 |
| `native_learned_jiang5_s2` | [说明.md](../artifacts/experiments/native_learned_jiang5_s2/说明.md) | 有 | 1 份 |
| `native_learned_rizvi3` | [说明.md](../artifacts/experiments/native_learned_rizvi3/说明.md) | 有 | 1 份 |
| `native_learned_rizvi3_s1` | [说明.md](../artifacts/experiments/native_learned_rizvi3_s1/说明.md) | 有 | 1 份 |
| `native_learned_rizvi3_s2` | [说明.md](../artifacts/experiments/native_learned_rizvi3_s2/说明.md) | 有 | 1 份 |
| `native_learned_shen3` | [说明.md](../artifacts/experiments/native_learned_shen3/说明.md) | 有 | 1 份 |
| `native_learned_shen3_s1` | [说明.md](../artifacts/experiments/native_learned_shen3_s1/说明.md) | 有 | 1 份 |
| `native_learned_shen3_s2` | [说明.md](../artifacts/experiments/native_learned_shen3_s2/说明.md) | 有 | 1 份 |

### 定参/启发式基线（48 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `fixed_burst_n100` | [说明.md](../artifacts/experiments/fixed_burst_n100/说明.md) | — | 1 份 |
| `fixed_burst_n12` | [说明.md](../artifacts/experiments/fixed_burst_n12/说明.md) | — | 1 份 |
| `fixed_burst_n30` | [说明.md](../artifacts/experiments/fixed_burst_n30/说明.md) | — | 1 份 |
| `fixed_burst_n60` | [说明.md](../artifacts/experiments/fixed_burst_n60/说明.md) | — | 1 份 |
| `fixed_flow_n100` | [说明.md](../artifacts/experiments/fixed_flow_n100/说明.md) | — | 1 份 |
| `fixed_flow_n12` | [说明.md](../artifacts/experiments/fixed_flow_n12/说明.md) | — | 1 份 |
| `fixed_flow_n30` | [说明.md](../artifacts/experiments/fixed_flow_n30/说明.md) | — | 1 份 |
| `fixed_flow_n60` | [说明.md](../artifacts/experiments/fixed_flow_n60/说明.md) | — | 1 份 |
| `greedy_burst_n100` | [说明.md](../artifacts/experiments/greedy_burst_n100/说明.md) | — | 1 份 |
| `greedy_burst_n12` | [说明.md](../artifacts/experiments/greedy_burst_n12/说明.md) | — | 1 份 |
| `greedy_burst_n30` | [说明.md](../artifacts/experiments/greedy_burst_n30/说明.md) | — | 1 份 |
| `greedy_burst_n60` | [说明.md](../artifacts/experiments/greedy_burst_n60/说明.md) | — | 1 份 |
| `greedy_flow_n100` | [说明.md](../artifacts/experiments/greedy_flow_n100/说明.md) | — | 1 份 |
| `greedy_flow_n12` | [说明.md](../artifacts/experiments/greedy_flow_n12/说明.md) | — | 1 份 |
| `greedy_flow_n30` | [说明.md](../artifacts/experiments/greedy_flow_n30/说明.md) | — | 1 份 |
| `greedy_flow_n60` | [说明.md](../artifacts/experiments/greedy_flow_n60/说明.md) | — | 1 份 |
| `jiang_burst_n100` | [说明.md](../artifacts/experiments/jiang_burst_n100/说明.md) | — | 1 份 |
| `jiang_burst_n12` | [说明.md](../artifacts/experiments/jiang_burst_n12/说明.md) | — | 1 份 |
| `jiang_burst_n30` | [说明.md](../artifacts/experiments/jiang_burst_n30/说明.md) | — | 1 份 |
| `jiang_burst_n60` | [说明.md](../artifacts/experiments/jiang_burst_n60/说明.md) | — | 1 份 |
| `jiang_flow_n100` | [说明.md](../artifacts/experiments/jiang_flow_n100/说明.md) | — | 1 份 |
| `jiang_flow_n12` | [说明.md](../artifacts/experiments/jiang_flow_n12/说明.md) | — | 1 份 |
| `jiang_flow_n30` | [说明.md](../artifacts/experiments/jiang_flow_n30/说明.md) | — | 1 份 |
| `jiang_flow_n60` | [说明.md](../artifacts/experiments/jiang_flow_n60/说明.md) | — | 1 份 |
| `rand_burst_n100` | [说明.md](../artifacts/experiments/rand_burst_n100/说明.md) | — | 1 份 |
| `rand_burst_n12` | [说明.md](../artifacts/experiments/rand_burst_n12/说明.md) | — | 1 份 |
| `rand_burst_n30` | [说明.md](../artifacts/experiments/rand_burst_n30/说明.md) | — | 1 份 |
| `rand_burst_n60` | [说明.md](../artifacts/experiments/rand_burst_n60/说明.md) | — | 1 份 |
| `rand_flow_n100` | [说明.md](../artifacts/experiments/rand_flow_n100/说明.md) | — | 1 份 |
| `rand_flow_n12` | [说明.md](../artifacts/experiments/rand_flow_n12/说明.md) | — | 1 份 |
| `rand_flow_n30` | [说明.md](../artifacts/experiments/rand_flow_n30/说明.md) | — | 1 份 |
| `rand_flow_n60` | [说明.md](../artifacts/experiments/rand_flow_n60/说明.md) | — | 1 份 |
| `rr_burst_n100` | [说明.md](../artifacts/experiments/rr_burst_n100/说明.md) | — | 1 份 |
| `rr_burst_n12` | [说明.md](../artifacts/experiments/rr_burst_n12/说明.md) | — | 1 份 |
| `rr_burst_n30` | [说明.md](../artifacts/experiments/rr_burst_n30/说明.md) | — | 1 份 |
| `rr_burst_n60` | [说明.md](../artifacts/experiments/rr_burst_n60/说明.md) | — | 1 份 |
| `rr_flow_n100` | [说明.md](../artifacts/experiments/rr_flow_n100/说明.md) | — | 1 份 |
| `rr_flow_n12` | [说明.md](../artifacts/experiments/rr_flow_n12/说明.md) | — | 1 份 |
| `rr_flow_n30` | [说明.md](../artifacts/experiments/rr_flow_n30/说明.md) | — | 1 份 |
| `rr_flow_n60` | [说明.md](../artifacts/experiments/rr_flow_n60/说明.md) | — | 1 份 |
| `shen_burst_n100` | [说明.md](../artifacts/experiments/shen_burst_n100/说明.md) | — | 1 份 |
| `shen_burst_n12` | [说明.md](../artifacts/experiments/shen_burst_n12/说明.md) | — | 1 份 |
| `shen_burst_n30` | [说明.md](../artifacts/experiments/shen_burst_n30/说明.md) | — | 1 份 |
| `shen_burst_n60` | [说明.md](../artifacts/experiments/shen_burst_n60/说明.md) | — | 1 份 |
| `shen_flow_n100` | [说明.md](../artifacts/experiments/shen_flow_n100/说明.md) | — | 1 份 |
| `shen_flow_n12` | [说明.md](../artifacts/experiments/shen_flow_n12/说明.md) | — | 1 份 |
| `shen_flow_n30` | [说明.md](../artifacts/experiments/shen_flow_n30/说明.md) | — | 1 份 |
| `shen_flow_n60` | [说明.md](../artifacts/experiments/shen_flow_n60/说明.md) | — | 1 份 |

### 场族对照·调优势场（旧链 pfg）（8 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `pfg_burst_n100` | [说明.md](../artifacts/experiments/pfg_burst_n100/说明.md) | — | 1 份 |
| `pfg_burst_n12` | [说明.md](../artifacts/experiments/pfg_burst_n12/说明.md) | — | 1 份 |
| `pfg_burst_n30` | [说明.md](../artifacts/experiments/pfg_burst_n30/说明.md) | — | 1 份 |
| `pfg_burst_n60` | [说明.md](../artifacts/experiments/pfg_burst_n60/说明.md) | — | 1 份 |
| `pfg_flow_n100` | [说明.md](../artifacts/experiments/pfg_flow_n100/说明.md) | — | 1 份 |
| `pfg_flow_n12` | [说明.md](../artifacts/experiments/pfg_flow_n12/说明.md) | — | 1 份 |
| `pfg_flow_n30` | [说明.md](../artifacts/experiments/pfg_flow_n30/说明.md) | — | 1 份 |
| `pfg_flow_n60` | [说明.md](../artifacts/experiments/pfg_flow_n60/说明.md) | — | 1 份 |

### 其他（a2a / 特别档）（9 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `burst_n100_a2a_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_a2a_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n100_learnedfull_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/burst_n100_learnedfull_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_a2a_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_a2a_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `burst_n30_a2a_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/burst_n30_a2a_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |
| `flow_a2a_qs7_zero` | [说明.md](../artifacts/experiments/flow_a2a_qs7_zero/说明.md) | 有 | 1 份 |
| `flow_a2a_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_a2a_tau5_s0_umi_srand/说明.md) | 有 | 0 份 |
| `flow_n100_a2a_tau5_s0_umi_srand` | [说明.md](../artifacts/experiments/flow_n100_a2a_tau5_s0_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_a2a_tau5_s1_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_a2a_tau5_s1_umi_srand/说明.md) | 有 | 1 份 |
| `flow_n30_a2a_tau5_s2_umi_srand` | [说明.md](../artifacts/experiments/flow_n30_a2a_tau5_s2_umi_srand/说明.md) | 有 | 1 份 |

### n200（按用户令永久终止档）（20 格）

| 格子 | 三级文档 | 训练件 | 评估数据 |
|---|---|---|---|
| `burst_n200_dqnm_tau5_s0_uma_srand` | [说明.md](../artifacts/experiments/burst_n200_dqnm_tau5_s0_uma_srand/说明.md) | 有 | 1 份 |
| `burst_n200_learned_real_tau5_s0_uma_srand` | [说明.md](../artifacts/experiments/burst_n200_learned_real_tau5_s0_uma_srand/说明.md) | 有 | 0 份 |
| `burst_n200_learned_tau5_s0_uma_srand` | [说明.md](../artifacts/experiments/burst_n200_learned_tau5_s0_uma_srand/说明.md) | 有 | 1 份 |
| `fixed_burst_n200` | [说明.md](../artifacts/experiments/fixed_burst_n200/说明.md) | — | 1 份 |
| `fixed_flow_n200` | [说明.md](../artifacts/experiments/fixed_flow_n200/说明.md) | — | 1 份 |
| `flow_n200_dqnm_tau5_s0_uma_srand` | [说明.md](../artifacts/experiments/flow_n200_dqnm_tau5_s0_uma_srand/说明.md) | 有 | 1 份 |
| `flow_n200_learned_real_tau5_s0_uma_srand` | [说明.md](../artifacts/experiments/flow_n200_learned_real_tau5_s0_uma_srand/说明.md) | 有 | 0 份 |
| `flow_n200_learned_tau5_s0_uma_srand` | [说明.md](../artifacts/experiments/flow_n200_learned_tau5_s0_uma_srand/说明.md) | 有 | 1 份 |
| `greedy_burst_n200` | [说明.md](../artifacts/experiments/greedy_burst_n200/说明.md) | — | 1 份 |
| `greedy_flow_n200` | [说明.md](../artifacts/experiments/greedy_flow_n200/说明.md) | — | 1 份 |
| `jiang_burst_n200` | [说明.md](../artifacts/experiments/jiang_burst_n200/说明.md) | — | 1 份 |
| `jiang_flow_n200` | [说明.md](../artifacts/experiments/jiang_flow_n200/说明.md) | — | 1 份 |
| `pfg_burst_n200` | [说明.md](../artifacts/experiments/pfg_burst_n200/说明.md) | — | 1 份 |
| `pfg_flow_n200` | [说明.md](../artifacts/experiments/pfg_flow_n200/说明.md) | — | 1 份 |
| `rand_burst_n200` | [说明.md](../artifacts/experiments/rand_burst_n200/说明.md) | — | 1 份 |
| `rand_flow_n200` | [说明.md](../artifacts/experiments/rand_flow_n200/说明.md) | — | 1 份 |
| `rr_burst_n200` | [说明.md](../artifacts/experiments/rr_burst_n200/说明.md) | — | 1 份 |
| `rr_flow_n200` | [说明.md](../artifacts/experiments/rr_flow_n200/说明.md) | — | 1 份 |
| `shen_burst_n200` | [说明.md](../artifacts/experiments/shen_burst_n200/说明.md) | — | 1 份 |
| `shen_flow_n200` | [说明.md](../artifacts/experiments/shen_flow_n200/说明.md) | — | 1 份 |

> 索引合计 215 格。
