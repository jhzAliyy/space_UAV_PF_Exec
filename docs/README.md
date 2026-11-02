# space4_real · 项目结构

无人机群通信中继：**势场学习版**（场出方向 + 神经网络执行器出模长）对比外部
同类与相邻问题基线。规模轴 n12 → n30 → n60 → n100 → n200，两场景 flow/burst，
种子 0。

## 基本设计

**场只输出方向，执行器决定模长。**

- 场（field.py）：从滞后 0.5 s 的需求快照闭式算出最优点，输出**纯方向**——
  标量维（带宽/价格/功率）是符号 ±1，位置维是平面单位矢量。零参数、
  零力度、零速度。
- 执行器（executors/，拔插）：拿到场输出包后处理再输出动作。
  - `max`：模长恒 1（真纯势场——速度恒最大，没有任何其他东西）
  - `net`：神经网络出模长 ∈[0,1]×物理步幅（+ 位置维偏转角）
  - `tuned`：2026-10 调参律遗迹（对照专用，常数显式化）
- 训练只有从零模式（无续跑代码路径）。
- 全部超参走命令行，一个参数换算法/执行器/口径。

## 目录

```
code/
  run.py            统一入口（唯一命令行，13 臂注册表）
  evaluate.py       统一评估（场族同一条链；外部基线/免训基线同列表分派）
  field.py          方向决定器（唯一闭式分支；FieldReading 输出包）
  executors/        执行器注册表（base/max/tuned/net，一行注册即插）
  algorithms/       算法注册表（field/tuned/learned=场+执行器预绑；
                    ddpga/dqnm/mfdqn/tmarl=外部基线；
                    rand/fixed/greedy/rr/shen/jiang=免训基线）
  trainer.py        TD3 管线（连续步流 256、探索噪声 0.3、梯度裁剪、
                    旧早停口径、isfinite 护栏、drain_stats）
  uav_env.py        环境（物理/奖励/关联；STEP_* 为环境物理属性）
  config.py         环境与奖励参数（执行器零参数；调优遗迹两常数在尾注）
  channel_env.py    环境构造唯一入口
  heuristics.py     免训六基线控制器
  pf_channel.py     解析信道标定（UMa/UMi 3GPP 36.777 回归）
  dqnm.py ddpga.py mfdqn.py tmarl.py    外部基线四臂（旧实现逐位沿用）
  pfmath.py         观测映射数学
  fastlink3.c / crowd_sfm.cpp / crowd_sfm.dll / fastlink3.dll
env_data/channel/   信道表（n12~n200 与原生场景）
deps/               MuJoCo-drones-gym、lightsfm
paper_work/         论文层（理论篇/工程篇/叙事/解耦定理/方案/核验，直接复制）
ref/                文献库（PDF、36.777 标准原件、提取文本）
ext/                外部对照实现（uav_bs_ctrl：drqn / madrqn）
ops/                云端通道与运维脚本（连接凭据在本目录，不入文档）
artifacts/
  experiments/<格子>/     实验格：ckpt/ + eval/ + 说明.md
  lists/ lists_old/       评估清单（新/旧）
  logs/                   运行日志
results/curves/     评估逐场曲线（评估器写）
docs/               01 数学 · 02 势场建模 · 03 实验（全格状态总表）·
                    04 基线 · 05 总览 · 06 代码结构 · 致命错误文档
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
- 输出 6 维/机：m∈(0,1)^5（sigmoid）+ φ∈(−π,π)（tanh×π）。
- 合成：a = m⊙U⊙s（xy 旋转 φ）。纯乘法恒在物理界内，无 clip。
- 起步偏置 = 调优遗迹等效模长（logit 域，可学习初值，非常数）。
- critic 逐机（奖励是逐机 (N,)，环境事实），评执行域动作 (m,φ)。

## 归档纪律

- 每个训练实验格子：`experiments/<algo>_<scene>_n<n>[_<preset>]/ckpt/`
  （latest.pt / train_log.csv / progress.json；snaps 不入归档）。
- 该格评估行：同目录 `eval/<格>.csv`（三档 2000/3000/4000）。
- 该格身份：同目录 `说明.md`（算法/口径/轮数/来源/状态/时间）。
- **三件齐备且 03 文档状态同步，该格才算闭环**；文件到位不算完成。
- 旧项目学习执行器产物一律隔离不转移；本目录已转格全部为非学习类
  （外部基线/免训基线/对照）。
