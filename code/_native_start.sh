#!/bin/bash
# 四基线原生场景训练（2026-10-30 重订版）。
# 每个基线一条原生场景，场景参数逐项取自论文（config.BASELINE_SCENES）：
#   ddpga  × cheng7   Cheng TIM 2024     7 架 / 50 GU / 1000×1000 / 静态
#   mfdqn  × song361  Song TWC 2025      361 架 / 1444 GU / 19km / 静态
#   dqnm   × rizvi3   Rizvi TMLCN 2025   3 架 / 6 用户 / 1000×1000 / 用户 10 m/s
#   tmarl  × chen6    Chen T-ITS 2024    6 架 / 100 PoI / 1000×1000 / 静态
# 用法: _native_start.sh local|cloud
cd "$(dirname "$0")" || exit 1
L=../artifacts/logs/train
mkdir -p "$L"
PY="/c/Users/Aliyy Gingko/miniconda3/envs/py311/python.exe"
if [ "$1" = "cloud" ]; then PY=/usr/bin/python3; fi
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PF_CHANNEL=table

run() {  # task dxy script mode n preset log
  local task=$1 dxy=$2 script=$3 mode=$4 n=$5 preset=$6 lg=$7
  if [ -f "$L/${task}.log" ] && grep -q "原地评估完成" "$L/${task}.log"; then
    echo "[skip] $task 已完成"; return
  fi
  echo "[start] $(date '+%F %T') $task"
  PF_DXY_MAX=$dxy nohup "$PY" "$script" --mode "$mode" --scene static \
    --n-uav "$n" --preset "$preset" --eta 1.0 --total-rollouts 600 \
    --task "$task" --fresh > "$L/${task}.log" 2>&1 &
}

if [ "$1" = "cloud" ]; then
  run native_mfdqn_song361 0.5 mfdqn_arm.py mfdqn 361 song361 x
else
  run native_ddpga_cheng7  2.0 ddp_arm.py    ddpga 7   cheng7 x
  run native_dqnm_rizvi3   0.5 dqn_arm.py    dqnm  3   rizvi3 x
  run native_tmarl_chen6   2.5 t_arm.py      tmarl 6   chen6  x
fi
sleep 2
echo "[done] 已派发"
