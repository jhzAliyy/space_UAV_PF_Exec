#!/bin/bash
# 本地基线批量（2026-10-30）：ddpga / mfdqn × n12 / n30 × flow / burst = 8 条
# 每进程 2 线程，8 × 2 = 16 线程，本地 16 核满载。
# 幂等：已完成的跳过，已在跑的跳过，被杀过的断点恢复。
# 用法: bash _local_run_baselines.sh

cd "$(dirname "$0")" || exit 1
PY="${PY:-/c/Users/Aliyy Gingko/miniconda3/envs/py311/python.exe}"
LOGDIR="../artifacts/logs/train"
PIDF="../artifacts/logs/ops/local_pids.tsv"
mkdir -p "$LOGDIR" "$(dirname "$PIDF")" ../artifacts/ckpt

alive_pid() {  # 键 → 存活的 PID（无则空）
  [ -f "$PIDF" ] || return 0
  local p
  p=$(awk -F'\t' -v k="$1" '$1==k{print $2}' "$PIDF" | tail -1)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

run_one() {
  local scene=$1 n=$2 mode=$3 ch=$4
  local key="${scene}_n${n}_${mode}"
  local nn=""; [ "$n" != "12" ] && nn="_n$n"
  local lf="$LOGDIR/u8_${scene}${nn}_${mode}_tau5_s0_${ch}_srand.log"

  if [ -f "$lf" ] && grep -qE "/1200|早停" "$lf"; then
    echo "[skip-done] $key"
    return
  fi
  local p
  p=$(alive_pid "$key")
  if [ -n "$p" ]; then
    echo "[skip-run] $key pid=$p"
    return
  fi

  case "$mode" in
    ddpga) SCRIPT="ddp_arm.py" ;;
    mfdqn) SCRIPT="mfdqn_arm.py" ;;
    *)     echo "[skip] 未知 mode=$mode"; return ;;
  esac

  echo "[start] $(date '+%F %T') TRAIN $scene n=$n $mode $ch sigma=random seed=0" > "$lf"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    PF_CHANNEL=$ch PF_SHADOW_DB=random PF_SUFFIX="${ch}_srand" \
    nohup "$PY" "$SCRIPT" --mode "$mode" --tau 5 --seed 0 \
      --scene "$scene" --n-uav "$n" --reward u8 --eta 1.0 \
      --total-rollouts 1200 >> "$lf" 2>&1 &
  local pid=$!
  printf '%s\t%s\n' "$key" "$pid" >> "$PIDF"
  echo "[launched] $(date '+%F %T') $key pid=$pid script=$SCRIPT"
  sleep 1
}

for scene in flow burst; do
  for n in 12 30; do
    run_one "$scene" "$n" ddpga umi
    run_one "$scene" "$n" mfdqn umi
  done
done

echo "[done] $(date '+%F %T') 全部 8 条已派发"
