# Shared by run_rbd.sh, run_ssmb.sh and run_planner.sh: one training run, and the comparison of two runs.
#
# MODEL   tiny | small (default) | 63b | 173b, the shapes of model.py
# LAUNCH  starts one process per GPU: "deepspeed --num_gpus 8" by default; on a Slurm cluster, for example
#           LAUNCH="srun -N4 -n32 --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest python -u"
#         (the launcher is followed by the script: end an srun command with the interpreter; train.py reads
#         SLURM_PROCID, SLURM_NTASKS and SLURM_LOCALID; export MASTER_ADDR and MASTER_PORT)
# STEPS   steps per run (30); WARMUP steps left out of the comparison (10)
# EXTRA   more train.py options for every run, e.g. --no-grouped-mm on AMD GPUs, --data-path tokens.bin
# OUT     where the logs and the per-step CSV files go (./runs)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/runs}"
MODEL="${MODEL:-small}"
LAUNCH="${LAUNCH:-deepspeed --num_gpus 8}"
STEPS="${STEPS:-30}"
WARMUP="${WARMUP:-10}"
EXTRA="${EXTRA:-}"
mkdir -p "$OUT"

train() { # <run name> <train.py options...>
    local name="$1"
    shift
    echo "== $name: train.py --model $MODEL $* $EXTRA"
    set +e
    # shellcheck disable=SC2086
    $LAUNCH "$HERE/train.py" --model "$MODEL" --steps "$STEPS" --warmup "$WARMUP" \
        --metrics-out "$OUT/$name.csv" $EXTRA "$@" 2>&1 | tee "$OUT/$name.log" | grep -E "^(model|SUMMARY)|Error"
    local status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -ne 0 ]; then
        echo "run $name failed with exit code $status; see $OUT/$name.log" >&2
        exit "$status"
    fi
}

compare() { # <baseline run> <feature run>
    python "$HERE/compare.py" "$OUT/$1.csv" "$OUT/$2.csv" --skip "$WARMUP"
}
