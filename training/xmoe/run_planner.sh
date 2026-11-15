#!/bin/bash
# X-MoE-4D's pipeline planner against what DeepSpeed offers without it, X-MoE-4D's layout: expert parallelism inside a
# node, pipeline stages across nodes.
#   1. profile_layers.py measures the embedding, a block and the output layer on one node (EP GPUs);
#   2. the planner chooses the blocks and the recomputed blocks of every stage (and the micro-batch size);
#   3. three runs of the same model, weights and batches:
#        planner_even_all   the even split, every block recomputed (activation_checkpoint_interval=1)
#        planner_even_N     the even split, the first N blocks of every stage recomputed (by class name, the partial
#                           recompute upstream DeepSpeed offers), N the smallest the planner's memory model accepts;
#                           skipped when N is every block of a stage, which is planner_even_all
#        planner_plan       the planner's split
#   bash run_planner.sh                                              small model, 2 nodes, PP 2 x EP 8
#   MODEL=63b PP=4 NODES=4 LAUNCH="srun -N4 ... python -u" PROFILE_LAUNCH="srun -N1 -n8 ... python -u" bash run_planner.sh
#   MODEL=173b PP=8 NODES=16 GLOBAL_BATCH=2048 bash run_planner.sh  (X-MoE-4D's 173B layout: PP 8 x EP 8, expert DP 2)
# ROUTING_IMBALANCE (default 2.5): the profile measures a block's memory with one expert-parallel rank receiving this many
#   times the mean rows. A smaller value lets the planner recompute fewer blocks and leaves less memory margin.
: "${NODES:=2}" "${GPUS_PER_NODE:=8}"
: "${LAUNCH:=deepspeed --num_nodes $NODES --num_gpus $GPUS_PER_NODE}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PP="${PP:-2}"
EP="${EP:-8}"
WORLD=$((NODES * GPUS_PER_NODE))
GLOBAL_BATCH="${GLOBAL_BATCH:-$((WORLD * 4))}"  # 4 x PP micro-batches of 1 per data-parallel rank
MBS_LIST="${MBS_LIST:-1,2}"
DEVICE_GIB="${DEVICE_GIB:-64}"
ROUTING_IMBALANCE="${ROUTING_IMBALANCE:-2.5}"
PROFILE_LAUNCH="${PROFILE_LAUNCH:-deepspeed --num_gpus $EP}"
LAYERS=$(python -c "import sys; sys.path.insert(0, '$HERE'); import model; print(model.SHAPES['$MODEL'].layers)")
COSTS="$OUT/costs_${MODEL}_ep${EP}_mbs${MBS_LIST//,/-}_imb${ROUTING_IMBALANCE}.json"
PLANNER=(python -m deepspeed.runtime.pipe.planner)
ARGS=(--costs "$COSTS" --layers "Embed,Block*$LAYERS,Head" --blocks Block --world "$WORLD" --pp "$PP" --ep "$EP"
      --global-batch "$GLOBAL_BATCH" --device-gib "$DEVICE_GIB")

if [ ! -f "$COSTS" ]; then
    echo "== profile -> $COSTS"
    # shellcheck disable=SC2086
    $PROFILE_LAUNCH "$HERE/profile_layers.py" --model "$MODEL" --ep "$EP" --mbs-list "$MBS_LIST" \
        --routing-imbalance "$ROUTING_IMBALANCE" --out "$COSTS" $EXTRA
fi
"${PLANNER[@]}" plan "${ARGS[@]}" | tee "$OUT/plan.txt"
grep '^BEST {' "$OUT/plan.txt" > "$OUT/best.txt" || { echo "the planner found no split that fits" >&2; exit 1; }
MBS=$(python -c "import json; print(json.loads(open('$OUT/best.txt').read()[5:])['b'])")

# the smallest number of recomputed blocks per stage with which the even split fits
N=""
for n in $(seq 0 $((LAYERS / PP))); do
    # a failed prediction stops the script; it must not read as "fits"
    prediction=$("${PLANNER[@]}" predict "${ARGS[@]}" --mbs-list "$MBS" --ckpts "$(python -c "print(','.join(['$n'] * $PP))")")
    if ! grep -q "DOES NOT FIT" <<< "$prediction"; then
        N=$n
        break
    fi
done
[ -n "$N" ] || { echo "the even split does not fit even with every block recomputed" >&2; exit 1; }

train planner_even_all --pp "$PP" --ep "$EP" --mbs "$MBS" --global-batch "$GLOBAL_BATCH" --recompute all
if [ "$N" -lt $((LAYERS / PP)) ]; then  # with N = every block, planner_even_N would repeat planner_even_all
    train "planner_even_$N" --pp "$PP" --ep "$EP" --mbs "$MBS" --global-batch "$GLOBAL_BATCH" --recompute "$N"
fi
train planner_plan --pp "$PP" --ep "$EP" --global-batch "$GLOBAL_BATCH" --plan "$OUT/best.txt"
compare planner_even_all planner_plan
if [ "$N" -lt $((LAYERS / PP)) ]; then
    compare "planner_even_$N" planner_plan
fi
