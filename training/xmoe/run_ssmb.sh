#!/bin/bash
# X-MoE's sequence-sharded MoE block (SSMB) under tensor parallelism. Three runs of the same model, weights and batches,
# TP 2:
#   ssmb_folding            AutoEP + AutoTP folding: every TP peer routes all tokens and keeps 1/TP of the routed rows
#   ssmb_folding_recompute  the same with every block recomputed, the usual way to fit when memory is short
#   ssmb                    sequence sharding: every TP peer routes, sends and combines 1/TP of the tokens
# RECOMPUTE (none | all | N, see train.py --recompute) applies to ssmb_folding and ssmb; with RECOMPUTE=all, as a model
# that needs it at pipeline size 1 (63b, 173b), ssmb_folding_recompute would repeat ssmb_folding and is left out.
#   bash run_ssmb.sh                                          small model, 1 node of 8 GPUs, TP 2 x EP 4
#   MODEL=63b TP=2 EP=32 RECOMPUTE=all LAUNCH="srun -N8 ... python -u" bash run_ssmb.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TP="${TP:-2}"
EP="${EP:-4}"
MBS="${MBS:-1}"
RECOMPUTE="${RECOMPUTE:-none}"
[ "${ROUTING:-own}" = even ] && EXTRA="--even-routing $EXTRA"
train ssmb_folding --tp "$TP" --ep "$EP" --mbs "$MBS" --recompute "$RECOMPUTE" "$@"
[ "$RECOMPUTE" = all ] || train ssmb_folding_recompute --tp "$TP" --ep "$EP" --mbs "$MBS" --recompute all "$@"
train ssmb --tp "$TP" --ep "$EP" --mbs "$MBS" --recompute "$RECOMPUTE" --ssmb "$@"
compare ssmb_folding ssmb
[ "$RECOMPUTE" = all ] || compare ssmb_folding_recompute ssmb
