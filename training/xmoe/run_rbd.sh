#!/bin/bash
# X-MoE's redundancy-bypassing dispatch (RBD) against AutoEP's default all-to-all: the same model, weights and batches,
# with comm_backend "comm", then "two_level", which sends a token across the network once per destination node and
# copies it to the node's other experts inside the node. It removes rows only when an expert-parallel group spans
# several nodes, so EP must be larger than the GPUs of a node.
#   bash run_rbd.sh                                         small model, 2 nodes of 8 GPUs, EP 16
#   MODEL=63b EP=64 RECOMPUTE=all LAUNCH="srun -N8 ... python -u" bash run_rbd.sh   63B model, 8 nodes, EP 64
#   ROUTING=even ...                                        the routers' scores replaced by noise (see train.py)
#   RECOMPUTE=all ...                                       every block recomputed in both runs (needed for 63b, 173b)
: "${LAUNCH:=deepspeed --num_nodes 2 --num_gpus 8}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EP="${EP:-16}"
MBS="${MBS:-1}"
RECOMPUTE="${RECOMPUTE:-none}"
[ "${ROUTING:-own}" = even ] && EXTRA="--even-routing $EXTRA"
train rbd_comm --ep "$EP" --mbs "$MBS" --recompute "$RECOMPUTE" "$@"
train rbd_two_level --ep "$EP" --mbs "$MBS" --recompute "$RECOMPUTE" --rbd "$@"
compare rbd_comm rbd_two_level
