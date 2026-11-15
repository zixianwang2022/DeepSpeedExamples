#!/usr/bin/env python3
"""Train the decoder of model.py with DeepSpeed, with or without each X-MoE / X-MoE-4D feature, and log every step.

Layouts: --pp pipeline stages, --tp tensor parallelism (AutoTP), --ep expert parallelism (AutoEP); the rest of the GPUs
is data parallelism. Features:
  --rbd             X-MoE's redundancy-bypassing dispatch: AutoEP comm_backend "two_level" (pipeline size 1)
  --ssmb            X-MoE's sequence-sharded MoE block: expert_parallel.sequence_sharded_moe (needs --tp > 1)
  --plan FILE       X-MoE-4D's pipeline planner: the layers and recomputed blocks of every stage, from the "BEST" line
                    that python -m deepspeed.runtime.pipe.planner prints (or --layers-per-stage, --recompute-per-stage)
Baselines: the same command without the feature. --recompute sets the activation checkpointing of a baseline: 'none',
'all' (every block), or a number N (pipeline: the first N blocks of every stage of the even split, the only partial
recompute upstream DeepSpeed offers without per-stage counts; pipeline size 1: the first N blocks).

Routing: --even-routing replaces every router's scores by seeded noise, so tokens spread evenly over the experts (an
untrained router sends most tokens to a few experts). It is a measurement condition, stated with every number; the
busiest expert-parallel rank's rows over the mean are logged either way. No router auxiliary loss is added: the
examples measure speed and memory, and the loss is the language-model loss.

Every step is a row of --metrics-out (CSV, rank 0): step, loss, seconds, model TFLOP/s per GPU (model.flops_per_token,
recomputation not counted), the largest allocated and reserved memory of any rank, allocator retries, and the busiest
expert-parallel rank's rows over the mean.

Launch one process per GPU: deepspeed --num_gpus 8 train.py ..., torchrun, or srun (RANK, WORLD_SIZE and LOCAL_RANK are
then taken from SLURM_PROCID, SLURM_NTASKS and SLURM_LOCALID; MASTER_ADDR and MASTER_PORT must be set).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

import model as M


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='small', choices=sorted(M.SHAPES))
    ap.add_argument('--layers', type=int, default=0, help='override the number of blocks of --model')
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--ep', type=int, default=8)
    ap.add_argument('--mbs', type=int, default=1, help='micro-batch size')
    ap.add_argument('--global-batch', type=int, default=0, help='sequences per step (default: one micro-batch per '
                    'data-parallel rank and 2 x pp micro-batches under pipeline parallelism)')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5, help='steps left out of the summary (logged all the same)')
    ap.add_argument('--zero', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-warmup', type=int, default=0, help='steps of linear learning-rate warm-up (0: none)')
    ap.add_argument('--clip-grad', type=float, default=1.0)
    ap.add_argument('--rbd', action='store_true')
    ap.add_argument('--ssmb', action='store_true')
    ap.add_argument('--plan', default='', help='a JSON file, or the BEST line, printed by the planner')
    ap.add_argument('--layers-per-stage', default='')
    ap.add_argument('--recompute-per-stage', default='')
    ap.add_argument('--recompute', default='none', help="'none', 'all' or a number of blocks (see the docstring)")
    ap.add_argument('--even-routing', action='store_true')
    ap.add_argument('--no-grouped-mm', action='store_true', help="AutoEP's sequential expert loop (AMD GPUs)")
    ap.add_argument('--data-path', default='', help='flat file of token ids (e.g. Megatron-LM .bin); random if empty')
    ap.add_argument('--data-dtype', default='uint16')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--metrics-out', default='')
    ap.add_argument('--local_rank', type=int, default=-1, help='set by the deepspeed launcher; LOCAL_RANK is used')
    a = ap.parse_args(argv)
    if a.rbd and a.pp > 1:
        ap.error('--rbd runs with --pp 1 (DeepSpeed refuses comm_backend "two_level" under pipeline parallelism)')
    if a.ssmb and a.tp < 2:
        ap.error('--ssmb needs --tp 2 or more: it shards the tokens of the MoE block over the tensor-parallel peers')
    if a.even_routing and a.pp > 1:
        ap.error('--even-routing runs with --pp 1')
    if (a.plan or a.layers_per_stage) and a.pp < 2:
        ap.error('--plan and --layers-per-stage need --pp 2 or more')
    if a.recompute not in ('none', 'all') and not a.recompute.isdigit():
        ap.error("--recompute takes 'none', 'all' or a number of blocks")
    if a.recompute.isdigit() and a.pp > 1 and (a.plan or a.layers_per_stage or a.recompute_per_stage):
        ap.error('--recompute N recomputes N blocks of every stage of the even split; with --plan or per-stage lists '
                 'give the recomputed blocks per stage instead')
    return a


def launch_env():
    env = os.environ
    if 'RANK' not in env and 'SLURM_PROCID' in env:
        env.update(RANK=env['SLURM_PROCID'], WORLD_SIZE=env['SLURM_NTASKS'], LOCAL_RANK=env['SLURM_LOCALID'])
    return int(env['RANK']), int(env['WORLD_SIZE']), int(env['LOCAL_RANK'])


def bind_device(acc, local):
    """Select this process's GPU. A launcher may show each process all of a node's GPUs (deepspeed, torchrun) or one
    (srun --gpus-per-task=1); DeepSpeed and PipelineModule place tensors on device LOCAL_RANK, so with one visible GPU
    LOCAL_RANK becomes 0, the index of that GPU."""
    device = local % max(1, acc.device_count())
    os.environ['LOCAL_RANK'] = str(device)
    acc.set_device(device)


def ints(s):
    return [int(x) for x in s.split(',')] if s else None


def read_plan(plan):
    """The planner prints BEST {"layers_per_stage": [...], "activation_checkpoints_per_stage": [...], ...}."""
    text = open(plan).read() if os.path.exists(plan) else plan
    text = text[text.index('{'):] if 'BEST' in text else text
    p = json.loads(text)
    return p['layers_per_stage'], p['activation_checkpoints_per_stage'], p['b']


# The micro-batch being computed, for --even-routing: the recompute of a block in the backward must route as its forward.
ROUTING_CLOCK = {'micro_batch': 0}


def spread_routing_evenly(module, seed, dp_rank):
    """Replace the scores of every AutoEP router by noise that depends on the layer, the data-parallel rank and the
    micro-batch only; tensor-parallel peers of one data-parallel rank get the same noise. The router still runs."""
    layers = [m for m in module.modules() if getattr(m, '_is_autoep_layer', False)]
    for index, layer in enumerate(layers):

        def noise(gate, inputs, scores, index=index):
            g = torch.Generator(device=scores.device)
            g.manual_seed(seed + 1_000_003 * dp_rank + 10_007 * index + 101 * ROUTING_CLOCK['micro_batch'])
            return scores * 0 + (4 * torch.randn(scores.shape, generator=g, device=scores.device)).to(scores.dtype)

        layer.router.gate.register_forward_hook(noise)


def busiest_expert_rank(module):
    """Rows of the busiest rank of an expert-parallel group over the mean, since the first step, the largest over the MoE
    layers of this rank and over the ranks. 1.0 is even."""
    worst = torch.ones(1)
    for layer in module.modules():
        if not getattr(layer, '_is_autoep_layer', False) or layer.ep_size <= 1:
            continue
        rows = layer.tokens_per_expert.detach().clone().float()  # on the device: the collective needs it there
        dist.all_reduce(rows, group=layer.ep_group)
        per_rank = rows.view(layer.ep_size, -1).sum(1)
        if per_rank.sum() > 0:
            worst = torch.maximum(worst, (per_rank.max() / per_rank.mean()).cpu().view(1))
    return worst


def build(a, c, rank):
    """The model (a plain module, or a PipelineModule) and its DeepSpeed config."""
    pipeline = a.pp > 1
    recompute = a.recompute
    if pipeline:
        from deepspeed.pipe import LayerSpec, PipelineModule
        layers_per_stage = ints(a.layers_per_stage)
        ckpts = ints(a.recompute_per_stage)
        if a.plan:
            layers_per_stage, ckpts, _ = read_plan(a.plan)
        per_class = recompute not in ('none', 'all') and ckpts is None
        span = c.layers // a.pp
        blocks = [
            LayerSpec(M.BlockRecomputed if per_class and i % span < int(recompute) else M.Block, c)
            for i in range(c.layers)
        ]
        kw = dict(layers=[LayerSpec(M.Embed, c)] + blocks + [LayerSpec(M.Head, c)],
                  num_stages=a.pp,
                  loss_fn=M.lm_loss,
                  partition_method='type:Block',
                  layers_per_stage=layers_per_stage,
                  activation_checkpoints_per_stage=ckpts)
        if ckpts is not None or recompute == 'all':
            kw.update(activation_checkpoint_interval=1, checkpointable_layers=['Block', 'BlockRecomputed'])
        elif per_class:
            kw.update(activation_checkpoint_interval=1, checkpointable_layers=['BlockRecomputed'])
        model = PipelineModule(**kw)
    else:
        n = {'none': 0, 'all': c.layers}.get(recompute)
        model = M.Decoder(c, recompute=int(recompute) if n is None else n)
    dp = dist.get_world_size() // (a.pp * a.tp)
    gb = a.global_batch or a.mbs * dp * (2 * a.pp if pipeline else 1)
    if gb % (a.mbs * dp):
        raise ValueError(f'--global-batch {gb} must be a multiple of mbs x data-parallel size = {a.mbs * dp}')
    cfg = {
        'train_micro_batch_size_per_gpu': a.mbs,
        'gradient_accumulation_steps': gb // (a.mbs * dp),
        'bf16': {'enabled': get_accelerator().device_name() != 'cpu'},
        'zero_optimization': {'stage': a.zero},
        'optimizer': {'type': 'Adam', 'params': {'lr': a.lr}},
        'gradient_clipping': a.clip_grad,
        **({
            'scheduler': {
                'type': 'WarmupLR',
                'params': {
                    'warmup_min_lr': 0.0,
                    'warmup_max_lr': a.lr,
                    'warmup_num_steps': a.lr_warmup,
                    'warmup_type': 'linear'
                }
            }
        } if a.lr_warmup else {}),
        'expert_parallel': M.expert_parallel_config(c, a.ep, pipeline, a.rbd, a.ssmb, not a.no_grouped_mm),
        'steps_per_print': 10**9,
    }
    if get_accelerator().device_name() == 'cpu':
        cfg['optimizer']['params']['torch_adam'] = True
    if a.tp > 1:
        cfg['tensor_parallel'] = M.tensor_parallel_config(a.tp)
    return model, cfg, gb, dp


def main(argv=None):
    a = parse(argv)
    rank, world, local = launch_env()
    acc = get_accelerator()
    bind_device(acc, local)
    deepspeed.init_distributed(dist_backend=acc.communication_backend_name())
    c = M.SHAPES[a.model]
    if a.layers:
        c = M.Shape(**{**vars(c), 'layers': a.layers})
    if a.recompute.isdigit() and a.pp > 1 and c.layers % a.pp:
        raise SystemExit(f'--recompute {a.recompute} needs the even split: {c.layers} blocks over {a.pp} stages')
    if a.plan:
        a.mbs = read_plan(a.plan)[2]  # the micro-batch size the planner chose
    torch.manual_seed(a.seed)
    model, cfg, gb, dp = build(a, c, rank)
    shapes = M.full_shapes(model)
    engine, _, _, _ = deepspeed.initialize(model=model, config=cfg)
    from deepspeed.utils import groups
    tp_rank = groups.get_tensor_model_parallel_rank() if a.tp > 1 else 0
    M.initialize_parameters(engine.module, a.seed, shapes, c.layers, tp_rank=tp_rank, tp_size=a.tp)
    if hasattr(engine.optimizer, 'refresh_fp32_params'):
        engine.optimizer.refresh_fp32_params()  # the optimizer copied the weights before they were initialized
    dp_rank = engine.grid.get_data_parallel_rank() if a.pp > 1 else rank // a.tp
    if a.even_routing:
        spread_routing_evenly(engine.module, a.seed, dp_rank)

    g = torch.Generator().manual_seed(a.seed + dp_rank)  # same samples on a data-parallel rank's stages and TP peers
    tokens = np.memmap(a.data_path, dtype=np.dtype(a.data_dtype), mode='r') if a.data_path else None

    def batches():
        while True:
            if tokens is None:
                ids = torch.randint(0, c.vocab, (a.mbs, c.seq + 1), generator=g)
            else:
                starts = torch.randint(0, len(tokens) - c.seq - 1, (a.mbs, ), generator=g).tolist()
                ids = torch.from_numpy(np.stack([tokens[i:i + c.seq + 1] for i in starts]).astype(np.int64))
            yield ids[:, :-1].contiguous().to(engine.device), ids[:, 1:].contiguous().to(engine.device)

    it = batches()
    accum = gb // (a.mbs * dp)
    flops = M.flops_per_token(c) * gb * c.seq
    rows = []
    if rank == 0:
        print(f'model {a.model} ({M.parameter_count(c) / 1e9:.1f}B parameters), world {world}: pp {a.pp} x tp {a.tp} '
              f'x dp {dp}, ep {a.ep}; {gb} sequences of {c.seq} per step ({accum} micro-batches of {a.mbs}); '
              f'rbd {a.rbd} ssmb {a.ssmb} plan {a.plan or a.layers_per_stage or "-"} recompute {a.recompute} '
              f'even routing {a.even_routing}',
              flush=True)
    for step in range(a.steps):
        acc.synchronize()
        t0 = time.time()
        if a.pp > 1:
            loss = engine.train_batch(data_iter=it).float()
        else:
            total = torch.zeros((), device=engine.device)
            for _ in range(accum):
                ids, labels = next(it)
                ROUTING_CLOCK['micro_batch'] += 1
                micro_loss = M.lm_loss(engine(ids), labels)
                engine.backward(micro_loss)
                engine.step()
                total += micro_loss.detach().float() / accum
            loss = total
            dist.all_reduce(loss)
            loss = loss / world  # every TP peer holds its data-parallel rank's loss
        acc.synchronize()
        seconds = time.time() - t0
        stats = torch.tensor([
            acc.max_memory_allocated() / 2**30, acc.max_memory_reserved() / 2**30,
            float(acc.memory_stats().get('num_alloc_retries', 0)) if acc.device_name() != 'cpu' else 0.0
        ],
                             device=engine.device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        busiest = busiest_expert_rank(engine.module).to(engine.device)
        dist.all_reduce(busiest, op=dist.ReduceOp.MAX)
        row = dict(step=step,
                   loss=round(float(loss), 6),
                   seconds=round(seconds, 4),
                   tflops_per_gpu=float(f'{flops / seconds / world / 1e12:.6g}'),
                   max_allocated_gib=round(float(stats[0]), 3),
                   max_reserved_gib=round(float(stats[1]), 3),
                   alloc_retries=int(stats[2]),
                   busiest_expert_rank=round(float(busiest), 3))
        rows.append(row)
        if rank == 0:
            print(' '.join(f'{k} {v}' for k, v in row.items()), flush=True)
    # Per pipeline stage: the largest memory any of its ranks allocated and reserved, and allocator retries, to set
    # against the planner's prediction of each stage.
    stage = engine.stage_id if a.pp > 1 else 0
    mine = torch.tensor([float(stage), acc.max_memory_allocated() / 2**30, acc.max_memory_reserved() / 2**30,
                         float(acc.memory_stats().get('num_alloc_retries', 0)) if acc.device_name() != 'cpu' else 0.0],
                        device=engine.device)
    every = [torch.zeros_like(mine) for _ in range(world)]
    dist.all_gather(every, mine)
    if rank == 0:
        for s_ in range(a.pp):
            got = [t.tolist() for t in every if int(t[0]) == s_]
            print(f'STAGE {s_}: max allocated {max(g[1] for g in got):.2f} GiB, max reserved '
                  f'{max(g[2] for g in got):.2f} GiB, allocator retries {int(max(g[3] for g in got))} '
                  f'({len(got)} ranks)',
                  flush=True)
    if rank == 0:
        timed = rows[a.warmup:] or rows
        mean = sum(r['tflops_per_gpu'] for r in timed) / len(timed)
        print(f'SUMMARY model TFLOP/s per GPU {mean:.2f} over steps {timed[0]["step"]}-{timed[-1]["step"]}, '
              f'max allocated {max(r["max_allocated_gib"] for r in rows):.2f} GiB, '
              f'max reserved {max(r["max_reserved_gib"] for r in rows):.2f} GiB, '
              f'allocator retries {rows[-1]["alloc_retries"]}, last loss {rows[-1]["loss"]}',
              flush=True)
        if a.metrics_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.metrics_out)), exist_ok=True)
            with open(a.metrics_out, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == '__main__':
    main()
