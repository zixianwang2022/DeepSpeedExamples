#!/usr/bin/env python3
"""Measure each layer type of model.py (Embed, Block, Head) for DeepSpeed's pipeline planner and write the costs file it
reads (python -m deepspeed.runtime.pipe.planner plan --costs <file> ...).

Run it on one node with as many processes as --ep, so that the block's MoE runs over its real expert-parallel group:
the expert parameters and the activations measured are then a rank's own. The block is converted by AutoEP exactly as
in train.py; the profiler builds nothing else and does not train.

  deepspeed --num_gpus 8 profile_layers.py --model 63b --ep 8 --mbs-list 1,2 --out costs_63b_ep8.json
"""
import argparse

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.planner import profile_layer_types, write_costs

import model as M
from train import bind_device, launch_env


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='small', choices=sorted(M.SHAPES))
    ap.add_argument('--ep', type=int, default=8)
    ap.add_argument('--mbs-list', default='1,2')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--no-grouped-mm', action='store_true')
    ap.add_argument('--routing-imbalance',
                    type=float,
                    default=2.5,
                    help='the memory of the MoE block is measured with one expert-parallel rank receiving this many '
                    'times the mean rows (the busiest rank of training); the times stay those of the router\'s own '
                    'routing. 0 measures both with the router\'s own routing')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', required=True, help='the costs file to write')
    ap.add_argument('--local_rank', type=int, default=-1)
    return ap.parse_args()


MEMORY_FIELDS = ('kept_bytes', 'kept_ckpt_bytes', 'transient_bytes', 'transient_ckpt_bytes', 'fwd_extra_bytes',
                 'fwd_extra_ckpt_bytes', 'stale_bytes', 'stale_ckpt_bytes', 'kept_tensors', 'kept_ckpt_tensors')


def skew_routing(layer, ep, seed):
    """Replace the router's scores of an AutoEP layer by seeded noise with a bias on the experts of expert-parallel rank
    0, while ``state['on']``. Every rank draws the same noise, so rank 0 receives ``share x ep`` times the mean rows,
    where ``share`` is the fraction of the top-k choices that fall on its experts. AutoEP drops no token, so the memory
    of an MoE block follows the rows its rank receives, and the busiest rank of a training run receives more than the
    mean."""
    n_local, k = layer.num_local_experts, layer.top_k
    state = {'on': False, 'bias': 0.0}

    def noise(shape, device):
        g = torch.Generator(device=device).manual_seed(seed)
        return 4 * torch.randn(shape, generator=g, device=device)

    def share(bias, shape, device):
        scores = noise(shape, device)
        scores[:, :n_local] += bias
        return (scores.topk(k, dim=-1).indices < n_local).float().mean().item()

    def hook(gate, inputs, scores):
        if not state['on']:
            return scores
        z = noise(scores.shape, scores.device)
        z[..., :n_local] += state['bias']
        return scores * 0 + z.to(scores.dtype)

    layer.router.gate.register_forward_hook(hook)

    def calibrate(imbalance, tokens, device):
        """The bias with which rank 0's share of the choices is imbalance / ep; returns the imbalance reached."""
        shape, target, lo, hi = (tokens, layer.num_experts), imbalance / ep, 0.0, 32.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if share(mid, shape, device) < target else (lo, mid)
        state['bias'] = hi
        return share(hi, shape, device) * ep

    return state, calibrate


def main():
    a = parse()
    rank, world, local = launch_env()
    acc = get_accelerator()
    bind_device(acc, local)
    deepspeed.init_distributed(dist_backend=acc.communication_backend_name())
    if world != a.ep:
        raise SystemExit(f'profile with one expert-parallel group: launch {a.ep} processes, not {world}')
    c = M.SHAPES[a.model]
    one = M.Shape(**{**vars(c), 'layers': 1})
    dtype = torch.float32 if acc.device_name() == 'cpu' else torch.bfloat16

    # one block, converted by AutoEP over the real expert-parallel group, as train.py converts every block
    model = M.Decoder(one)
    shapes = M.full_shapes(model)
    # An fp32 engine without an optimizer only converts and places the layers (a bf16 engine would build an optimizer
    # wrapper that needs MoE parameter groups); the layers are cast to the training dtype after initialization.
    cfg = {
        'train_micro_batch_size_per_gpu': 1,
        'expert_parallel': M.expert_parallel_config(one, a.ep, pipeline=False, grouped_mm=not a.no_grouped_mm),
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=cfg)
    M.initialize_parameters(engine.module, a.seed, shapes, c.layers)
    engine.module.to(dtype)
    embed, block, head = engine.module.layers
    dev = engine.device

    def make_input(kind, b):
        if kind == 'Embed':
            return torch.randint(0, c.vocab, (b, c.seq), device=dev)
        return torch.randn(b, c.seq, c.hidden, device=dev, dtype=dtype, requires_grad=True)

    def head_loss(out):
        return M.lm_loss(out, torch.randint(0, c.vocab, out.shape[:2], device=dev))

    # the GEMM workspaces: the first matrix product of the forward and of the autograd thread each leave one allocated
    workspace = 0
    if acc.device_name() != 'cpu':
        acc.synchronize()
        w0 = acc.memory_allocated()
        lin = torch.nn.Linear(c.hidden, c.hidden, bias=False).to(dev, dtype)
        lin(torch.randn(8, c.hidden, device=dev, dtype=dtype, requires_grad=True)).sum().backward()
        del lin
        acc.synchronize()
        workspace = acc.memory_allocated() - w0

    def profile(kinds):
        factories = {'Embed': lambda: embed, 'Block': lambda: block, 'Head': lambda: head}
        return profile_layer_types({k: factories[k] for k in kinds},
                                   make_input, [int(v) for v in a.mbs_list.split(',')],
                                   reps=a.reps,
                                   warmup=a.warmup,
                                   loss_fns={'Head': head_loss},
                                   recompute={'Block', 'Head'},
                                   group=dist.get_world_group())

    rows = profile(['Embed', 'Block', 'Head'])
    reached = 0.0
    if a.routing_imbalance > 0 and a.ep > 1:
        # the block again with its busiest rank receiving routing_imbalance times the mean rows: its memory fields, the
        # larger of the two measurements, describe the busiest rank; its times stay those of the router's own routing
        state, calibrate = skew_routing(block.mlp, a.ep, a.seed)
        reached = calibrate(a.routing_imbalance, c.seq, dev)
        state['on'] = True
        skewed = profile(['Block'])['Block']['by_mbs']
        state['on'] = False
        for mbs, row in rows['Block']['by_mbs'].items():
            for field in MEMORY_FIELDS:
                if field in row:
                    row[field] = max(row[field], skewed[mbs][field])
    if rank == 0:
        key_fields = {'device': acc.device_name(), 'device_total_bytes': acc.total_memory(local), 'dtype': str(dtype),
                      'torch': torch.__version__, 'deepspeed': deepspeed.__version__, 'shape': vars(c), 'ep': a.ep,
                      'tp': 1, 'memory_routing_imbalance': round(reached, 3)}
        key = write_costs(a.out, rows, key_fields, blas_workspace_bytes=workspace)
        print(f'WROTE {a.out} (key {key}); block memory measured at a routing imbalance of {reached:.2f}', flush=True)


if __name__ == '__main__':
    main()
