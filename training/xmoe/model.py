"""The decoder that every script of this folder trains, and the DeepSpeed settings that turn on each feature.

Model: token embedding -> L blocks (pre-norm causal attention, then a pre-norm mixture of experts) -> final norm and
output layer -> cross entropy. The MoE block has the two children AutoEP looks for, a router ``gate`` and ``experts``
whose weights are two 3-D tensors (``gate_up_proj`` [E, 2F, H] and ``down_proj`` [E, H, F], SwiGLU experts), so
DeepSpeed's AutoEP converts it with a ``moe_layer_pattern`` and runs expert parallelism, the two-level exchange (RBD)
and the sequence-sharded block (SSMB) on it. The attention has separate q, k, v and o projections, so AutoTP shards it
with four regex rules; the attention takes its number of heads from the width of its projection, so a column shard
keeps its own heads. The same layers form a plain module (pipeline size 1) or a PipelineModule (pipeline size > 1), on
which the pipeline planner sets the layers and the recomputed blocks of every stage.

The shapes are those of X-MoE-4D's model registry (examples_xmoe_4d/utils/model_registry.py). X-MoE's experts have two
matrices (H x FFN and FFN x H); AutoEP's experts are gated and have three, so FFN_SwiGLU = 2 x FFN / 3 keeps each
expert's parameter count: 1024 for the 63B model (exact), 1365 for the 173B model (0.02 % fewer).

Parameters are created with torch.empty and not written: a rank touches only the pages of the experts AutoEP keeps for
it, so a model larger than a node's host memory can be built on every rank. ``initialize_parameters`` fills what each
rank holds after AutoEP and AutoTP have placed it, from generators seeded by the parameter's name (and an expert's
global index, a shard's tensor-parallel rank), so that the initial weights do not depend on the layout.
"""
import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    layers: int
    hidden: int
    heads: int
    ffn: int  # width of one SwiGLU expert
    experts: int
    topk: int
    seq: int
    vocab: int = 50304


SHAPES = {
    # for a laptop or a CPU test
    'tiny': Shape(layers=4, hidden=64, heads=4, ffn=32, experts=8, topk=2, seq=16, vocab=128),
    # X-MoE-4D's 10B shape with 8 layers: fits one or two nodes of 8 GPUs
    'small': Shape(layers=8, hidden=2048, heads=16, ffn=939, experts=64, topk=6, seq=2048),
    # X-MoE-4D's "Medium" (63B) and "Large" (173B), Table I of the X-MoE-4D paper
    '63b': Shape(layers=32, hidden=5120, heads=40, ffn=1024, experts=128, topk=6, seq=4096),
    '173b': Shape(layers=24, hidden=7168, heads=56, ffn=1365, experts=256, topk=8, seq=4096),
}


def _empty(*shape):
    return nn.Parameter(torch.empty(*shape))


def _linear(n_in, n_out):
    return nn.utils.skip_init(nn.Linear, n_in, n_out, bias=False)  # allocated, not written


class Embed(nn.Module):

    def __init__(self, c: Shape):
        super().__init__()
        self.weight = _empty(c.vocab, c.hidden)

    def forward(self, ids):
        return F.embedding(ids, self.weight)


class Experts(nn.Module):
    """The experts' weights in the layout of Hugging Face's Mixtral experts, which AutoEP reads.

    The two tensors hold every expert of the layer, far more than one device can hold in a large model, so they stay
    in host memory, never written, until AutoEP has cut out this rank's experts (DeepSpeed's engine converts the model
    before it moves it to the device). A PipelineModule moves the layers it builds to the device in its constructor,
    before the engine runs: this module ignores that move. AutoEP's layer replaces it, and the engine moves that one."""

    def __init__(self, c: Shape):
        super().__init__()
        self.gate_up_proj = _empty(c.experts, 2 * c.ffn, c.hidden)
        self.down_proj = _empty(c.experts, c.hidden, c.ffn)

    def _apply(self, fn, recurse=True):
        return self


class MoE(nn.Module):
    """The block AutoEP converts. Its own forward is the unconverted reference (softmax router, top-k, weights
    renormalized to sum to one, SwiGLU experts); after conversion AutoEP's layer runs instead."""

    def __init__(self, c: Shape):
        super().__init__()
        self.topk = c.topk
        self.gate = _linear(c.hidden, c.experts)
        self.experts = Experts(c)

    def forward(self, x):
        b, s, h = x.shape
        t = x.reshape(-1, h)
        weights, index = F.softmax(self.gate(t).float(), dim=-1).topk(self.topk, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(t.dtype)
        out = torch.zeros_like(t)
        for j in range(self.topk):
            gate, up = torch.einsum('th,tfh->tf', t, self.experts.gate_up_proj[index[:, j]]).chunk(2, dim=-1)
            out += weights[:, j:j + 1] * torch.einsum('tf,thf->th', F.silu(gate) * up,
                                                      self.experts.down_proj[index[:, j]])
        return out.view(b, s, h)


class Block(nn.Module):

    def __init__(self, c: Shape):
        super().__init__()
        self.head_dim = c.hidden // c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.hidden), nn.LayerNorm(c.hidden)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (_linear(c.hidden, c.hidden) for _ in range(4))
        self.mlp = MoE(c)

    def forward(self, x):
        b, s, _ = x.shape
        y = self.ln1(x)
        # the number of heads comes from the projection's width, so a tensor-parallel shard keeps its own heads
        q, k, v = (p(y).view(b, s, -1, self.head_dim).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(o.transpose(1, 2).reshape(b, s, -1))
        y = self.mlp(self.ln2(x))
        return x + (y[0] if isinstance(y, tuple) else y)


class BlockRecomputed(Block):
    """The same block under another class name. Upstream DeepSpeed without per-stage counts can recompute part of the
    blocks only by class (checkpointable_layers): the baseline of run_planner.sh builds the blocks to recompute as this
    class."""


class Head(nn.Module):

    def __init__(self, c: Shape):
        super().__init__()
        self.ln = nn.LayerNorm(c.hidden)
        self.weight = _empty(c.vocab, c.hidden)

    def forward(self, x):
        return F.linear(self.ln(x), self.weight)


def lm_loss(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


class Decoder(nn.Module):
    """The layers as one module, for pipeline size 1. ``recompute`` blocks run under activation checkpointing."""

    def __init__(self, c: Shape, recompute=0):
        super().__init__()
        self.layers = nn.Sequential(Embed(c), *[Block(c) for _ in range(c.layers)], Head(c))
        self.recompute = recompute

    def forward(self, ids):
        x = self.layers[0](ids)
        for i, block in enumerate(self.layers[1:-1]):
            if self.training and i < self.recompute:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.layers[-1](x)


def flops_per_token(c: Shape):
    """Model FLOPs of one trained token: forward + backward = 3 x forward, recomputation not counted, attention over the
    whole sequence (no causal halving, as in Megatron-LM's formula). Per block: 8 h^2 (q, k, v, o) + 4 s h (scores and
    values) + 2 h E (router) + k x 6 h f (the k experts' three matrices); plus the output layer 2 h V."""
    h, f = c.hidden, c.ffn
    block = 8 * h * h + 4 * c.seq * h + 2 * h * c.experts + c.topk * 6 * h * f
    return 3 * (c.layers * block + 2 * h * c.vocab)


def parameter_count(c: Shape):
    block = 4 * c.hidden * c.hidden + c.experts * 3 * c.hidden * c.ffn + c.hidden * c.experts + 4 * c.hidden
    return c.layers * block + 2 * c.vocab * c.hidden + 2 * c.hidden


# --- DeepSpeed settings of each feature ---

TP_SPECS = [
    {"patterns": [r".*\.o_proj\.weight$"], "partition_type": "row"},
    {"patterns": [r".*\.[qkv]_proj\.weight$"], "partition_type": "column"},
]


def expert_parallel_config(c: Shape, ep, pipeline, rbd=False, ssmb=False, grouped_mm=True):
    """AutoEP for this model. ``rbd``: X-MoE's redundancy-bypassing dispatch (comm_backend "two_level"). ``ssmb``:
    X-MoE's sequence-sharded MoE block (needs tensor parallelism)."""
    cfg = {
        "enabled": True,
        "autoep_size": ep,
        # a plain module names a block's MoE "layers.<i>.mlp", a PipelineModule "<i>.mlp"
        "moe_layer_pattern": r"\d+\.mlp" if pipeline else r"layers\.\d+\.mlp",
        "top_k": c.topk,
        "use_grouped_mm": grouped_mm,
    }
    if rbd:
        cfg["comm_backend"] = "two_level"
    if ssmb:
        cfg["sequence_sharded_moe"] = True
    return cfg


def tensor_parallel_config(tp):
    return {"autotp_size": tp, "partition_config": {"use_default_specs": False, "layer_specs": TP_SPECS}}


def full_shapes(module):
    """The shape of every parameter before AutoEP and AutoTP place them (call before deepspeed.initialize)."""
    return {name: tuple(p.shape) for name, p in module.named_parameters()}


def initialize_parameters(module, seed, shapes, layers, tp_rank=0, tp_size=1, std=0.02):
    """Fill, in place, the parameters this rank holds after AutoEP and AutoTP have placed them: matrices normal(0, std)
    (the output projections of attention and of the experts scaled by 1 / sqrt(2 L), as GPT-2 does), norm weights one,
    biases zero. Each tensor has its own generator, seeded from its name and, for an expert, its global index; a
    tensor-parallel shard is cut from the whole tensor drawn the same way. So every copy of a tensor gets the same values
    and the initial weights do not depend on the pipeline split, the expert placement or the tensor-parallel size.
    ``shapes`` comes from full_shapes; ``layers`` is the number of blocks of the whole model (a pipeline stage holds
    only some)."""
    from deepspeed.checkpoint.constants import AUTOEP_PARAM_LOCAL_EXPERTS, DS_AUTOEP_UC_META

    def draw(shape, device, scale, *key):
        digest = hashlib.sha256("|".join(map(str, (seed, *key))).encode()).digest()
        g = torch.Generator(device=device).manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale

    with torch.no_grad():
        for name, p in module.named_parameters():
            # a PipelineModule names a layer by its global index, as nn.Sequential does: the same name in every split
            key = name.replace('layers.', '', 1)
            # output projections: attention's o_proj, the experts' down_proj (w2 once AutoEP has converted them)
            scale = std / math.sqrt(2 * layers) if any(k in name for k in ('o_proj', 'down_proj', 'experts.w2')) else std
            meta = getattr(p, DS_AUTOEP_UC_META, None)
            if meta is not None:
                for local, expert in enumerate(meta[AUTOEP_PARAM_LOCAL_EXPERTS]):
                    p[local].copy_(draw(p.shape[1:], p.device, scale, key, 'expert', expert))
            elif p.ndim >= 2:
                # AutoEP renames what it converts (mlp.gate becomes mlp.router.gate); only AutoTP's shards differ
                whole = draw(shapes.get(name, tuple(p.shape)), p.device, scale, key)
                if tuple(p.shape) != whole.shape:  # an AutoTP shard: cut along the dimension it is split on
                    dim = next(d for d, (a, b) in enumerate(zip(p.shape, whole.shape)) if a != b)
                    whole = whole.chunk(tp_size, dim)[tp_rank]
                p.copy_(whole)
            elif name.endswith('bias'):
                p.zero_()
            else:
                p.fill_(1.0)
