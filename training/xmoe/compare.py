#!/usr/bin/env python3
"""Compare two runs of train.py from their per-step CSV files: throughput, memory and loss over the steps both ran.

  python compare.py runs/rbd_comm.csv runs/rbd_two_level.csv --skip 10

Both runs must have logged the same steps; a run that stopped early is an error, not a shorter mean. The throughput is
the mean model TFLOP/s per GPU over the steps after --skip (both files use model.flops_per_token, so the ratio is the
ratio of step times). The loss is compared step by step: two runs that differ only in a communication or memory
feature start from the same weights and batches, so their losses agree to rounding at the first steps.
"""
import argparse
import csv
import statistics
import sys


def read(path):
    with open(path, newline='') as f:
        return {int(r['step']): r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('feature')
    ap.add_argument('--skip', type=int, default=10, help='steps left out of the throughput (warm-up)')
    a = ap.parse_args()
    base, feat = read(a.baseline), read(a.feature)
    if set(base) != set(feat):
        sys.exit(f'the runs logged different steps: {a.baseline} {min(base, default=None)}-{max(base, default=None)}, '
                 f'{a.feature} {min(feat, default=None)}-{max(feat, default=None)}; a run stopped early')
    timed = [s for s in sorted(base) if s >= a.skip]
    if not timed:
        sys.exit(f'no step after --skip {a.skip}')

    def mean(rows, key):
        return statistics.mean(float(rows[s][key]) for s in timed)

    def peak(rows, key):
        return max(float(r[key]) for r in rows.values())

    tb, tf = mean(base, 'tflops_per_gpu'), mean(feat, 'tflops_per_gpu')
    first = min(base)
    loss_gap = max(abs(float(base[s]['loss']) - float(feat[s]['loss'])) for s in base)
    print(f'{a.baseline} -> {a.feature}, steps {timed[0]}-{timed[-1]}:')
    ratio = f'{tf / tb:.3f}x' if tb > 0 else 'n/a'
    print(f'  model TFLOP/s per GPU  {tb:.4g} -> {tf:.4g}  ({ratio})')
    print(f'  max allocated GiB      {peak(base, "max_allocated_gib"):.2f} -> {peak(feat, "max_allocated_gib"):.2f}')
    print(f'  max reserved GiB       {peak(base, "max_reserved_gib"):.2f} -> {peak(feat, "max_reserved_gib"):.2f}')
    print(f'  allocator retries      {base[max(base)]["alloc_retries"]} -> {feat[max(feat)]["alloc_retries"]}')
    print(f'  busiest expert rank    {peak(base, "busiest_expert_rank"):.2f} -> {peak(feat, "busiest_expert_rank"):.2f}'
          ' (rows over the mean; 1.0 is even)')
    print(f'  loss at step {first}       {float(base[first]["loss"]):.6f} -> {float(feat[first]["loss"]):.6f}; '
          f'largest gap over all steps {loss_gap:.2e}')
    last = sorted(base)[-10:]
    lb, lf = (statistics.mean(float(r[s]['loss']) for s in last) for r in (base, feat))
    print(f'  mean loss, last {len(last)} steps {lb:.6f} -> {lf:.6f} ({(lf - lb) / lb:+.3%})')


if __name__ == '__main__':
    main()
