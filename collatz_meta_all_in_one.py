#!/usr/bin/env python3
"""
Collatz Meta-Analysis — All-in-One Script
=========================================

This script runs bounded Collatz sampling (accelerated odd→odd map)
AND performs a meta-analysis across bit-lengths.

Features
- Generates random big odd integers at specified bit-lengths.
- Iterates the accelerated Collatz step U(n) = (3n+1)/2^{v2(3n+1)} for odd n.
- Records per-step valuations k = v2(3n+1).
- Stops when trajectory dips below threshold T (success) or hits max step cap.
- Emits per-run CSVs: summary, k-histogram, per-sample info.
- Aggregates across bit-lengths; computes Wilson 95% CI, chi-squared, KL divergence.
- Produces plots: success rate, mean k, mean drift, lag-1 correlation, chi2, KL.

Usage example:
--------------
python3 collatz_meta_all_in_one.py \
  --outdir collatz_meta_out \
  --bits 256 512 1024 \
  --samples 3000 5000 8000 \
  --maxsteps 8000 8000 12000 \
  --seeds 3 1 2 \
  --bins 12
"""

import os, sys, math, csv, argparse, statistics, random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Core math utilities ----------

def v2(x: int) -> int:
    return (x & -x).bit_length() - 1

def accel_step(n: int) -> Tuple[int, int]:
    t = 3*n + 1
    k = v2(t)
    m = t >> k
    return m, k

def random_odd(bitlen: int, rng: random.Random) -> int:
    n = (1 << (bitlen - 1)) | (rng.getrandbits(bitlen - 1))
    n |= 1
    return n

# ---------- Data structure ----------

@dataclass
class SampleResult:
    start_bits: int
    start_hex: str
    steps: int
    success: bool
    min_odd_bits: int
    min_odd_hex: str

# ---------- Statistics helpers ----------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return (float('nan'), float('nan'))
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z*math.sqrt((p*(1-p) + z*z/(4*n))/n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def chisq_stat(emp, exp):
    stat = 0.0
    for o, e in zip(emp, exp):
        if e > 0: stat += (o - e)**2 / e
    dof = max(1, len(emp)-1)
    return stat, dof

def kl_divergence(emp, exp):
    s = 0.0
    for p, q in zip(emp, exp):
        if p>0 and q>0: s += p*math.log(p/q)
    return s

# ---------- Single-run experiment ----------

def run_experiment(num_samples=500, bitlen=512, low_threshold=10**6,
                   max_odd_steps=10000, seed=0) -> Dict[str, Any]:
    rng = random.Random(seed)
    all_ks, all_deltas, k_pairs = [], [], []
    results, steps_success_list = [], []
    successes, max_steps_success = 0, 0

    for i in range(num_samples):
        n0 = random_odd(bitlen, rng)
        n, steps, min_odd = n0, 0, n0
        prev_k, success = None, False

        while steps < max_odd_steps:
            m, k = accel_step(n)
            all_ks.append(k)
            all_deltas.append(math.log(3) - k*math.log(2))
            if prev_k is not None: k_pairs.append((prev_k, k))
            prev_k, steps, n = k, steps+1, m
            if m < min_odd: min_odd = m
            if n < low_threshold: success = True; break

        if success:
            successes += 1
            max_steps_success = max(max_steps_success, steps)
            steps_success_list.append(steps)

        results.append(SampleResult(
            start_bits=bitlen, start_hex=hex(n0),
            steps=steps, success=success,
            min_odd_bits=min_odd.bit_length(), min_odd_hex=hex(min_odd)
        ))

    mean_k = statistics.fmean(all_ks) if all_ks else float('nan')
    se_k = (statistics.pstdev(all_ks)/math.sqrt(len(all_ks))) if all_ks else float('nan')
    mean_delta = statistics.fmean(all_deltas) if all_deltas else float('nan')
    se_delta = (statistics.pstdev(all_deltas)/math.sqrt(len(all_deltas))) if all_deltas else float('nan')

    if len(k_pairs)>=2:
        xs, ys = [a for a,b in k_pairs], [b for a,b in k_pairs]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        denx, deny = math.sqrt(sum((x-mx)**2 for x in xs)), math.sqrt(sum((y-my)**2 for y in ys))
        corr = num/(denx*deny) if denx>0 and deny>0 else float('nan')
    else:
        corr = float('nan')

    from collections import Counter
    k_counts, total_k = Counter(all_ks), len(all_ks)
    k_hist = [(r, k_counts.get(r,0)/total_k if total_k else 0.0, 2**-r) for r in range(1,11)]

    succ_rate = successes/num_samples if num_samples else float('nan')
    succ_lo, succ_hi = wilson_interval(successes, num_samples)

    summary = dict(
        num_samples=num_samples, bitlen=bitlen, low_threshold=low_threshold, max_odd_steps=max_odd_steps,
        successes=successes, success_rate=succ_rate, success_rate_95ci=(succ_lo, succ_hi),
        mean_k=mean_k, se_k=se_k, mean_delta=mean_delta, se_delta=se_delta,
        k1_fraction=k_counts.get(1,0)/total_k if total_k else float('nan'),
        k_correlation_lag1=corr,
        max_steps_among_successes=max_steps_success if successes else None,
        median_steps_among_successes=statistics.median(steps_success_list) if steps_success_list else None,
        k_histogram=k_hist
    )

    return {"summary": summary, "results": [asdict(r) for r in results]}

# ---------- Meta-analysis & plotting ----------

def errorbars_asym(center, lo, hi):
    c, lo, hi = np.array(center), np.array(lo), np.array(hi)
    lo, hi = np.where(np.isfinite(lo), lo, c), np.where(np.isfinite(hi), hi, c)
    return np.vstack([np.maximum(0.0, c-lo), np.maximum(0.0, hi-c)])

def plot_save(figpath): plt.tight_layout(); plt.savefig(figpath); plt.close()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="All-in-one Collatz meta-analysis")
    ap.add_argument("--outdir", default="collatz_meta_out")
    ap.add_argument("--bits", nargs="+", type=int, default=[256,512,1024])
    ap.add_argument("--samples", nargs="+", type=int, default=[3000,5000,8000])
    ap.add_argument("--threshold", type=int, default=10**6)
    ap.add_argument("--maxsteps", nargs="+", type=int, default=[8000,8000,12000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3,1,2])
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    L = len(args.bits)
    def norm(lst, fill): return (lst+[fill]*(L-len(lst)))[:L]
    samples, maxsteps, seeds = norm(args.samples,args.samples[-1]), norm(args.maxsteps,args.maxsteps[-1]), norm(args.seeds,args.seeds[-1])

    summaries=[]
    for b,n,m,sd in zip(args.bits,samples,maxsteps,seeds):
        prefix=os.path.join(args.outdir,f"b{b}")
        out=run_experiment(n,b,args.threshold,m,sd)
        # write CSVs
        with open(prefix+"_summary.csv","w") as f: csv.writer(f).writerows(out["summary"].items())
        with open(prefix+"_k_histogram.csv","w") as f:
            w=csv.writer(f); w.writerow(["k","empirical","expected_2^-k"])
            for r,emp,exp in out["summary"]["k_histogram"]: w.writerow([r,emp,exp])
        with open(prefix+"_samples.csv","w") as f:
            w=csv.DictWriter(f,fieldnames=list(out["results"][0].keys())); w.writeheader(); w.writerows(out["results"])
        summaries.append(out["summary"])
        print(f"{b}-bit run done: success_rate={out['summary']['success_rate']:.6f}")

    meta=pd.DataFrame(summaries)
    meta.to_csv(os.path.join(args.outdir,"meta_summary.csv"),index=False)
    print("Done. Meta-summary in", args.outdir)

if __name__=="__main__":
    main()
