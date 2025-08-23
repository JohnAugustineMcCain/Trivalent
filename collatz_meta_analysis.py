#!/usr/bin/env python3
import os, sys, math, csv, json, subprocess
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def chisq_stat(emp, exp):
    # Chi-squared statistic over aligned probability vectors (not counts).
    # Lower is better; for p-values you need counts and SciPy (not required here).
    stat = 0.0
    for o, e in zip(emp, exp):
        if e > 0:
            stat += (o - e)**2 / e
    dof = max(1, len(emp) - 1)
    return stat, dof

def kl_div(emp, exp):
    s = 0.0
    for p, q in zip(emp, exp):
        if p > 0 and q > 0:
            s += p * math.log(p/q)
    return s

def load_summary(prefix):
    s = {}
    with open(prefix + "_summary.csv", newline="") as f:
        r = csv.reader(f)
        for k, v in r:
            s[k] = v
    # parse CI tuple if present as string
    ci = s.get("success_rate_95ci")
    if isinstance(ci, str) and ci.startswith("(") and ci.endswith(")"):
        inner = ci[1:-1]
        parts = inner.split(",")
        if len(parts) == 2:
            try:
                s["success_rate_lo"] = float(parts[0])
                s["success_rate_hi"] = float(parts[1])
            except:
                pass
    # cast numerics where expected
    def cast_float(key):
        if key in s:
            try: s[key] = float(s[key])
            except: pass
    for key in ["success_rate", "success_rate_lo", "success_rate_hi",
                "mean_k", "se_k", "mean_delta", "se_delta",
                "k1_fraction", "k_correlation_lag1",
                "max_steps_among_successes", "median_steps_among_successes"]:
        cast_float(key)
    for key in ["num_samples", "bitlen", "low_threshold", "max_odd_steps", "successes"]:
        if key in s:
            try: s[key] = int(float(s[key]))
            except: pass
    return s

def load_hist(prefix):
    return pd.read_csv(prefix + "_k_histogram.csv")

def errorbars_asym(center, lo, hi):
    # Build asymmetric yerr = [[lower],[upper]] suitable for matplotlib.errorbar
    c = np.array(center, dtype=float)
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    lo = np.where(np.isfinite(lo), lo, c)
    hi = np.where(np.isfinite(hi), hi, c)
    lower = np.maximum(0.0, c - lo)
    upper = np.maximum(0.0, hi - c)
    return np.vstack([lower, upper])

def plot_save(figpath):
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def run_bitlength(sampler, out_prefix, samples, bits, threshold, maxsteps, seed):
    cmd = [
        sys.executable, sampler,
        "--samples", str(samples),
        "--bits", str(bits),
        "--threshold", str(threshold),
        "--maxsteps", str(maxsteps),
        "--seed", str(seed),
        "--out_prefix", out_prefix
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Meta-analysis of Collatz heuristic confidence vs bit-length")
    ap.add_argument("--sampler", default="collatz_sampling.py")
    ap.add_argument("--outdir", default="collatz_meta_out")
    ap.add_argument("--bits", nargs="+", type=int, default=[256,512,1024])
    ap.add_argument("--samples", nargs="+", type=int, default=[3000,5000,8000])
    ap.add_argument("--threshold", type=int, default=10**6)
    ap.add_argument("--maxsteps", nargs="+", type=int, default=[8000,8000,12000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3,1,2])
    ap.add_argument("--bins", type=int, default=12, help="number of k-bins to compare for goodness-of-fit")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    sampler = args.sampler
    if not os.path.isabs(sampler):
        sampler = os.path.abspath(sampler)

    # Normalize length of parameter lists
    L = len(args.bits)
    def norm(lst, fill):
        return (lst + [fill]*(L - len(lst)))[:L]
    samples = norm(args.samples, args.samples[-1])
    maxsteps = norm(args.maxsteps, args.maxsteps[-1])
    seeds = norm(args.seeds, args.seeds[-1])

    # Run experiments (skip if outputs exist)
    prefixes = []
    for i, (b, n, m, sd) in enumerate(zip(args.bits, samples, maxsteps, seeds)):
        prefix = os.path.join(args.outdir, f"b{b}")
        prefixes.append(prefix)
        if not (os.path.exists(prefix + "_summary.csv") and os.path.exists(prefix + "_k_histogram.csv")):
            run_bitlength(sampler, prefix, n, b, args.threshold, m, sd)

    # Aggregate
    rows = []
    for b, prefix in zip(args.bits, prefixes):
        s = load_summary(prefix)
        h = load_hist(prefix)
        hK = h.head(args.bins).copy()
        emp = hK["empirical"].to_numpy()
        exp = hK["expected_2^-k"].to_numpy()
        # renormalize on first K bins to compare shapes
        if emp.sum() > 0: emp = emp/emp.sum()
        if exp.sum() > 0: exp = exp/exp.sum()
        chi, dof = chisq_stat(emp, exp)
        kl = kl_div(emp, exp)
        rows.append({
            "bits": b,
            "num_samples": s.get("num_samples"),
            "success_rate": s.get("success_rate"),
            "success_rate_lo": s.get("success_rate_lo"),
            "success_rate_hi": s.get("success_rate_hi"),
            "mean_k": s.get("mean_k"),
            "se_k": s.get("se_k"),
            "mean_delta": s.get("mean_delta"),
            "se_delta": s.get("se_delta"),
            "k1_fraction": s.get("k1_fraction"),
            "k_corr_lag1": s.get("k_correlation_lag1"),
            "steps_median": s.get("median_steps_among_successes"),
            "steps_max": s.get("max_steps_among_successes"),
            "chi2_stat_firstK": chi,
            "chi2_dof": dof,
            "kl_div_firstK": kl,
        })
    meta = pd.DataFrame(rows).sort_values("bits").reset_index(drop=True)
    meta_path = os.path.join(args.outdir, "meta_summary.csv")
    meta.to_csv(meta_path, index=False)

    # Plots
    # A) success rate
    plt.figure()
    yerr = errorbars_asym(meta["success_rate"], meta["success_rate_lo"], meta["success_rate_hi"])
    plt.errorbar(meta["bits"], meta["success_rate"], yerr=yerr, fmt="o-")
    plt.xlabel("Bit length of starting n")
    plt.ylabel("Success rate to threshold (95% CI)")
    plt.title("Collatz heuristic: descent success vs. bit-length")
    plt.grid(True)
    plot_save(os.path.join(args.outdir, "plot_success_rate.png"))

    # B) mean k
    plt.figure()
    se_k = np.array(meta["se_k"], dtype=float)
    se_k = np.where(np.isfinite(se_k) & (se_k>=0), se_k, 0.0)
    plt.errorbar(meta["bits"], meta["mean_k"], yerr=se_k, fmt="o-")
    plt.axhline(2.0)
    plt.xlabel("Bit length of starting n")
    plt.ylabel("Mean k = E[v2(3n+1)] (±1 SE)")
    plt.title("Geometric(1/2) prediction: E[k]=2")
    plt.grid(True)
    plot_save(os.path.join(args.outdir, "plot_mean_k.png"))

    # C) mean delta
    plt.figure()
    se_d = np.array(meta["se_delta"], dtype=float)
    se_d = np.where(np.isfinite(se_d) & (se_d>=0), se_d, 0.0)
    plt.errorbar(meta["bits"], meta["mean_delta"], yerr=se_d, fmt="o-")
    plt.axhline(math.log(3)-2*math.log(2))
    plt.xlabel("Bit length of starting n")
    plt.ylabel("Mean log-step change Δ (±1 SE)")
    plt.title("Negative drift prediction: E[Δ]=ln(3)-2ln(2)")
    plt.grid(True)
    plot_save(os.path.join(args.outdir, "plot_mean_delta.png"))

    # D) lag-1 correlation
    plt.figure()
    plt.plot(meta["bits"], meta["k_corr_lag1"], marker="o")
    plt.axhline(0.0)
    plt.xlabel("Bit length of starting n")
    plt.ylabel("Lag-1 correlation of k")
    plt.title("Independence check: corr(k_t, k_{t+1}) ≈ 0")
    plt.grid(True)
    plot_save(os.path.join(args.outdir, "plot_corr.png"))

    # E) chi2
    plt.figure()
    plt.plot(meta["bits"], meta["chi2_stat_firstK"], marker="o")
    plt.xlabel("Bit length of starting n")
    plt.ylabel(f"Chi-squared statistic (first {args.bins} k-bins)")
    plt.title("Fit to geometric law P(k=r)=2^{-r} (lower is better)")
    plt.grid(True)
    plot_save(os.path.join(args.outdir, "plot_chi2.png"))

    # F) KL
    plt.figure()
    plt.plot(meta["bits"], meta["kl_div_firstK"], marker="o")
    plt.xlabel("Bit length of starting n")
    plt.ylabel("KL divergence (nats) (first K bins)")
    plt.title("Distance to geometric law P(k=r)=2^{-r} (lower is better)")
    plt.grid(True)
    plot_save(os.path.join(args.outdir, "plot_kl.png"))

    print("Done. Outputs in:", args.outdir)
    print("Meta summary:", meta_path)

if __name__ == "__main__":
    main()
