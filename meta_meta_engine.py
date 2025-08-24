#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, random, time, math, sys
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict

# ============================================================
#                    Primes utilities
# ============================================================

def sieve_upto(n: int) -> List[int]:
    if n < 2: return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(n**0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:n+1:step] = b"\x00" * (((n - start)//step) + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]

def first_k_odd_primes(k: int) -> List[int]:
    if k <= 0: return []
    if k < 6:
        limit = 100
    else:
        kk = float(k)
        limit = int(max(100, kk * (math.log(kk) + math.log(math.log(kk))) * 8))
    while True:
        primes = sieve_upto(limit)
        odd_primes = [p for p in primes if p % 2 == 1]
        if len(odd_primes) >= k:
            return odd_primes[:k]
        limit *= 2  # grow until we have enough

# ============================================================
#                     Miller–Rabin (MR)
# ============================================================

def _dec(n: int):
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    return d, s

def _mr_witness(a: int, d: int, n: int, s: int) -> bool:
    x = pow(a, d, n)
    if x in (1, n-1): return False
    for _ in range(s-1):
        x = (x * x) % n
        if x == n-1: return False
    return True

def is_probable_prime(n: int, rounds: int = 12, rng: random.Random | None = None) -> bool:
    if n < 2: return False
    # quick trial division by small primes up to 97
    for p in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97):
        if n == p: return True
        if n % p == 0: return False
    d, s = _dec(n)
    if rng is None: rng = random.Random(2025)
    for _ in range(rounds):
        a = rng.randrange(2, n-1)
        if _mr_witness(a, d, n, s):
            return False
    return True

# ============================================================
#                Sampling & Goldbach engine
# ============================================================

def random_even_with_digits(rng: random.Random, d: int) -> int:
    if d < 1: raise ValueError("digits must be >= 1")
    if d == 1:
        return rng.choice([2,4,6,8])
    first = rng.randint(1,9)
    middle = ''.join(str(rng.randint(0,9)) for _ in range(d-2))
    last = rng.choice("02468")
    return int(str(first) + middle + last)

def goldbach_decomps(n: int, subtractors: List[int], mr_rounds: int, rng: random.Random,
                     cap_per_n: int | None = None) -> List[Tuple[int,int]]:
    """
    Return unordered Goldbach decompositions found via the subtractor scan.
    Deduplicates by unordered pair. If cap_per_n is set, record at most that many (still scans all p).
    NOTE: This intentionally **skips p=2**, i.e., it will not include (2, n-2).
    """
    if n % 2: n -= 1
    seen: Set[Tuple[int,int]] = set()
    recorded: List[Tuple[int,int]] = []
    for p in subtractors:
        q = n - p
        if q < 2: continue
        # with p odd and n even, q is odd (or 2); even q>2 would be composite
        if q != 2 and (q & 1) == 0:
            continue
        if is_probable_prime(q, rounds=mr_rounds, rng=rng):
            a, b = (p, q) if p <= q else (q, p)
            if (a, b) not in seen:
                seen.add((a, b))
                if cap_per_n is None or len(recorded) < cap_per_n:
                    recorded.append((a, b))
    return recorded

# ============================================================
#              Meta–Meta–Heuristic Engine (full)
# ============================================================

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    from math import lgamma
    return math.exp(-lam + k*math.log(lam) - lgamma(k+1))

def _capped_poisson_pmf(y: int, lam: float, cap: Optional[int]) -> float:
    """Y = min(X, cap) where X ~ Poisson(lam). If cap is None => plain Poisson."""
    if cap is None:
        return _poisson_pmf(y, lam)
    if y < cap:
        return _poisson_pmf(y, lam)
    # pile the tail onto 'cap'
    cdf = 0.0
    for k in range(cap):
        cdf += _poisson_pmf(k, lam)
    return max(0.0, 1.0 - cdf)

def _capped_poisson_cdf_table(lam: float, cap: Optional[int], ymax: int) -> List[float]:
    """Return [F(0), F(1), ..., F(ymax)] for Y=min(X,cap)."""
    F = []
    s = 0.0
    if cap is None:
        for k in range(ymax+1):
            s += _poisson_pmf(k, lam)
            F.append(min(1.0, s))
        return F
    for k in range(ymax+1):
        if k < cap:
            s += _poisson_pmf(k, lam)
            F.append(min(1.0, s))
        else:
            F.append(1.0)
    return F

def _rand_pit_discrete(y: int, cdf_vals: List[float], rng: random.Random) -> float:
    """
    Randomized PIT for discrete Y with CDF F:
      U = F(y-) + V*(F(y)-F(y-)), V~Uniform(0,1)
    """
    y = max(0, min(y, len(cdf_vals)-1))
    Fy   = cdf_vals[y]
    Fy_1 = 0.0 if y == 0 else cdf_vals[y-1]
    v = rng.random()
    return Fy_1 + v * (Fy - Fy_1)

# ---------- Heuristic models (plug-in) ----------

class HeuristicModel:
    name: str = "base"
    def predict_lambda(self, n: int, subtractors: List[int]) -> float:
        raise NotImplementedError

class H0_Naive(HeuristicModel):
    """λ ≈ K / ln n"""
    name = "H0_naive"
    def predict_lambda(self, n: int, subtractors: List[int]) -> float:
        K = len(subtractors)
        return K / max(1.0, math.log(n))

class H1_Sieved(HeuristicModel):
    """
    Downweight subtractors eliminated by small primes:
      keep p only if (n - p) % r != 0 for all r in small_primes (unless q==r).
    Then λ ≈ sum_{p in survivors} 1 / ln(n - p).
    """
    name = "H1_sieved"
    def __init__(self, small_primes: List[int] = [3,5,7,11,13,17,19,23,29]):
        self.small_primes = small_primes
    def predict_lambda(self, n: int, subtractors: List[int]) -> float:
        total = 0.0
        for p in subtractors:
            q = n - p
            if q <= 2:
                continue
            knocked = False
            for r in self.small_primes:
                if q % r == 0 and q != r:
                    knocked = True
                    break
            if not knocked:
                total += 1.0 / max(1.0, math.log(q))
        return total

class H2_FractionalSieve(HeuristicModel):
    """
    Faster approximation: multiply K by product over small primes (survival fractions),
    then λ ≈ K_eff / ln n.
    """
    name = "H2_frac_sieve"
    def __init__(self, small_primes: List[int] = [3,5,7,11,13,17,19,23,29]):
        self.small_primes = small_primes
        prod = 1.0
        for r in self.small_primes:
            prod *= (1.0 - 1.0/(r-1))
        self.survival = max(0.0, min(1.0, prod))
    def predict_lambda(self, n: int, subtractors: List[int]) -> float:
        K_eff = len(subtractors) * self.survival
        return K_eff / max(1.0, math.log(n))

# ---------- Evidence aggregator ----------

class MetaMetaEngine:
    """
    Online mixture over heuristic models using log-likelihood scores on counts Y.
    Also computes calibration via randomized PIT and residue-bucket stability.
    """
    def __init__(self, models: List[HeuristicModel], cap_per_n: Optional[int], rng_seed: int = 2025,
                 pit_ymax: int = 64):
        assert models, "need at least one heuristic model"
        self.models = models
        self.cap = cap_per_n
        self.rng = random.Random(rng_seed)
        self.pit_ymax = pit_ymax

        # mixture weights (start uniform)
        self.w = [1.0/len(models)] * len(models)

        # running stats
        self.n_obs = 0
        self.loglik_sum_each = [0.0]*len(models)   # cumulative log-likelihood per model
        self.loglik_sum_mix  = 0.0                 # cumulative log-likelihood of the mixture
        self.pit_vals: List[float] = []            # randomized PIT values (for calibration)
        self.bucket_scores: Dict[Tuple[int,int,int], float] = defaultdict(float)  # e.g., (n%3, n%5, n%7) -> loglik
        self.conf_trend: List[float] = []

    def _pmf(self, y: int, lam: float) -> float:
        return _capped_poisson_pmf(y, lam, self.cap)

    def _cdf_table(self, lam: float, ymax: int) -> List[float]:
        return _capped_poisson_cdf_table(lam, self.cap, ymax)

    def update(self, n: int, subtractors: List[int], y_count: int):
        """
        One observation Y=y_count at even n.
        - Compute per-model λ_i(n)
        - Score log PMF_i(y)
        - Update mixture weights
        - Randomized PIT calibration on mixture CDF
        - Bucket stability (n mod 3,5,7)
        """
        self.n_obs += 1

        # Predict lambdas (clamp tiny)
        lambdas = [max(1e-12, m.predict_lambda(n, subtractors)) for m in self.models]

        # Per-model likelihoods
        pmfs = [self._pmf(y_count, lam) for lam in lambdas]
        pmfs = [max(p, 1e-300) for p in pmfs]  # guard zeros
        logs = [math.log(p) for p in pmfs]
        for i, l in enumerate(logs):
            self.loglik_sum_each[i] += l

        # Mixture predictive & weight update (expert advice)
        mix_pred = sum(w*p for w,p in zip(self.w, pmfs))
        mix_pred = max(mix_pred, 1e-300)
        self.loglik_sum_mix += math.log(mix_pred)
        new_w = [w*p for w,p in zip(self.w, pmfs)]
        Z = sum(new_w)
        self.w = [w/Z for w in new_w] if Z > 0 else [1.0/len(self.models)]*len(self.models)

        # Randomized PIT using mixture CDF
        ymax = self.cap if (self.cap is not None and self.cap < self.pit_ymax) else self.pit_ymax
        cdfs_each = [self._cdf_table(lam, ymax) for lam in lambdas]
        cdf_mix   = [sum(self.w[j]*cdfs_each[j][t] for j in range(len(self.models))) for t in range(ymax+1)]
        pit = _rand_pit_discrete(min(y_count, ymax), cdf_mix, self.rng)
        self.pit_vals.append(pit)

        # Residue-bucket stability
        bkey = (n % 3, n % 5, n % 7)
        self.bucket_scores[bkey] += math.log(mix_pred)

        # Confidence index
        self.conf_trend.append(self._confidence_index())

    # ---- Internal scoring helpers ----

    def _pit_calibration_score(self) -> float:
        """
        Score in [0,1], where 1 = perfect uniformity (KS-style penalty).
        """
        if not self.pit_vals:
            return 0.0
        pits = sorted(self.pit_vals)
        n = len(pits)
        D = 0.0
        for i, u in enumerate(pits, start=1):
            D = max(D, abs(i/n - u), abs(u - (i-1)/n))
        c = 50.0  # tunable
        return math.exp(-c * (D**2))

    def _stability_penalty(self) -> float:
        """
        Penalize large variance of per-bucket mean log-score.
        Maps variance -> [0,1] with a simple 1/(1+α var).
        """
        if not self.bucket_scores:
            return 1.0
        vals = list(self.bucket_scores.values())
        mean = sum(vals)/len(vals)
        var = sum((v-mean)**2 for v in vals)/max(1, len(vals)-1)
        return 1.0 / (1.0 + 0.05*var)

    def _avg_logscore_advantage(self) -> float:
        """
        Average log-score advantage of the best model over H0 (if present),
        squashed to [0,1].
        """
        try:
            i0 = next(i for i,m in enumerate(self.models) if m.name.startswith("H0"))
        except StopIteration:
            i0 = 0
        best = max(self.loglik_sum_each)
        base = self.loglik_sum_each[i0]
        adv_per_obs = (best - base) / max(1, self.n_obs)
        k = 2.0
        return 1.0 - math.exp(-k * max(0.0, adv_per_obs))

    def _confidence_index(self) -> float:
        """
        Combine:
          - calibration (PIT uniformity)   -> s_cal
          - stability across buckets       -> s_stab
          - predictive advantage           -> s_pred
        CI = geometric mean in [0,1].
        """
        s_cal  = self._pit_calibration_score()
        s_stab = self._stability_penalty()
        s_pred = self._avg_logscore_advantage()
        eps = 1e-6
        g = math.exp((math.log(s_cal+eps) + math.log(s_stab+eps) + math.log(s_pred+eps))/3.0)
        return max(0.0, min(1.0, g))

    # ---- Public summaries ----

    def weights(self) -> List[float]:
        return list(self.w)

    def summary(self) -> Dict[str, float | str | int]:
        best_i = max(range(len(self.models)), key=lambda i: self.loglik_sum_each[i])
        out: Dict[str, float | str | int] = {
            "observations": self.n_obs,
            "mix_avg_logscore": self.loglik_sum_mix / max(1, self.n_obs),
            "best_model_index": best_i,
            "best_model_name": self.models[best_i].name,
            "best_avg_logscore": self.loglik_sum_each[best_i] / max(1, self.n_obs),
            "calibration_score": self._pit_calibration_score(),
            "stability_score": self._stability_penalty(),
            "predictive_advantage": self._avg_logscore_advantage(),
            "confidence_index": self.conf_trend[-1] if self.conf_trend else 0.0,
        }
        # Include H0 score if present
        try:
            i0 = next(i for i,m in enumerate(self.models) if m.name.startswith("H0"))
            out["H0_avg_logscore"] = self.loglik_sum_each[i0] / max(1, self.n_obs)
        except StopIteration:
            pass
        return out

    def pit_histogram(self, bins: int = 10) -> List[Tuple[float, float]]:
        """Return list of (bin_right_edge, frequency) for PIT values."""
        if not self.pit_vals: return []
        counts = [0]*bins
        for u in self.pit_vals:
            j = min(bins-1, int(u*bins))
            counts[j] += 1
        total = len(self.pit_vals)
        edges = [ (i+1)/bins for i in range(bins) ]
        freqs = [ c/total for c in counts ]
        return list(zip(edges, freqs))

# ============================================================
#                        CLI & main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Goldbach PoC + Meta-Meta-Heuristic Engine")
    ap.add_argument("--trials", type=int, default=1000, help="number of random tests to run (default: 1000)")
    ap.add_argument("--digits", type=int, default=30, help="number of digits in n (even numbers only; default: 30)")
    ap.add_argument("--subtractors", type=int, default=40, help="how many odd primes to subtract (10..100; default: 40)")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed (default: 2025)")
    ap.add_argument("--mr-rounds", type=int, default=12, help="Miller–Rabin rounds per candidate (default: 12)")
    ap.add_argument("--max-per-n", type=int, default=None, help="cap on how many decomps to record per n (scan still runs fully)")
    ap.add_argument("--print-examples", type=int, default=0, help="print up to K example decomps for the first few n with any hits")

    # Engine options
    ap.add_argument("--engine-pit-ymax", type=int, default=64, help="max y for PIT CDF table (default: 64)")
    ap.add_argument("--engine-pit-bins", type=int, default=10, help="bins for PIT histogram in report (default: 10)")
    ap.add_argument("--engine-report-every", type=int, default=0, help="if >0, print engine snapshots every N trials")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    if not (10 <= args.subtractors <= 100):
        print(f"error: --subtractors must be between 10 and 100 (got {args.subtractors})", file=sys.stderr)
        sys.exit(2)
    if args.digits < 1:
        print(f"error: --digits must be >= 1 (got {args.digits})", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    subtractor_primes = first_k_odd_primes(args.subtractors)

    # --- Meta-Meta engine setup ---
    models: List[HeuristicModel] = [H0_Naive(), H1_Sieved(), H2_FractionalSieve()]
    engine = MetaMetaEngine(
        models=models,
        cap_per_n=args.max_per_n,
        rng_seed=args.seed,
        pit_ymax=args.engine_pit_ymax
    )

    t0 = time.time()
    total_hits = 0
    total_decomps = 0
    per_n_counts: List[int] = []
    example_counter = 0
    example_limit = int(max(0, args.print_examples)) if isinstance(args.print_examples, int) else 0  # guard

    # We will store up to 'print-examples' example entries (n, decomps list)
    examples: List[Tuple[int, List[Tuple[int,int]]]] = []

    for i in range(args.trials):
        n = random_even_with_digits(rng, args.digits)
        decomps = goldbach_decomps(n, subtractor_primes, args.mr_rounds, rng, cap_per_n=args.max_per_n)
        c = len(decomps)

        # Update meta-meta engine with observed count
        engine.update(n, subtractor_primes, c)

        if c > 0:
            total_hits += 1
            total_decomps += c
            per_n_counts.append(c)
            if example_counter < example_limit:
                examples.append((n, decomps))
                example_counter += 1
        else:
            per_n_counts.append(0)

        if args.engine_report_every and (i+1) % args.engine_report_every == 0:
            w = engine.weights()
            print(f"[engine] after {i+1} trials: weights={['%.3f'%x for x in w]} CI={engine.conf_trend[-1]:.3f}")

    elapsed = time.time() - t0
    per_n = elapsed / args.trials if args.trials else float('nan')
    hit_rate = total_hits / args.trials if args.trials else float('nan')
    mean_decomps = (total_decomps / total_hits) if total_hits else 0.0
    max_decomps = max(per_n_counts) if per_n_counts else 0
    min_decomps = min(per_n_counts) if per_n_counts else 0

    # ---- PoC Report ----
    print("=== Goldbach Proof of Concept (multi-decomp) ===")
    print(f"trials: {args.trials}   digits(n): {args.digits}   subtractors: {args.subtractors} odd primes")
    print(f"mr_rounds: {args.mr_rounds}   seed: {args.seed}   max_per_n: {args.max_per_n}")
    print(f"elapsed: {elapsed:.6f} s   per_n: {per_n*1000:.3f} ms")
    print(f"n with ≥1 decomp: {total_hits}   hit_rate: {hit_rate:.3f}")
    print(f"total decomps recorded: {total_decomps}")
    print(f"per-hit mean decomps: {mean_decomps:.3f}   per-n min/max (over all n): {min_decomps}/{max_decomps}")
    print(f"first few subtractors: {subtractor_primes[:10]} ...")

    if examples:
        print("\n--- example decomps (n → up to first K unordered pairs (p,q)) ---")
        for n, ds in examples:
            shown = ", ".join(f"({p},{q})" for p, q in ds)
            print(f"{n} → {shown}")

    # ---- Meta-Meta Engine Report ----
    print("\n=== Meta–Meta–Heuristic Engine ===")
    summary = engine.summary()
    for k in ["observations", "mix_avg_logscore", "best_model_index", "best_model_name",
              "best_avg_logscore", "H0_avg_logscore",
              "calibration_score", "stability_score",
              "predictive_advantage", "confidence_index"]:
        if k in summary:
            print(f"{k}: {summary[k]}")

    # PIT histogram to eyeball calibration
    pit_hist = engine.pit_histogram(bins=args.engine_pit_bins)
    if pit_hist:
        print("\nPIT histogram (bin_edge → freq):")
        for edge, freq in pit_hist:
            print(f"{edge:.2f} → {freq:.3f}")

if __name__ == '__main__':
    main()
