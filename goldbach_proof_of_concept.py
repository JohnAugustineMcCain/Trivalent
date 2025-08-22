#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, random, hashlib, time, sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# =========================
# CONFIG — EDIT FREELY
# =========================
CONFIG = {
    # Sampling
    "STEPS": 60,                 # samples per run (or per digit in SWEEP_DIGITS mode)
    "MIN_DIGITS": 19,            # used if MODE == "RANGE"
    "MAX_DIGITS": 30,            # used if MODE == "RANGE"
    "EXACT_DIGITS": 24,          # used if MODE == "EXACT"
    "MODE": "RANGE",             # "RANGE" | "EXACT" | "SWEEP_DIGITS"
    "SWEEP_DIGITS": [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],

    # Confidence update
    "EPSILON": 0.002,            # bump to Cc per witnessed hit

    # RNG
    "SEED": 2025,

    # Primality testing knobs
    "SMALL_TRIAL_LIMIT": 1000,   # primes <= this for quick trial division
    # Miller–Rabin rounds by number of digits (choose your bands)
    "MR_ROUNDS_BY_DIGITS": [
        (18, 8),
        (40, 10),
        (80, 12),
        (10**9, 14),             # catch-all upper band
    ],

    # Tiny-budget Goldbach search
    "SUBTRACTORS": [3,5,7,11,13,17,19,23,29],
    "OFFSETS":     [-22,-14,-6,-2,2,6,14,22],   # 0 removed on purpose

    # Output / logging
    "PRINT_EXAMPLES": 8,         # print up to N example hits
    "SHOW_STATS": True,          # print detailed stats at end
}

# =========================
# IMPLEMENTATION
# =========================

def _primes_upto(n: int) -> List[int]:
    if n < 2: return []
    sieve = [True]*(n+1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(n**0.5)+1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:n+1:step] = [False]*(((n - start)//step)+1)
    return [i for i, ok in enumerate(sieve) if ok]

_SMALL_TRIAL: List[int] = _primes_upto(CONFIG["SMALL_TRIAL_LIMIT"])

@dataclass
class Stats:
    primality_tests_attempted: int = 0
    trial_div_composite_rejects: int = 0
    mr_calls: int = 0
    mr_composite_rejects: int = 0
    mr_rounds_total: int = 0
    probable_primes: int = 0

def _dec(n: int) -> Tuple[int,int]:
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

def _mr_rounds_for_digits(digits: int) -> int:
    for max_digits, rounds in CONFIG["MR_ROUNDS_BY_DIGITS"]:
        if digits <= max_digits:
            return rounds
    return CONFIG["MR_ROUNDS_BY_DIGITS"][-1][1]

def is_probable_prime(n: int, rounds: int, rng: random.Random, stats: Optional[Stats]=None) -> bool:
    if stats: stats.primality_tests_attempted += 1
    if n < 2:
        return False
    for p in _SMALL_TRIAL:
        if n == p:
            if stats: stats.probable_primes += 1
            return True
        if n % p == 0:
            if stats: stats.trial_div_composite_rejects += 1
            return False
    d,s = _dec(n)
    if stats: stats.mr_calls += 1
    for _ in range(rounds):
        a = rng.randrange(2, n-1)
        if _mr_witness(a, d, n, s):
            if stats: stats.mr_composite_rejects += 1
            return False
    if stats:
        stats.probable_primes += 1
        stats.mr_rounds_total += rounds
    return True

def mr(n: int, rng: random.Random, stats: Optional[Stats]=None) -> bool:
    digits = len(str(n))
    rounds = _mr_rounds_for_digits(digits)
    return is_probable_prime(n, rounds, rng, stats)

def random_even_with_digits(rng: random.Random, d: int) -> int:
    if d < 1: raise ValueError("digits must be >= 1")
    if d == 1:
        return rng.choice([2,4,6,8])
    first = rng.randint(1,9)
    middle = [rng.randint(0,9) for _ in range(d-2)]
    last = rng.choice([0,2,4,6,8])
    n = int(str(first) + "".join(map(str, middle)) + str(last))
    if n % 2 == 1:
        n -= 1
    if len(str(n)) != d:
        return random_even_with_digits(rng, d)
    return max(n, 2)

def sample_even_in_digit_range(rng: random.Random, dmin: int, dmax: int) -> int:
    return random_even_with_digits(rng, rng.randint(dmin, dmax))

def try_goldbach_with_tiny_budget(n: int, rng: random.Random, stats: Stats) -> Optional[Tuple[int,int,int]]:
    for p in CONFIG["SUBTRACTORS"]:
        q = n - p
        if q > 1 and mr(q, rng, stats):
            return (n, p, q)
    for d in CONFIG["OFFSETS"]:
        m = n + d
        if m < 4 or m % 2: continue
        for p in CONFIG["SUBTRACTORS"]:
            q = m - p
            if q > 1 and mr(q, rng, stats):
                return (m, p, q)
    return None

@dataclass
class Cc:
    value: float = 0.0
    def bump(self, eps: float) -> None:
        self.value = min(1.0, self.value + eps)

def short_tag(*ints: int) -> str:
    h = hashlib.sha256(".".join(map(str,ints)).encode()).hexdigest()
    return h[:12]

def run_once(steps: int, dmin: int, dmax: int, epsilon: float, seed: int) -> None:
    rng = random.Random(seed)
    cc  = Cc(0.0)
    hits = 0
    examples: List[Tuple[int,int,int,str,int]] = []
    total_stats = Stats()
    t0 = time.time()

    for k in range(steps):
        n = sample_even_in_digit_range(rng, dmin, dmax)
        found = try_goldbach_with_tiny_budget(n, rng, total_stats)
        if found:
            m, p, q = found
            hits += 1
            cc.bump(epsilon)
            if len(examples) < CONFIG["PRINT_EXAMPLES"]:
                tag = short_tag(m, p, q)
                examples.append((k, m, p, tag, len(str(q))))

    elapsed = time.time() - t0

    print("=== Minimal PoC Summary ===")
    print(f"mode: RANGE   steps: {steps}   digits: [{dmin}, {dmax}]   epsilon: {epsilon}   seed: {seed}")
    print(f"tiny-budget Goldbach hits: {hits}  (hit rate: {hits/steps:.3f})")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"final Cc: {cc.value:.6f}  (only increases on witnessed hits; bounded misses unpenalized)")

    if CONFIG["SHOW_STATS"]:
        print("\n--- Search exposure ---")
        print(f"primality tests attempted: {total_stats.primality_tests_attempted}")
        print(f"  trial-division rejects:  {total_stats.trial_div_composite_rejects}")
        print(f"  Miller–Rabin calls:      {total_stats.mr_calls}")
        print(f"  MR composite rejects:    {total_stats.mr_composite_rejects}")
        print(f"  MR rounds total:         {total_stats.mr_rounds_total}")
        print(f"  probable primes seen:    {total_stats.probable_primes}")

    if examples:
        print("\nexample hits (k, m, p, tag, digits(q))  # tag = sha256(m.p.q)[:12] for audit")
        for rec in examples:
            print(rec)

def run_exact(steps: int, digits: int, epsilon: float, seed: int) -> None:
    print(f"=== EXACT digits={digits} ===")
    run_once(steps, digits, digits, epsilon, seed)

def run_sweep(steps: int, digits_list: List[int], epsilon: float, seed: int) -> None:
    print(f"=== SWEEP over digits {digits_list} ===")
    for d in digits_list:
        run_once(steps, d, d, epsilon, seed + d)  # vary seed per digit to decorrelate

# ------------------ CLI (optional; overrides CONFIG) --------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tiny-budget Goldbach sampler with top-level CONFIG.")
    ap.add_argument("--steps", type=int, help="number of sampled n (per run or per sweep digit)")
    ap.add_argument("--mode", choices=["RANGE","EXACT","SWEEP_DIGITS"])
    ap.add_argument("--min-digits", type=int)
    ap.add_argument("--max-digits", type=int)
    ap.add_argument("--digits", type=int, help="exact digit length if MODE=EXACT")
    ap.add_argument("--epsilon", type=float)
    ap.add_argument("--seed", type=int)
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    steps   = args.steps   if args.steps   is not None else CONFIG["STEPS"]
    mode    = args.mode    if args.mode    is not None else CONFIG["MODE"]
    dmin    = args.min_digits if args.min_digits is not None else CONFIG["MIN_DIGITS"]
    dmax    = args.max_digits if args.max_digits is not None else CONFIG["MAX_DIGITS"]
    ed      = args.digits  if args.digits  is not None else CONFIG["EXACT_DIGITS"]
    eps     = args.epsilon if args.epsilon is not None else CONFIG["EPSILON"]
    seed    = args.seed    if args.seed    is not None else CONFIG["SEED"]

    if mode == "RANGE":
        run_once(steps, dmin, dmax, eps, seed)
    elif mode == "EXACT":
        run_exact(steps, ed, eps, seed)
    elif mode == "SWEEP_DIGITS":
        run_sweep(steps, CONFIG["SWEEP_DIGITS"], eps, seed)
    else:
        print(f"Unknown MODE: {mode}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
