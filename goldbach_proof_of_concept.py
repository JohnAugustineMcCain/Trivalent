#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, random, hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- small primes via sieve (for fast trial division) -------------------

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

_SMALL_TRIAL = _primes_upto(1000)  # primes ≤ 1000 for quick elimination

# --- stats ---------------------------------------------------------------

@dataclass
class Stats:
    primality_tests_attempted: int = 0
    trial_div_composite_rejects: int = 0
    mr_calls: int = 0
    mr_composite_rejects: int = 0
    mr_rounds_total: int = 0
    probable_primes: int = 0

# --- tiny Miller–Rabin (probabilistic) ----------------------------------

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

def is_probable_prime(n: int, rounds: int, rng: random.Random, stats: Optional[Stats]=None) -> bool:
    if stats: stats.primality_tests_attempted += 1
    if n < 2:
        return False
    # trial division first
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
    if digits <= 18: rounds = 8
    elif digits <= 40: rounds = 10
    elif digits <= 80: rounds = 12
    else: rounds = 14
    return is_probable_prime(n, rounds, rng, stats)

# --- sampling & tiny Goldbach checks -----------------------------------

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

SUBTRACTORS = [3,5,7,11,13,17,19,23,29]
OFFSETS     = [-22,-14,-6,-2,2,6,14,22]  # removed 0

def try_goldbach_with_tiny_budget(n: int, rng: random.Random, stats: Stats) -> Optional[Tuple[int,int,int]]:
    for p in SUBTRACTORS:
        q = n - p
        if q > 1 and mr(q, rng, stats):
            return (n, p, q)
    for d in OFFSETS:
        m = n + d
        if m < 4 or m % 2: continue
        for p in SUBTRACTORS:
            q = m - p
            if q > 1 and mr(q, rng, stats):
                return (m, p, q)
    return None

# --- minimal Cc state (asymmetric) -------------------------------------

@dataclass
class Cc:
    value: float = 0.0
    def bump(self, eps: float) -> None:
        self.value = min(1.0, self.value + eps)

# --- helpers ------------------------------------------------------------

def short_tag(*ints: int) -> str:
    h = hashlib.sha256(".".join(map(str,ints)).encode()).hexdigest()
    return h[:12]

# --- main loop ----------------------------------------------------------

def run(steps: int, dmin: int, dmax: int, epsilon: float, seed: int) -> None:
    rng = random.Random(seed)
    cc  = Cc(0.0)
    hits = 0
    examples: List[Tuple[int,int,int,str,int]] = []
    total_stats = Stats()

    for k in range(steps):
        n = sample_even_in_digit_range(rng, dmin, dmax)
        found = try_goldbach_with_tiny_budget(n, rng, total_stats)
        if found:
            m, p, q = found
            hits += 1
            cc.bump(epsilon)
            if len(examples) < 8:
                tag = short_tag(m, p, q)
                examples.append((k, m, p, tag, len(str(q))))

    print("=== Minimal PoC Summary (Updated) ===")
    print(f"steps: {steps}   digits: [{dmin}, {dmax}]   epsilon: {epsilon}   seed: {seed}")
    print(f"tiny-budget Goldbach hits: {hits}  (hit rate: {hits/steps:.3f})")
    print(f"final Cc: {cc.value:.6f}  (only increases on witnessed hits; bounded misses unpenalized)")
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

# --- CLI ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal PoC (updated): asymmetric belief under tiny bounded Goldbach checks.")
    ap.add_argument("--steps", type=int, default=60, help="number of sampled n")
    ap.add_argument("--min-digits", type=int, default=19, help="min digits for n")
    ap.add_argument("--max-digits", type=int, default=30, help="max digits for n")
    ap.add_argument("--epsilon", type=float, default=0.002, help="Cc bump per witnessed hit")
    ap.add_argument("--seed", type=int, default=2025, help="PRNG seed")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.steps, args.min_digits, args.max_digits, args.epsilon, args.seed)
