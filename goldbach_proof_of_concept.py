#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goldbach_proof_of_concept.py  —  Minimal PoC

Claim illustrated:
We can treat ambiguity as a core feature of reality/computation and still reason
asymptotically about computationally impossible spaces by using:
Tiny, bounded verification (no exhaustive search),
Asymmetric belief updates (successes nudge up; bounded failures don’t penalize).

Mechanics (very small, on purpose):
- Sample an even n by digit-length (e.g., 19..100 digits ~ 4e18..1e100).
- Try to find n = p + q with a fixed set of subtractor primes p.
- If that fails, try a few nearby even numbers (± small offsets) with the same tiny set.
- If a (p,q) is found, increment belief Cc by epsilon (clamped to ≤ 1).
- If not found within the tiny budget, do nothing (bounded ignorance ≠ evidence against).

Note:
At these magnitudes, exhaustive resolution would take longer than the age of the universe.
"""

from __future__ import annotations
import argparse, random
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- tiny Miller–Rabin (probabilistic) ---------------------------------

_SMALL = [2,3,5,7,11,13,17,19,23,29,31,37]

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

def is_probable_prime(n: int, rounds: int, rng: random.Random) -> bool:
    if n < 2: return False
    for p in _SMALL:
        if n == p: return True
        if n % p == 0: return False
    d,s = _dec(n)
    for _ in range(rounds):
        a = rng.randrange(2, n-1)
        if _mr_witness(a, d, n, s): return False
    return True

def mr(n: int, rng: random.Random) -> bool:
    # small, fixed rounds; we are just demoing
    digits = len(str(n))
    rounds = 8 if digits <= 18 else (6 if digits <= 30 else 4)
    return is_probable_prime(n, rounds, rng)

# --- sampling & tiny Goldbach checks -----------------------------------

def random_even_with_digits(rng: random.Random, d: int) -> int:
    if d < 1: raise ValueError("digits must be >= 1")
    first = rng.randint(1,9)
    rest = [rng.randint(0,9) for _ in range(d-1)]
    n = int(str(first) + "".join(map(str, rest)))
    return n if n % 2 == 0 else n + 1

def sample_even_in_digit_range(rng: random.Random, dmin: int, dmax: int) -> int:
    return random_even_with_digits(rng, rng.randint(dmin, dmax))

SUBTRACTORS = [3,5,7,11,13,17,19,23,29]                 # tiny, fixed set
OFFSETS     = [-22,-14,-6,-2,0,2,6,14,22]               # small even neighborhood

def try_goldbach_with_tiny_budget(n: int, rng: random.Random) -> Optional[Tuple[int,int,int]]:
    """Return (m,p,q) if found with tiny budget; else None. m==n or nearby even."""
    # exact n first
    for p in SUBTRACTORS:
        q = n - p
        if q > 1 and mr(q, rng):
            return (n, p, q)
    # then nearby even m
    for d in OFFSETS:
        m = n + d
        if m < 4 or m % 2: continue
        for p in SUBTRACTORS:
            q = m - p
            if q > 1 and mr(q, rng):
                return (m, p, q)
    return None

# --- minimal Cc state (asymmetric) -------------------------------------

@dataclass
class Cc:
    value: float = 0.0
    def bump(self, eps: float) -> None:
        self.value = min(1.0, self.value + eps)

# --- main loop ----------------------------------------------------------

def run(steps: int, dmin: int, dmax: int, epsilon: float, seed: int) -> None:
    rng = random.Random(seed)
    cc  = Cc(0.0)
    hits = 0
    examples: List[Tuple[int,int,int,int]] = []  # (k, m, p, q_head)

    for k in range(steps):
        n = sample_even_in_digit_range(rng, dmin, dmax)
        found = try_goldbach_with_tiny_budget(n, rng)
        if found:
            m, p, q = found
            hits += 1
            cc.bump(epsilon)
            if len(examples) < 8:
                # only show head of q to keep logs tidy
                q_head = int(str(q)[:16])
                examples.append((k, m, p, q_head))

    # Summary for papers: short, clear, asymmetric update emphasized
    print("=== Minimal PoC Summary ===")
    print(f"steps: {steps}   digits: [{dmin}, {dmax}]   epsilon: {epsilon}")
    print(f"tiny-budget Goldbach hits: {hits}")
    print(f"final Cc: {cc.value:.6f}  (only increases on witnessed hits; bounded misses unpenalized)")
    if examples:
        print("\nexample hits (k, m, p, q_head):")
        for rec in examples:
            print(rec)

# --- CLI ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal PoC: asymmetric belief under tiny bounded Goldbach checks.")
    ap.add_argument("--steps", type=int, default=200, help="number of sampled n")
    ap.add_argument("--min-digits", type=int, default=19, help="min digits for n (19 ~ 4e18)")
    ap.add_argument("--max-digits", type=int, default=100, help="max digits for n (up to 1e100)")
    ap.add_argument("--epsilon", type=float, default=0.002, help="Cc bump per witnessed hit")
    ap.add_argument("--seed", type=int, default=2025, help="PRNG seed")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.steps, args.min_digits, args.max_digits, args.epsilon, args.seed)
