#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, random, time, math, sys
from typing import List, Tuple, Set, Dict

# ---------------- Primes utilities ----------------

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

# ---------------- Miller–Rabin ----------------

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

# ---------------- Sampling & engine ----------------

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
    """Return all unordered Goldbach decompositions found via the subtractor scan.
       Deduplicates by unordered pair. If cap_per_n is set, record at most that many (still scans all p)."""
    if n % 2: n -= 1
    seen: Set[Tuple[int,int]] = set()
    recorded: List[Tuple[int,int]] = []
    for p in subtractors:
        q = n - p
        if q < 2: continue
        # parity: with p odd and n even, q is odd
        if q != 2 and (q & 1) == 0:  # even > 2 ⇒ composite (kept from original; effectively never fires)
            continue
        if is_probable_prime(q, rounds=mr_rounds, rng=rng):
            a, b = (p, q) if p <= q else (q, p)
            if (a, b) not in seen:
                seen.add((a, b))
                if cap_per_n is None or len(recorded) < cap_per_n:
                    recorded.append((a, b))
    return recorded

# ---------------- CLI & main ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Goldbach PoC: collect multiple decomps per n without early stopping.")
    ap.add_argument("--trials", type=int, default=1000, help="number of random tests to run (default: 1000)")
    ap.add_argument("--digits", type=int, default=30, help="number of digits in n (even numbers only; default: 30)")
    ap.add_argument("--subtractors", type=int, default=40, help="how many odd primes to subtract (10..100; default: 40)")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed (default: 2025)")
    ap.add_argument("--mr-rounds", type=int, default=12, help="Miller–Rabin rounds per candidate (default: 12)")
    ap.add_argument("--max-per-n", type=int, default=None, help="cap on how many decomps to record per n (scan still runs fully)")
    ap.add_argument("--print-examples", type=int, default=0, help="print up to K example decomps for the first few n with any hits")
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

    t0 = time.time()
    total_hits = 0
    total_decomps = 0
    per_n_counts: List[int] = []
    example_counter = 0
    # FIX: corrected attribute name (was args.print-examples)
    example_limit = int(max(0, args.print_examples)) if isinstance(args.print_examples, int) else 0  # guard

    # We will store up to 'print-examples' example entries (n, decomps list)
    examples: List[Tuple[int, List[Tuple[int,int]]]] = []

    for _ in range(args.trials):
        n = random_even_with_digits(rng, args.digits)
        decomps = goldbach_decomps(n, subtractor_primes, args.mr_rounds, rng, cap_per_n=args.max_per_n)
        c = len(decomps)
        if c > 0:
            total_hits += 1
            total_decomps += c
            per_n_counts.append(c)
            if example_counter < example_limit:
                examples.append((n, decomps))
                example_counter += 1
        else:
            per_n_counts.append(0)

    elapsed = time.time() - t0
    per_n = elapsed / args.trials if args.trials else float('nan')
    hit_rate = total_hits / args.trials if args.trials else float('nan')
    mean_decomps = (total_decomps / total_hits) if total_hits else 0.0
    max_decomps = max(per_n_counts) if per_n_counts else 0
    min_decomps = min(per_n_counts) if per_n_counts else 0

    # Report
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
            # FIX: proper f-string quoting
            shown = ", ".join(f"({p},{q})" for p, q in ds)
            print(f"{n} → {shown}")

if __name__ == '__main__':
    main()
