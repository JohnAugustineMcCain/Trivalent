#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from typing import List, Tuple, Set, Dict

# ---------------- Utilities & small primes ----------------

def sieve_upto(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(n**0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            step = p
            sieve[start:n + 1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]

def first_k_odd_primes(k: int) -> List[int]:
    if k <= 0:
        return []
    # crude overestimate to ensure enough primes
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
        limit *= 2

# ---------------- Miller–Rabin (probable primality) ----------------

def _dec(n: int) -> Tuple[int, int]:
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return d, s

def _mr_witness(a: int, d: int, n: int, s: int) -> bool:
    x = pow(a, d, n)
    if x in (1, n - 1):
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True

def is_probable_prime(n: int, rounds: int = 6, rng: random.Random | None = None) -> bool:
    if n < 2:
        return False
    # quick trial division by small primes up to 97
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = _dec(n)
    if rng is None:
        rng = random.Random(2025)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        if _mr_witness(a, d, n, s):
            return False
    return True

# ---------------- Sampling ----------------

def random_even_with_digits(rng: random.Random, d: int) -> int:
    if d < 1:
        raise ValueError("digits must be >= 1")
    if d == 1:
        return rng.choice([2, 4, 6, 8])
    first = rng.randint(1, 9)
    middle = ''.join(str(rng.randint(0, 9)) for _ in range(d - 2))
    last = rng.choice("02468")
    return int(str(first) + middle + last)

# ---------------- Goldbach engines ----------------

def goldbach_subtractor_scan(
    n: int,
    subtractors: List[int],
    mr_rounds: int,
    rng: random.Random,
    cap_per_n: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Return up to cap_per_n unordered decompositions found by scanning fixed subtractor primes.
    Keeps scanning (no early stop) and deduplicates unordered pairs.
    """
    if n % 2:
        n -= 1
    seen: Set[Tuple[int, int]] = set()
    recorded: List[Tuple[int, int]] = []
    for p in subtractors:
        q = n - p
        if q < 2:
            continue
        # parity: q is odd unless q == 2; skip even q > 2 quickly
        if q != 2 and (q & 1) == 0:
            continue
        if is_probable_prime(q, rounds=mr_rounds, rng=rng):
            a, b = (p, q) if p <= q else (q, p)
            if (a, b) not in seen:
                seen.add((a, b))
                recorded.append((a, b))
                if cap_per_n is not None and len(recorded) >= cap_per_n:
                    break
    return recorded

# ---------------- Heuristics (Hardy–Littlewood style, baseline) ----------------

def hl_expected_reps(n: int) -> float:
    """
    Baseline expected number of Goldbach representations ~ n / (ln n)^2.
    (Singular series omitted for simplicity; sufficient for trend-checking.)
    """
    if n < 6:
        return 0.0
    ln = math.log(n)
    return float(n) / (ln * ln)

def heuristic_success_prob(n: int) -> float:
    """
    Tiny-budget success probability proxy:
    p_hit ≈ 1 - exp(-(1/ln(n/2))^2 * sqrt(n)/ln n).
    This increases with n, reflecting that expected reps grow ~ n/(ln n)^2.
    """
    if n < 6:
        return 0.0
    ln = math.log(n)
    ln_half = math.log(n / 2.0)
    p = 1.0 / (ln_half * ln_half)  # local prime density near n/2 squared
    k = math.sqrt(n / ln)          # crude effective trials in a narrow band
    x = -p * k
    if x < -700:                   # clamp to avoid underflow in exp
        return 1.0
    return max(0.0, min(1.0, 1.0 - math.exp(x)))

# ---------------- Cc-like tracker ----------------

class CC:
    def __init__(self, value: float = 0.0):
        self.value = float(value)
    def bump(self, eps: float) -> None:
        self.value = min(1.0, self.value + eps)

# ---------------- CLI & main ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Goldbach PoC with heuristic + verification modes.\n"
            "Modes:\n"
            "  verify:    subtractor-scan verification only\n"
            "  heuristic: heuristic expectations only (works at huge n)\n"
            "  hybrid:    heuristics + tiny scan; track agreement (Cc)\n"
        )
    )
    ap.add_argument("--mode", choices=["verify", "heuristic", "hybrid"], default="hybrid")
    ap.add_argument("--trials", type=int, default=500, help="number of random tests to run (default: 500)")
    ap.add_argument("--digits", type=int, default=30, help="number of digits in n (even numbers; default: 30)")
    ap.add_argument("--subtractors", type=int, default=40, help="how many odd primes to subtract (10..200; default: 40)")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed (default: 2025)")
    ap.add_argument("--mr-rounds", type=int, default=6, help="Miller–Rabin rounds per candidate (default: 6)")
    ap.add_argument("--cap-per-n", type=int, default=1, help="max decomps to record per n in scan (default: 1)")
    ap.add_argument("--eps", type=float, default=0.002, help="Cc bump per agreement (default: 0.002)")
    ap.add_argument("--print-examples", type=int, default=0, help="show up to K example n with results")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    if not (10 <= args.subtractors <= 200):
        print(f"error: --subtractors must be between 10 and 200 (got {args.subtractors})", file=sys.stderr)
        sys.exit(2)
    if args.digits < 1:
        print(f"error: --digits must be >= 1 (got {args.digits})", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    subtractor_primes = first_k_odd_primes(args.subtractors)

    t0 = time.time()

    # Verification stats
    total_hits = 0                # n with ≥1 decomp (verification/hybrid only)
    total_decomps = 0
    per_n_counts: List[int] = []
    example_counter = 0
    example_limit = max(0, int(args.print_examples))

    # Heuristic stats
    h_sum_expected = 0.0
    h_sum_success_prob = 0.0
    h_high_conf = 0

    # Agreement / Cc
    cc = CC(0.0)
    agreements = 0
    trials = args.trials

    examples: List[Tuple[int, Dict[str, object]]] = []

    for _ in range(trials):
        n = random_even_with_digits(rng, args.digits)

        # Heuristics
        H_E = hl_expected_reps(n)
        H_P = heuristic_success_prob(n)

        if args.mode in ("heuristic", "hybrid"):
            h_sum_expected += H_E
            h_sum_success_prob += H_P
            if H_P >= 0.8:
                h_high_conf += 1

        rec: Dict[str, object] = {"n": n, "H_E": H_E, "H_P": H_P}

        # Verification / hybrid
        if args.mode in ("verify", "hybrid"):
            decomps = goldbach_subtractor_scan(
                n, subtractor_primes, args.mr_rounds, rng, cap_per_n=args.cap_per_n
            )
            c = len(decomps)
            per_n_counts.append(c)
            total_decomps += c
            if c > 0:
                total_hits += 1
                rec["decomps"] = decomps
                rec["hit"] = True
            else:
                rec["decomps"] = []
                rec["hit"] = False

            if args.mode == "hybrid":
                # An "agreement": heuristic predicts likely success (H_P >= 0.6) and we hit, or predicts unlikely and we miss
                pred_hit = (H_P >= 0.6)
                if (c > 0) == pred_hit:
                    agreements += 1
                    cc.bump(args.eps)

        if example_counter < example_limit:
            examples.append((n, rec))
            example_counter += 1

    elapsed = time.time() - t0
    per_n_time = elapsed / trials if trials else float("nan")

    # Summaries
    print("=== Goldbach Proof of Concept (heuristic-aware) ===")
    print(f"mode: {args.mode}   trials: {trials}   digits(n): {args.digits}")
    print(f"subtractors: {args.subtractors}   mr_rounds: {args.mr_rounds}   cap_per_n: {args.cap_per_n}")
    print(f"seed: {args.seed}")
    print(f"elapsed: {elapsed:.6f} s   per_n: {per_n_time*1000:.3f} ms")

    if args.mode in ("heuristic", "hybrid"):
        avg_E = h_sum_expected / trials if trials else 0.0
        avg_P = h_sum_success_prob / trials if trials else 0.0
        print("\n--- Heuristic summary ---")
        print(f"avg expected reps E[R(n)] ~ n/(ln n)^2 : {avg_E:.3f}")
        print(f"avg success-prob proxy (tiny-budget)    : {avg_P:.3f}")
        print(f"high-confidence cases (H_P >= 0.8)      : {h_high_conf}/{trials}")

    if args.mode in ("verify", "hybrid"):
        hit_rate = total_hits / trials if trials else float("nan")
        max_decomps = max(per_n_counts) if per_n_counts else 0
        min_decomps = min(per_n_counts) if per_n_counts else 0
        mean_decomps_per_hit = (total_decomps / total_hits) if total_hits else 0.0
        print("\n--- Verification (subtractor scan) ---")
        print(f"n with ≥1 decomp: {total_hits}   hit_rate: {hit_rate:.3f}")
        print(f"total decomps recorded: {total_decomps}")
        print(f"per-hit mean decomps: {mean_decomps_per_hit:.3f}   per-n min/max (over all n): {min_decomps}/{max_decomps}")

    if args.mode == "hybrid":
        agree_rate = agreements / trials if trials else 0.0
        print("\n--- Agreement & Cc-like metric ---")
        print(f"agreements (heuristic vs verification): {agreements}/{trials}   agree_rate: {agree_rate:.3f}")
        print(f"final Cc (bumped by eps on agreement): {cc.value:.6f}   eps: {args.eps}")

    # Examples
    if examples:
        print("\n--- examples (first few) ---")
        for n, rec in examples:
            if args.mode == "heuristic":
                print(f"{n}  H_E={rec['H_E']:.3f}  H_P={rec['H_P']:.3f}")
            elif args.mode == "verify":
                ds = rec["decomps"]
                note = "HIT" if rec.get("hit") else "miss"
                shown = ", ".join(f"({p},{q})" for (p, q) in ds) if ds else ""
                print(f"{n}  {note}  decomps: {shown}")
            else:
                ds = rec["decomps"]
                note = "HIT" if rec.get("hit") else "miss"
                tail = f" with {', '.join(f'({p},{q})' for (p,q) in ds)}" if ds else ""
                print(f"{n}  H_E={rec['H_E']:.3f}  H_P={rec['H_P']:.3f}  -> {note}{tail}")

if __name__ == "__main__":
    main()
