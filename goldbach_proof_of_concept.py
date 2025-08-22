#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goldbach_proof_of_concept.py

Context-Completeness (Cc) + Bounded Goldbach Probe (PoC)

This operationalizes the Verification Asymmetry:
- Tiny, context-bounded successes nudge Cc upward by a small epsilon.
- Bounded non-findings do NOT penalize belief.
- Only a VALIDATED counterexample (deterministic small-n branch) collapses belief to ~0.
- Additionally, we perform a **budgeted collapse** when either:
  (A) the sampled n ever exceeds 4 * 10^90, or
  (B) after 100 steps have completed (by default).

**Tweaks implemented for clarity:**
- Defaults now make movement visible in demos:
  --epsilon 0.002, --Pmax 2000000, --samples 20000
- Defaults now allow max-n collapse to trigger:
  --max-digits 100 (so n can exceed 4*10^90 in practice)

IMPORTANT NOTE:
If the collapse weren’t budgeted, completing exhaustive search or finding a true
counterexample at these magnitudes would likely take longer than the age of the universe.
This demo is about belief updates for reasoning about computationally absurd numbers under strict computational bounds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import random, csv, argparse, sys

# ---------------------------------------------------------------------
#  Deterministic primality for small integers (<= 2^64)
#  and probabilistic Miller–Rabin for general use
# ---------------------------------------------------------------------

_SMALL_PRIMES: List[int] = [2,3,5,7,11,13,17,19,23,29,31,37]

def _decompose(n: int) -> Tuple[int, int]:
    d = n - 1; s = 0
    while d % 2 == 0:
        d //= 2; s += 1
    return d, s

def _mr_witness(a: int, d: int, n: int, s: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True

def is_probable_prime(n: int, rounds: int = 12, rng: Optional[random.Random] = None) -> bool:
    """Probabilistic Miller–Rabin (strong)."""
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0: return False
    d, s = _decompose(n)
    if rng is None: rng = random.Random(0xC0FFEE)
    max_base = n - 2
    for _ in range(rounds):
        a = rng.randrange(2, max_base + 1) if max_base >= 2 else 2
        if _mr_witness(a, d, n, s):
            return False
    return True

def is_prime_det_64(n: int) -> bool:
    """Deterministic primality for 64-bit integers using known MR bases."""
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0 and n != p: return False
    # bases sufficient for n < 2^64
    bases = [2, 3, 5, 7, 11, 13, 17]
    d, s = _decompose(n)
    for a in bases:
        if a % n == 0:
            return True
        if _mr_witness(a, d, n, s):
            return False
    return True

# ---------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------

def random_even_with_digits(rng: random.Random, d: int) -> int:
    """Generate an even integer with exactly d digits."""
    if d <= 0: raise ValueError("digits must be >= 1")
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(d-1)]
    n = int(str(first) + "".join(map(str, rest)))
    if n % 2: n += 1
    return n

def sample_even_n(rng: random.Random, min_digits: int, max_digits: int) -> int:
    d = rng.randint(min_digits, max_digits)
    return random_even_with_digits(rng, d)

# ---------------------------------------------------------------------
#  Bounded Goldbach search (verification effort is intentionally tiny)
# ---------------------------------------------------------------------

def bounded_goldbach_pair_random(m: int, Pmax: int, samples: int, rng: random.Random, mr_rounds: int) -> Optional[Tuple[int,int]]:
    """
    Try to find (p,q) with p random odd <= Pmax, p prime and q=m-p prime, using probabilistic MR.
    We only take 'samples' random draws for p. If none succeed, return None.
    """
    if m % 2 != 0 or m < 4: return None
    for _ in range(samples):
        p = rng.randrange(3, Pmax+1, 2)
        if is_probable_prime(p, rounds=mr_rounds, rng=rng):
            q = m - p
            if q > 1 and is_probable_prime(q, rounds=mr_rounds, rng=rng):
                return (p, q)
    return None

# ---------------------------------------------------------------------
#  Context-Completeness (Cc) policy and state
# ---------------------------------------------------------------------

@dataclass
class CCPolicy:
    epsilon: float = 2e-3   # small bump on bounded success (tweak: larger default)
    max_cc: float = 1.0

@dataclass
class CCState:
    value: float = 0.0
    def bump(self, eps: float, max_cc: float):
        self.value = min(max_cc, self.value + eps)

# ---------------------------------------------------------------------
#  Counterexample validation (small-n deterministic branch)
# ---------------------------------------------------------------------

def sieve_upto(limit: int) -> List[int]:
    if limit < 2: return []
    s = bytearray(b"\x01") * (limit + 1)
    s[0:2] = b"\x00\x00"
    r = int(limit ** 0.5)
    for p in range(2, r + 1):
        if s[p]:
            step_count = ((limit - p*p)//p) + 1
            s[p*p:limit+1:p] = b"\x00" * step_count
    return [i for i,b in enumerate(s) if b]

def validate_counterexample_small_n(n: int, small_limit: int) -> Optional[bool]:
    """
    For even n <= small_limit, exhaustively test small p primes (deterministic up to 2^64 for q).
    Returns:
      True  -> proven NO pair (counterexample) for this n (collapses belief)
      False -> found a pair (not a counterexample)
      None  -> n too large for this method
    """
    if n % 2 or n < 4: return False
    if n > small_limit: return None
    primes = sieve_upto(n // 2)
    for p in primes:
        q = n - p
        if q > 1 and is_prime_det_64(q):
            return False
    return True

# ---------------------------------------------------------------------
#  Main probe loop with budgeted collapse rules
# ---------------------------------------------------------------------

def run_probes(trials: int,
               min_digits: int,
               max_digits: int,
               offsets: List[int],
               Pmax: int,
               samples: int,
               epsilon: float,
               small_n_limit: int,
               mr_rounds: int,
               seed: int,
               out_path: str,
               max_n_collapse: int,
               max_steps_collapse: int,
               collapse_value: float = 1e-9) -> Dict[str, Any]:

    rng = random.Random(seed)
    cc = CCState(0.0)
    policy = CCPolicy(epsilon=epsilon, max_cc=1.0)

    rows: List[Dict[str, Any]] = []
    success_count = 0
    ce_proven = False
    ce_n: Optional[int] = None
    collapsed = False
    collapse_reason: Optional[str] = None

    for t in range(trials):
        # Budgeted collapse after completing max_steps_collapse steps
        if not collapsed and (t >= max_steps_collapse):
            collapsed = True
            collapse_reason = f"budgeted_collapse_max_steps({max_steps_collapse})"
            cc.value = collapse_value  # collapse belief
        # Record a collapsed row directly (no more probing)
        if collapsed:
            rec = {
                "trial": t,
                "n_digits": None,
                "n_head": None,
                "success": False,
                "m": None,
                "p": None,
                "q_head": None,
                "Cc_before": cc.value,
                "Cc_after": cc.value,
                "event": "collapsed_budget",
                "collapse_reason": collapse_reason
            }
            rows.append(rec)
            continue

        # Sample n and check the "max n" budgeted collapse rule
        n = sample_even_n(rng, min_digits, max_digits)
        if (not collapsed) and (n > max_n_collapse):
            collapsed = True
            collapse_reason = f"budgeted_collapse_max_n(>{max_n_collapse})"
            cc.value = collapse_value
            # Log the collapse event row for this trial
            rec = {
                "trial": t,
                "n_digits": len(str(n)),
                "n_head": str(n)[:12]+"...",
                "success": False,
                "m": None,
                "p": None,
                "q_head": None,
                "Cc_before": cc.value,
                "Cc_after": cc.value,
                "event": "collapsed_budget",
                "collapse_reason": collapse_reason
            }
            rows.append(rec)
            continue

        rec = {
            "trial": t,
            "n_digits": len(str(n)),
            "n_head": str(n)[:12]+"...",
            "success": False,
            "m": None,
            "p": None,
            "q_head": None,
            "Cc_before": cc.value,
            "Cc_after": None,
            "event": "none",
            "collapse_reason": None
        }

        # Deterministic small-n counterexample check
        ce = validate_counterexample_small_n(n, small_n_limit)
        if ce is True:
            ce_proven = True
            ce_n = n
            cc.value = collapse_value
            rec["event"] = "counterexample_confirmed_small_n"
            rec["Cc_after"] = cc.value
            rows.append(rec)
            # After a real counterexample we treat belief as collapsed.
            collapsed = True
            collapse_reason = "true_counterexample_small_n"
            continue
        elif ce is False:
            rec["event"] = "small_n_pair_found"

        # Try nearby even m with bounded random p-sampling
        found = False
        for d in offsets:
            m = n + d
            if m % 2: continue
            pair = bounded_goldbach_pair_random(m, Pmax=Pmax, samples=samples, rng=rng, mr_rounds=mr_rounds)
            if pair:
                p, q = pair
                success_count += 1
                cc.bump(policy.epsilon, policy.max_cc)
                rec.update({
                    "success": True,
                    "m": m,
                    "p": p,
                    "q_head": str(q)[:16]+"...",
                    "event": "bounded_success",
                    "Cc_after": cc.value
                })
                found = True
                break

        if not found and rec["Cc_after"] is None:
            # bounded non-finding: do NOT penalize
            rec["event"] = "bounded_failure_no_penalty"
            rec["Cc_after"] = cc.value

        rows.append(rec)

    # write CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return {
        "trials": trials,
        "successes": success_count,
        "final_Cc": cc.value,
        "counterexample_proven": ce_proven,
        "counterexample_n": ce_n,
        "collapsed": collapsed,
        "collapse_reason": collapse_reason,
        "out_csv": out_path
    }

# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Context-Completeness (Cc) + Bounded Goldbach Probe (PoC) with budgeted collapse.")
    ap.add_argument("--trials", type=int, default=120, help="number of independent n samples (steps)")
    ap.add_argument("--min-digits", type=int, default=20, help="min digits for n")
    ap.add_argument("--max-digits", type=int, default=100, help="max digits for n")  # tweak for potential max-n collapse
    ap.add_argument("--Pmax", type=int, default=2_000_000, help="max p to sample (random odd <= Pmax)")  # tweak (bigger)
    ap.add_argument("--samples", type=int, default=20000, help="random p samples per m")  # tweak (bigger)
    ap.add_argument("--epsilon", type=float, default=0.002, help="Cc bump on bounded success")  # tweak (bigger)
    ap.add_argument("--small-n-limit", type=int, default=2_000_000, help="deterministic small-n counterexample ceiling")
    ap.add_argument("--mr-rounds", type=int, default=12, help="Miller–Rabin rounds for probable primality")
    ap.add_argument("--seed", type=int, default=2025, help="PRNG seed")
    ap.add_argument("--out", type=str, default="cc_goldbach_probes.csv", help="CSV output path")
    # Budgeted collapse controls (defaults per request)
    ap.add_argument("--max-n-collapse", type=int, default=4 * (10 ** 90), help="collapse immediately if n > this")
    ap.add_argument("--max-steps-collapse", type=int, default=100, help="collapse after this many completed steps")
    ap.add_argument("--collapse-value", type=float, default=1e-9, help="belief value when collapsed")
    return ap.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    offsets = [-1, +1, -3, +3, -7, +7, -11, +11, -13, +13, -17, +17, -19, +19]
    res = run_probes(
        trials=args.trials,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        offsets=offsets,
        Pmax=args.Pmax,
        samples=args.samples,
        epsilon=args.epsilon,
        small_n_limit=args.small_n_limit,
        mr_rounds=args.mr_rounds,
        seed=args.seed,
        out_path=args.out,
        max_n_collapse=args.max_n_collapse,
        max_steps_collapse=args.max_steps_collapse,
        collapse_value=args.collapse_value
    )
    print("=== Summary ===")
    for k,v in res.items():
        print(f"{k}: {v}")
    if res["collapsed"]:
        print("\nNOTE:")
        print("This run collapsed by budget (max-n or max-steps). If collapse weren’t budgeted,")
        print("exhaustively resolving cases at these magnitudes would likely take longer than the")
        print("age of the universe. This demo showcases belief updates under strict bounds.")
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
