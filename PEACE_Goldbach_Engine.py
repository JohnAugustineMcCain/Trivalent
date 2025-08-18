#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEACE Goldbach Engine
---------------------

A pragmatic, PEACE-aligned, trivalent/fuzzy confidence engine for the claim:

    Goldbach: "Every even n ≥ 4 is the sum of two primes."

Core ideas implemented (from the PEACE framework):
- Default neutral truth: B ("Both-ish") rather than hard Unknown.
- Perspectives are load-bearing only when they actually change evaluation.
- Evidence is designated when it carries true-content; absence of quick evidence is epistemically neutral.
- No explosion: contradictory or paradox-laden contexts don’t contaminate everything else.

Heuristic policy:
- For exponentially growing n (with semi-random jumps), try to "exhibit" a Goldbach decomposition quickly:
  1) Try small subtractor primes p (e.g., 3,5,7,11,13,17,19,23,29); if q = n - p is prime, we have an exact certificate.
  2) Otherwise, try small even offsets δ ∈ {±2, ±6, ±14, ±22} to find a nearby even m = n + δ that certifies.
  3) If still no certificate, record a "miss" and move on (no confidence decrease).

Confidence update:
- Start at 0.5 (Both-ish). Never hit exactly 0 or 1.
- Evidence below T = 4e18 yields a small support bump; above T yields a larger bump (more impactful).
- "nearby" certificates are discounted vs "exact".
- "miss" = no update (epistemic non-commitment).

This is NOT a proof engine. It accumulates designated evidence and will tend toward ~1 confidence
if hits keep arriving and misses remain neutral—consistent with PEACE’s perspectival, non-explosive stance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Iterable
import random
import math
import csv
import time

# =========================
#  Trivalent / fuzzy truth
# =========================

@dataclass
class Truth:
    """Continuous confidence in (0,1), with ≈0→False-ish, 0.5→Both-ish, ≈1→True-ish."""
    value: float = 0.5

    @staticmethod
    def both() -> "Truth":
        return Truth(0.5)

    def clamp(self) -> None:
        eps = 1e-12
        if self.value <= eps:
            self.value = eps
        elif self.value >= 1 - eps:
            self.value = 1 - eps

    def label(self) -> str:
        v = self.value
        if v < 0.33:
            return "False-ish"
        elif v > 0.67:
            return "True-ish"
        else:
            return "Both-ish"

    def support(self, strength: float) -> None:
        """Increase confidence toward 1 by a fraction 'strength' of the remaining gap."""
        self.value = self.value + strength * (1.0 - self.value)
        self.clamp()

    # Intentionally no 'counter' since misses are epistemically neutral here.


# ====================================
#  Probabilistic Miller–Rabin (arbitrary size)
# ====================================

_SMALL_PRIMES: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _decompose(n: int) -> Tuple[int, int]:
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return d, s

def _mr_witness(a: int, d: int, n: int, s: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True  # 'a' witnesses compositeness

def is_probable_prime(n: int, rounds: int, rng: random.Random) -> bool:
    """Miller–Rabin with random bases; suitable for very large n."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = _decompose(n)
    # choose random bases uniformly from [2..n-2]
    max_base = n - 2
    for _ in range(rounds):
        a = rng.randrange(2, max_base + 1) if max_base >= 2 else 2
        if _mr_witness(a, d, n, s):
            return False
    return True

def is_prime(n: int, rng: random.Random) -> bool:
    """Adaptive Miller–Rabin rounds by size (cheap but effective)."""
    digits = len(str(n))
    if digits <= 18:
        r = 8
    elif digits <= 22:
        r = 6
    else:
        r = 4
    return is_probable_prime(n, rounds=r, rng=rng)


# =============================
#  Goldbach certificate helpers
# =============================

def goldbach_pair_via_subtractors(n: int, subtractors: List[int], rng: random.Random) -> Optional[Tuple[int, int]]:
    """
    Try to certify n = p + q with small subtractor primes p (odd), q := n - p.
    Returns (p, q) if q is (probable) prime.
    """
    if n % 2 != 0 or n < 4:
        return None
    # Try the list in a random order to reduce systematic bias
    subs = subtractors[:]
    rng.shuffle(subs)
    for p in subs:
        q = n - p
        if q > 1 and is_prime(q, rng):
            return (p, q)
    return None

def goldbach_nearby(n: int, offsets: List[int], subtractors: List[int], rng: random.Random) -> Optional[Tuple[int, int, int]]:
    """
    Search a few nearby even m = n + δ (δ even, small magnitude).
    Returns (m, p, q) if a certificate is found for m.
    """
    offs = offsets[:]
    rng.shuffle(offs)
    for d in offs:
        m = n + d
        if m >= 4 and m % 2 == 0:
            pq = goldbach_pair_via_subtractors(m, subtractors, rng)
            if pq is not None:
                p, q = pq
                return (m, p, q)
    return None


# ======================
#  Engine configuration
# ======================

@dataclass
class EngineConfig:
    # Evidence impact
    threshold_T: int = 4 * 10**18   # evidence above this is more impactful
    alpha_low: float = 0.001        # exact hit bump for n < T
    alpha_high: float = 0.02        # exact hit bump for n ≥ T
    nearby_discount: float = 0.6    # nearby hit is discounted vs exact

    # Search policy
    subtractor_primes: List[int] = field(default_factory=lambda: [3, 5, 7, 11, 13, 17, 19, 23, 29])
    even_offsets: List[int] = field(default_factory=lambda: [-22, -14, -6, -2, 2, 6, 14, 22])
    use_nearby: bool = True

    # Progression policy (semi-random jumps ensure broader coverage)
    jump_min: int = 10
    jump_max: int = 100  # inclusive, multiples of 10
    double_then_jump: bool = True  # n_{k+1} = 2*n_k + J

    # Practicality bounds (optional): rebase if n grows too big to keep runs responsive
    max_digits_before_rebase: int = 22
    rebase_low: int = 10**12
    rebase_high: int = 10**20


# ======================
#  PEACE Goldbach Engine
# ======================

@dataclass
class PeaceGoldbachEngine:
    cfg: EngineConfig = field(default_factory=EngineConfig)
    seed: int = 2025
    confidence: Truth = field(default_factory=Truth.both)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    # ---- confidence policy ----
    def update_confidence(self, n: int, outcome: str) -> None:
        """
        outcome ∈ {"exact","nearby","miss"}.
        - exact: support(alpha_high) if n ≥ T else support(alpha_low)
        - nearby: discounted support
        - miss: no-op (epistemic non-commitment)
        """
        big = (n >= self.cfg.threshold_T)
        if outcome == "exact":
            self.confidence.support(self.cfg.alpha_high if big else self.cfg.alpha_low)
        elif outcome == "nearby":
            base = self.cfg.alpha_high if big else self.cfg.alpha_low
            self.confidence.support(self.cfg.nearby_discount * base)
        elif outcome == "miss":
            pass
        else:
            raise ValueError(f"Unknown outcome: {outcome}")

    # ---- one evaluation step at n ----
    def evaluate_n(self, n: int) -> Dict[str, Any]:
        """Return a row dict for logs/analysis; also updates confidence."""
        # Optional rebase for practicality: keep digits bounded
        if len(str(n)) > self.cfg.max_digits_before_rebase:
            n = (self.rng.randrange(self.cfg.rebase_low // 2, self.cfg.rebase_high // 2)) * 2

        # Ensure even
        if n % 2 != 0:
            n += 1

        # Try exact certificate at n
        pair = goldbach_pair_via_subtractors(n, self.cfg.subtractor_primes, self.rng)
        if pair is not None:
            p, q = pair
            outcome = "exact"
            nearby_m, nearby_p, nearby_q = None, None, None
        else:
            # Optionally try a small nearby search
            if self.cfg.use_nearby:
                near = goldbach_nearby(n, self.cfg.even_offsets, self.cfg.subtractor_primes, self.rng)
                if near is not None:
                    outcome = "nearby"
                    nearby_m, nearby_p, nearby_q = near
                    p, q = (None, None)
                else:
                    outcome = "miss"
                    p = q = nearby_m = nearby_p = nearby_q = None
            else:
                outcome = "miss"
                p = q = nearby_m = nearby_p = nearby_q = None

        # Update confidence
        self.update_confidence(n, outcome)

        return {
            "n": int(n),
            "digits(n)": len(str(n)),
            ">=4e18": n >= self.cfg.threshold_T,
            "outcome": outcome,
            "pair_p": p,
            "pair_q": q,
            "nearby_m": nearby_m,
            "nearby_p": nearby_p,
            "nearby_q": nearby_q,
            "confidence": self.confidence.value,
            "label": self.confidence.label(),
        }

    # ---- progression to next n ----
    def next_n(self, n: int) -> Tuple[int, int]:
        """
        Semi-random jump policy:
          n' = 2*n + J,  where J ∈ {10,20,...,100}
        Returns (n', J).
        """
        J = 10 * self.rng.randint(self.cfg.jump_min // 10, self.cfg.jump_max // 10)
        if self.cfg.double_then_jump:
            n_next = 2 * n + J
        else:
            n_next = n + J
        # parity preserved since n is even and J multiple of 10
        return n_next, J

    # ---- run loop ----
    def run(self, n0: int = 10, steps: int = 200) -> List[Dict[str, Any]]:
        """
        Run the heuristic engine for 'steps' iterations starting from even n0.
        Returns a list of result rows, one per step.
        """
        rows: List[Dict[str, Any]] = []
        n = n0 if n0 % 2 == 0 else n0 + 1
        last_jump = 0

        for k in range(steps):
            row = self.evaluate_n(n)
            row["k"] = k
            row["jump_used"] = last_jump
            rows.append(row)
            n, last_jump = self.next_n(n)

        return rows

    # ---- utility: write CSV ----
    @staticmethod
    def to_csv(rows: Iterable[Dict[str, Any]], path: str) -> None:
        rows = list(rows)
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)


# ======================
#  Example usage / CLI
# ======================

if __name__ == "__main__":
    cfg = EngineConfig(
        threshold_T = 4 * 10**18,
        alpha_low   = 0.001,    # tiny bump for n < T
        alpha_high  = 0.02,     # bigger bump for n ≥ T
        nearby_discount = 0.6,
        subtractor_primes = [3,5,7,11,13,17,19,23,29],
        even_offsets = [-22, -14, -6, -2, 2, 6, 14, 22],
        jump_min = 10,
        jump_max = 100,
        double_then_jump = True,
        max_digits_before_rebase = 22,
        rebase_low  = 10**12,
        rebase_high = 10**20,
    )

    engine = PeaceGoldbachEngine(cfg=cfg, seed=2025)
    rows = engine.run(n0=10, steps=500)  # adjust steps as you like

    # Print a small summary to stdout
    print("k  n                          outcome   confidence  label   jump_used")
    for r in (rows[:10] + rows[-10:]):
        print(f"{r['k']:>3} {r['n']:<26} {r['outcome']:<8} {r['confidence']:.6f}  {r['label']:<9} {r['jump_used']}")

    # Optionally, save CSV
    # PeaceGoldbachEngine.to_csv(rows, "peace_goldbach_run.csv")
    # print("Saved: peace_goldbach_run.csv")
