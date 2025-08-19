#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEACE Goldbach Engine — Counterexample-Collapse Edition
-------------------------------------------------------

Behavior change:
- If a single CONFIRMED counterexample is found, collapse confidence to ~0 for
  the whole conjecture (from then on, outcome reports reflect the collapse).

Misses (including deep hunts that don't find a pair) remain epistemically neutral.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Iterable
import random
import math
import csv

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
#  Goldbach helpers
# =============================

def goldbach_pair_via_subtractors(n: int, subtractors: List[int], rng: random.Random) -> Optional[Tuple[int, int]]:
    """Try n = p + q using small subtractor primes p; q = n - p must be prime."""
    if n % 2 != 0 or n < 4:
        return None
    subs = subtractors[:]
    rng.shuffle(subs)
    for p in subs:
        q = n - p
        if q > 1 and is_prime(q, rng):
            return (p, q)
    return None

def goldbach_nearby(n: int, offsets: List[int], subtractors: List[int], rng: random.Random) -> Optional[Tuple[int, int, int]]:
    """Try small even offsets m = n + δ; return (m,p,q) if a certificate is found."""
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

# Utilities for (rare) exhaustive proof at small n

def _sieve_upto(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            sieve[p*p: limit+1: p] = b"\x00" * (((limit - p*p)//p) + 1)
    return [i for i, b in enumerate(sieve) if b]

def prove_no_goldbach_pair_if_small(n: int, small_limit: int, rng: random.Random) -> Optional[bool]:
    """
    If n is small (<= small_limit), *exhaustively* test all prime subtractors p ≤ n/2.
    Return:
      - True  -> confirmed NO pair exists (counterexample)
      - False -> a pair exists (not a counterexample)
      - None  -> n too large for exhaustive proof path
    Note: uses MR for q primality; for small n this is reliable and fast.
    """
    if n % 2 != 0 or n < 4:
        return False
    if n > small_limit:
        return None
    primes = _sieve_upto(n // 2)
    for p in primes:
        q = n - p
        if is_prime(q, rng):
            return False  # found a pair -> NOT a counterexample
    return True  # no pair found across all p -> confirmed counterexample


# --------- Deep (budgeted) search used by counterexample mode ---------

def goldbach_pair_deep(n: int,
                       subtractor_ceiling: int,
                       max_checks: int,
                       rng: random.Random) -> Optional[Tuple[int, int]]:
    """
    Broader, budgeted hunt at the *current n*:
    - Build many subtractor primes up to 'subtractor_ceiling'
    - Scan them (shuffled) up to 'max_checks' attempts.

    Returns (p,q) if found; else None if budget exhausted (NOT a proof of absence).
    """
    if n % 2 != 0 or n < 4:
        return None
    big_subs = _sieve_upto(subtractor_ceiling)
    rng.shuffle(big_subs)
    attempts = 0
    for p in big_subs:
        if attempts >= max_checks:
            break
        q = n - p
        if q > 1 and is_prime(q, rng):
            return (p, q)
        attempts += 1
    return None


# ======================
#  Engine configuration
# ======================

@dataclass
class EngineConfig:
    # Evidence impact
    threshold_T: int = 4 * 10**18
    alpha_low: float = 0.001
    alpha_high: float = 0.02
    nearby_discount: float = 0.6

    # Search policy
    subtractor_primes: List[int] = field(default_factory=lambda: [3,5,7,11,13,17,19,23,29])
    even_offsets: List[int] = field(default_factory=lambda: [-22,-14,-6,-2,2,6,14,22])
    use_nearby: bool = True

    # Progression policy
    jump_min: int = 10
    jump_max: int = 100
    double_then_jump: bool = True

    # Practicality bounds
    max_digits_before_rebase: int = 22
    rebase_low: int = 10**12
    rebase_high: int = 10**20

    # -------- Counterexample hunt switch & parameters --------
    counterexample_mode: bool = False
    counterexample_threshold: int = 4 * 10**19
    counterexample_window: int = 50         # pick a random step within this window
    deep_subtractor_ceiling: int = 200_000  # primes up to this are allowed as subtractors
    deep_max_checks: int = 20_000           # cap on deep attempts at that n

    # -------- Counterexample collapse policy --------
    collapse_on_counterexample: bool = True
    collapse_value: float = 1e-6            # confidence after collapse
    ce_small_exhaustive_limit: int = 2_000_000  # only numbers ≤ this get true exhaustive proof
                                                # (safe default; adjust if you want)


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
        # CE scheduling / state
        self._ce_planned_k: Optional[int] = None
        self._ce_triggered: bool = False
        self._collapsed: bool = False
        self._counterexample_n: Optional[int] = None

    # Public hook to simulate/inject a discovered counterexample (for experiments)
    def inject_counterexample(self, n: int) -> None:
        self._collapsed = True
        self._counterexample_n = int(n)
        self.confidence.value = max(1e-12, self.cfg.collapse_value)

    # ---- confidence policy ----
    def update_confidence(self, n: int, outcome: str) -> None:
        if self._collapsed:
            return
        big = (n >= self.cfg.threshold_T)
        if outcome in ("exact", "counter_attempt_found_pair"):
            self.confidence.support(self.cfg.alpha_high if big else self.cfg.alpha_low)
        elif outcome in ("nearby",):
            base = self.cfg.alpha_high if big else self.cfg.alpha_low
            self.confidence.support(self.cfg.nearby_discount * base)
        elif outcome in ("miss", "counter_attempt_no_pair"):
            pass  # neutral
        elif outcome == "counterexample_confirmed":
            if self.cfg.collapse_on_counterexample:
                self.inject_counterexample(n)
        else:
            raise ValueError(f"Unknown outcome: {outcome}")

    # ---- schedule the counterexample attempt once n crosses threshold ----
    def _maybe_schedule_counterexample(self, k: int, n: int) -> None:
        if (not self.cfg.counterexample_mode) or self._ce_triggered or self._collapsed:
            return
        if n > self.cfg.counterexample_threshold and self._ce_planned_k is None:
            self._ce_planned_k = k + self.rng.randint(0, max(1, self.cfg.counterexample_window))

    # ---- one evaluation step at n ----
    def evaluate_n(self, n: int, k: int) -> Dict[str, Any]:
        """Return a row dict for logs/analysis; also updates confidence."""
        # If collapsed, short-circuit but still log the step
        if self._collapsed:
            return {
                "k": k,
                "n": int(n if n % 2 == 0 else n + 1),
                "digits(n)": len(str(n)),
                ">=4e18": (n >= self.cfg.threshold_T),
                "outcome": "collapsed_false_global",
                "pair_p": None,
                "pair_q": None,
                "nearby_m": None,
                "nearby_p": None,
                "nearby_q": None,
                "confidence": self.confidence.value,
                "label": self.confidence.label(),
                "ce_mode": self.cfg.counterexample_mode,
                "ce_planned_k": self._ce_planned_k,
                "counterexample_n": self._counterexample_n,
            }

        # Optional rebase for practicality
        if len(str(n)) > self.cfg.max_digits_before_rebase:
            n = (self.rng.randrange(self.cfg.rebase_low // 2, self.cfg.rebase_high // 2)) * 2

        # Ensure even
        if n % 2 != 0:
            n += 1

        # Schedule counterexample attempt if applicable
        self._maybe_schedule_counterexample(k, n)

        outcome = None
        p = q = nearby_m = nearby_p = nearby_q = None

        # If it's the planned counterexample step, run deep/budgeted search first
        if self.cfg.counterexample_mode and self._ce_planned_k is not None and k == self._ce_planned_k:
            self._ce_triggered = True

            # First, try an EXHAUSTIVE proof path if n is small enough
            proof = prove_no_goldbach_pair_if_small(n, self.cfg.ce_small_exhaustive_limit, self.rng)
            if proof is True:
                outcome = "counterexample_confirmed"
                self._counterexample_n = n
            elif proof is False:
                # a pair exists (we didn't compute which one; get one quickly now)
                fast = goldbach_pair_via_subtractors(n, self.cfg.subtractor_primes, self.rng)
                if fast:
                    p, q = fast
                    outcome = "counter_attempt_found_pair"
                else:
                    # fallback to deep to actually exhibit it
                    deep_pair = goldbach_pair_deep(
                        n=n,
                        subtractor_ceiling=self.cfg.deep_subtractor_ceiling,
                        max_checks=self.cfg.deep_max_checks,
                        rng=self.rng
                    )
                    if deep_pair:
                        p, q = deep_pair
                        outcome = "counter_attempt_found_pair"
                    else:
                        outcome = "counter_attempt_no_pair"  # neutral
            else:
                # proof path not applicable; do a big-budget hunt
                deep_pair = goldbach_pair_deep(
                    n=n,
                    subtractor_ceiling=self.cfg.deep_subtractor_ceiling,
                    max_checks=self.cfg.deep_max_checks,
                    rng=self.rng
                )
                if deep_pair is not None:
                    p, q = deep_pair
                    outcome = "counter_attempt_found_pair"
                else:
                    outcome = "counter_attempt_no_pair"  # neutral (not a proof of absence)

        # If not the special step, do the normal flow
        if outcome is None:
            pair = goldbach_pair_via_subtractors(n, self.cfg.subtractor_primes, self.rng)
            if pair is not None:
                p, q = pair
                outcome = "exact"
            else:
                if self.cfg.use_nearby:
                    near = goldbach_nearby(n, self.cfg.even_offsets, self.cfg.subtractor_primes, self.rng)
                    if near is not None:
                        outcome = "nearby"
                        nearby_m, nearby_p, nearby_q = near
                    else:
                        outcome = "miss"
                else:
                    outcome = "miss"

        # Update confidence per outcome (handles collapse if confirmed)
        self.update_confidence(n, outcome)

        return {
            "k": k,
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
            "ce_mode": self.cfg.counterexample_mode,
            "ce_planned_k": self._ce_planned_k,
            "counterexample_n": self._counterexample_n,
        }

    # ---- progression to next n ----
    def next_n(self, n: int) -> Tuple[int, int]:
        J = 10 * self.rng.randint(self.cfg.jump_min // 10, self.cfg.jump_max // 10)
        n_next = 2 * n + J if self.cfg.double_then_jump else n + J
        return n_next, J

    # ---- run loop ----
    def run(self, n0: int = 10, steps: int = 200) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        n = n0 if n0 % 2 == 0 else n0 + 1
        last_jump = 0
        for k in range(steps):
            row = self.evaluate_n(n, k)
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
        alpha_low   = 0.001,
        alpha_high  = 0.02,
        nearby_discount = 0.6,
        subtractor_primes = [3,5,7,11,13,17,19,23,29],
        even_offsets = [-22, -14, -6, -2, 2, 6, 14, 22],
        jump_min = 10,
        jump_max = 100,
        double_then_jump = True,
        max_digits_before_rebase = 22,
        rebase_low  = 10**12,
        rebase_high = 10**20,

        # Counterexample hunt
        counterexample_mode = True,
        counterexample_threshold = 4 * 10**19,
        counterexample_window = 50,
        deep_subtractor_ceiling = 200_000,
        deep_max_checks = 20_000,

        # Collapse policy
        collapse_on_counterexample = True,
        collapse_value = 1e-6,
        ce_small_exhaustive_limit = 2_000_000,
    )

    engine = PeaceGoldbachEngine(cfg=cfg, seed=2025)
    rows = engine.run(n0=10, steps=500)

    print("k   n                          outcome                          conf       label        ce_planned_k  ce_n")
    for r in (rows[:10] + rows[-15:]):
        print(f"{r['k']:>3} {r['n']:<26} {r['outcome']:<30} {r['confidence']:.8f}  {r['label']:<12} {str(r['ce_planned_k']):>5} {str(r['counterexample_n']):>8}")
