#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Iterable
import random, csv

# =========================
#  Trivalent / fuzzy truth
# =========================

@dataclass
class Truth:
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
        return "False-ish" if v < 0.33 else ("True-ish" if v > 0.67 else "Both-ish")
    def support(self, strength: float) -> None:
        self.value = self.value + strength * (1.0 - self.value)
        self.clamp()

# ====================================
#  Probabilistic Miller–Rabin
# ====================================

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

def is_probable_prime(n: int, rounds: int, rng: random.Random) -> bool:
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0: return False
    d, s = _decompose(n)
    max_base = n - 2
    for _ in range(rounds):
        a = rng.randrange(2, max_base + 1) if max_base >= 2 else 2
        if _mr_witness(a, d, n, s):
            return False
    return True

def is_prime(n: int, rng: random.Random) -> bool:
    digits = len(str(n))
    r = 8 if digits <= 18 else (6 if digits <= 22 else 4)
    return is_probable_prime(n, r, rng)

# =============================
#  Goldbach helpers
# =============================

def goldbach_pair_via_subtractors(n: int, subtractors: List[int], rng: random.Random) -> Optional[Tuple[int, int]]:
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

# Small sieve utilities

def _sieve_upto(limit: int) -> List[int]:
    if limit < 2: return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            sieve[p*p: limit+1: p] = b"\x00" * (((limit - p*p)//p) + 1)
    return [i for i, b in enumerate(sieve) if b]

def prove_no_goldbach_pair_if_small(n: int, small_limit: int, rng: random.Random) -> Optional[bool]:
    """
    If n <= small_limit, exhaustively test prime subtractors p ≤ n/2 (q primality tested via MR).
    Returns True iff no pair found under this method (treated as 'confirmed' in the engine).
    """
    if n % 2 != 0 or n < 4:
        return False
    if n > small_limit:
        return None
    primes = _sieve_upto(n // 2)
    for p in primes:
        q = n - p
        if is_prime(q, rng):
            return False
    return True

# --------- Deep (budgeted) search ---------

def goldbach_pair_deep(n: int,
                       subtractor_ceiling: int,
                       max_checks: Optional[int],
                       rng: random.Random) -> Optional[Tuple[int, int]]:
    """
    Budgeted hunt at current n:
      - Build subtractor primes up to 'subtractor_ceiling'
      - Scan them (shuffled) up to 'max_checks' attempts; if max_checks=None, scan *all* built primes.
    NOTE: Even with max_checks=None, runtime is still bounded by 'subtractor_ceiling'.
    """
    if n % 2 != 0 or n < 4:
        return None
    big_subs = _sieve_upto(subtractor_ceiling)
    rng.shuffle(big_subs)
    attempts = 0
    for p in big_subs:
        if (max_checks is not None) and (attempts >= max_checks):
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

    # Counterexample hunt
    counterexample_mode: bool = True
    counterexample_threshold: int = 4 * 10**19
    counterexample_window: int = 50
    deep_subtractor_ceiling: int = 200_000
    deep_max_checks: Optional[int] = 20_000  # None means: check *all* built primes (still bounded by ceiling)

    # Collapse policy
    collapse_on_counterexample: bool = True
    collapse_value: float = 1e-6
    ce_small_exhaustive_limit: int = 2_000_000

    # >>> NEW SWITCH: collapse on budgeted failure during CE attempt
    collapse_on_budgeted_failure: bool = False

# ======================
#  Engine
# ======================

@dataclass
class PeaceGoldbachEngine:
    cfg: EngineConfig = field(default_factory=EngineConfig)
    seed: int = 2025
    confidence: Truth = field(default_factory=Truth.both)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._ce_planned_k: Optional[int] = None
        self._ce_triggered: bool = False
        self._collapsed: bool = False
        self._counterexample_n: Optional[int] = None

    def inject_counterexample(self, n: int) -> None:
        self._collapsed = True
        self._counterexample_n = int(n)
        self.confidence.value = max(1e-12, self.cfg.collapse_value)

    def update_confidence(self, n: int, outcome: str) -> None:
        if self._collapsed:
            return
        big = (n >= self.cfg.threshold_T)
        if outcome in ("exact", "counter_attempt_found_pair"):
            self.confidence.support(self.cfg.alpha_high if big else self.cfg.alpha_low)
        elif outcome == "nearby":
            base = self.cfg.alpha_high if big else self.cfg.alpha_low
            self.confidence.support(self.cfg.nearby_discount * base)
        elif outcome == "counterexample_confirmed":
            if self.cfg.collapse_on_counterexample:
                self.inject_counterexample(n)
        elif outcome == "counter_attempt_no_pair":
            if self.cfg.collapse_on_budgeted_failure:
                self.inject_counterexample(n)
        elif outcome in ("miss", "collapsed_false_global"):
            pass
        else:
            raise ValueError(f"Unknown outcome: {outcome}")

    def _maybe_schedule_counterexample(self, k: int, n: int) -> None:
        if (not self.cfg.counterexample_mode) or self._ce_triggered or self._collapsed:
            return
        if n > self.cfg.counterexample_threshold and self._ce_planned_k is None:
            self._ce_planned_k = k + self.rng.randint(0, max(1, self.cfg.counterexample_window))

    def evaluate_n(self, n: int, k: int) -> Dict[str, Any]:
        if self._collapsed:
            return {
                "k": k, "n": int(n if n % 2 == 0 else n + 1),
                "digits(n)": len(str(n)), ">=4e18": (n >= self.cfg.threshold_T),
                "outcome": "collapsed_false_global",
                "pair_p": None, "pair_q": None,
                "nearby_m": None, "nearby_p": None, "nearby_q": None,
                "confidence": self.confidence.value, "label": self.confidence.label(),
                "ce_mode": self.cfg.counterexample_mode,
                "ce_planned_k": self._ce_planned_k,
                "counterexample_n": self._counterexample_n,
            }

        # Practical rebase
        if len(str(n)) > self.cfg.max_digits_before_rebase:
            n = (self.rng.randrange(self.cfg.rebase_low // 2, self.cfg.rebase_high // 2)) * 2
        if n % 2 != 0:
            n += 1

        self._maybe_schedule_counterexample(k, n)

        outcome = None
        p = q = nearby_m = nearby_p = nearby_q = None

        # Planned counterexample attempt
        if self.cfg.counterexample_mode and self._ce_planned_k is not None and k == self._ce_planned_k:
            self._ce_triggered = True

            proof = prove_no_goldbach_pair_if_small(n, self.cfg.ce_small_exhaustive_limit, self.rng)
            if proof is True:
                outcome = "counterexample_confirmed"
                self._counterexample_n = n
            elif proof is False:
                fast = goldbach_pair_via_subtractors(n, self.cfg.subtractor_primes, self.rng)
                if fast:
                    p, q = fast
                    outcome = "counter_attempt_found_pair"
                else:
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
                        outcome = "counter_attempt_no_pair"
            else:
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
                    outcome = "counter_attempt_no_pair"

        # Normal step
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

        self.update_confidence(n, outcome)

        return {
            "k": k, "n": int(n), "digits(n)": len(str(n)), ">=4e18": n >= self.cfg.threshold_T,
            "outcome": outcome, "pair_p": p, "pair_q": q,
            "nearby_m": nearby_m, "nearby_p": nearby_p, "nearby_q": nearby_q,
            "confidence": self.confidence.value, "label": self.confidence.label(),
            "ce_mode": self.cfg.counterexample_mode, "ce_planned_k": self._ce_planned_k,
            "counterexample_n": self._counterexample_n,
        }

    def next_n(self, n: int) -> Tuple[int, int]:
        J = 10 * self.rng.randint(self.cfg.jump_min // 10, self.cfg.jump_max // 10)
        n_next = 2 * n + J if self.cfg.double_then_jump else n + J
        return n_next, J

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

    @staticmethod
    def to_csv(rows: Iterable[Dict[str, Any]], path: str) -> None:
        rows = list(rows)
        if not rows: return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

# ======================
#  Demo / CLI
# ======================

def demo():
    seed = 2025

    print("\n=== Mode A: NEUTRAL on budget failure (no collapse) ===")
    cfgA = EngineConfig(
        counterexample_mode=True,
        deep_subtractor_ceiling=200_000,
        deep_max_checks=20_000,               # generous budget
        collapse_on_budgeted_failure=False,   # <-- switch OFF
    )
    engA = PeaceGoldbachEngine(cfg=cfgA, seed=seed)
    rowsA = engA.run(n0=10, steps=120)
    print("final:", rowsA[-1]["outcome"], rowsA[-1]["confidence"], rowsA[-1]["label"], "collapsed?", rowsA[-1]["outcome"]=="collapsed_false_global")

    print("\n=== Mode B: COLLAPSE on budget failure (forced budget exhaustion) ===")
    cfgB = EngineConfig(
        counterexample_mode=True,
        counterexample_window=0,              # schedule CE attempt immediately when threshold crossed
        deep_subtractor_ceiling=50,           # tiny pool
        deep_max_checks=0,                    # force 'counter_attempt_no_pair'
        collapse_on_budgeted_failure=True,    # <-- switch ON
    )
    engB = PeaceGoldbachEngine(cfg=cfgB, seed=seed)
    rowsB = engB.run(n0=10, steps=100)

    # Print around collapse
    for r in rowsB[55:66]:
        print(f"k={r['k']:<3} n={r['n']:<26} outcome={r['outcome']:<24} conf={r['confidence']:.10f} label={r['label']:<9} ce@{r['ce_planned_k']} ce_n={r['counterexample_n']}")

if __name__ == "__main__":
    demo()
