#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEACE Meta Engine — Dialetheic, Budgeted, Self-editing, LLM-driven
------------------------------------------------------------------
A truth-seeking engine for general math/CS problems framed as:
   “Is this problem class solvable within polynomial time?”

Core properties:
- Dialetheic fuzzy truth: 0≈False-ish, 0.5≈Both-ish, 1≈True-ish (confidence == truth value)
- Epistemically humble: never asserts absolute truth; budgets are explicit
- Intelligently leaping: uses LLM (stubbed here) to propose solvers & patches
- Safely self-editing: scans & sandboxes code before loading/applying
- Counterexample collapse: if a well-formed counterexample to the *question* appears,
  collapse confidence to ~0 for “solvable in polynomial time”.
"""

from __future__ import annotations

# ==========================
# Standard Library
# ==========================
import ast
import time
import types
import hashlib
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ==========================
# Dialetheic Truth / Confidence
# ==========================
@dataclass
class Truth:
    """
    Fuzzy dialetheic truth in (0,1). Confidence == truth value.
    label():
        <0.33 -> "False-ish"
        0.33..0.67 -> "Both-ish"
        >0.67 -> "True-ish"
    """
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
        """Move value toward 1 by fraction `strength` of remaining gap."""
        self.value = self.value + strength * (1.0 - self.value)
        self.clamp()

    def doubt(self, strength: float) -> None:
        """Move value toward 0 by fraction `strength` of current value."""
        self.value = self.value * (1.0 - strength)
        self.clamp()

# ==========================
# Engine Configuration
# ==========================
@dataclass
class EngineConfig:
    # Time/attempt budgets (pragmatic bounds)
    time_budget_sec: float = 2.0             # wall-clock per solve attempt
    patch_rounds_max: int = 3                # how many self-edits per problem
    samples_per_round: int = 3               # how many instance sizes to try per round

    # Confidence dynamics
    alpha_success_small: float = 0.02        # n within small regime
    alpha_success_large: float = 0.10        # n within large regime
    large_threshold_n: int = 10_000          # boundary between small/large

    # Collapse policy
    collapse_on_counterexample: bool = True
    collapse_value: float = 1e-6

    # Safety policy
    banned_modules: Tuple[str, ...] = ("os", "sys", "subprocess", "pickle", "marshal", "socket")
    banned_builtins: Tuple[str, ...] = ("eval", "exec", "__import__", "open", "compile")

    # Sandbox builtins (whitelist)
    safe_builtins: Tuple[str, ...] = (
        "abs", "all", "any", "bin", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "pow", "range", "reversed", "set", "sorted",
        "str", "sum", "tuple", "zip"
    )

# ==========================
# Problem & Patch Models
# ==========================
@dataclass(frozen=True)
class ProblemSpec:
    """
    A problem family P. The engine asks:
       "Is P solvable in polynomial time?"
    The LLM supplies code implementing a solver with the signature:
       solve_instance(instance) -> ("YES"/"NO"/None, stats_dict)
    and a generator:
       generate_instances(seed:int, sizes:List[int]) -> List[Any]
    """
    name: str
    description: str
    input_size_hint: List[int]                # e.g., [50, 100, 200, 400]
    domain: str = "computational_complexity"

@dataclass
class Patch:
    """
    A code patch (self-edit) proposed by the LLM:
      - module_code defines required symbols: solve_instance, generate_instances
      - reason explains intent (heuristic change, pruning, new data structure, etc.)
    """
    patch_id: str
    module_code: str
    reason: str
    timestamp: float = field(default_factory=lambda: time.time())

# ==========================
# Safety: AST & import checks
# ==========================
class SafeCodeError(Exception):
    pass

class SafeASTChecker(ast.NodeVisitor):
    """Reject dangerous imports, attributes, and builtins usage."""
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in self.cfg.banned_modules:
                self.errors.append(f"Import of banned module: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in self.cfg.banned_modules:
            self.errors.append(f"Import-from banned module: {node.module}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Disallow dunder attribute spelunking
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            self.errors.append(f"Access to dunder attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.cfg.banned_builtins:
            self.errors.append(f"Use of banned builtin: {node.id}")
        self.generic_visit(node)

def check_code_safety(code: str, cfg: EngineConfig) -> None:
    """Parse and validate AST; raise SafeCodeError if unsafe or invalid."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SafeCodeError(f"Syntax error: {e}") from e
    checker = SafeASTChecker(cfg)
    checker.visit(tree)
    if checker.errors:
        raise SafeCodeError(" ; ".join(checker.errors))

# ==========================
# Sandbox Loader (safe builtins)
# ==========================
import builtins as _builtins

def build_safe_globals(cfg: EngineConfig) -> Dict[str, Any]:
    """
    Build a whitelist-only __builtins__ dict (not a security boundary; for demo).
    """
    allowed: Dict[str, Any] = {}
    for name in cfg.safe_builtins:
        if hasattr(_builtins, name):
            allowed[name] = getattr(_builtins, name)
    # Ensure banned are not present even if misconfigured
    for b in cfg.banned_builtins:
        if b in allowed:
            allowed.pop(b, None)
    return {"__builtins__": allowed}

def load_module_safely(code: str, module_name: str, cfg: EngineConfig) -> types.ModuleType:
    """Compile & exec code into a sandboxed module object."""
    check_code_safety(code, cfg)
    globs = build_safe_globals(cfg)
    locs: Dict[str, Any] = {}
    compiled = compile(code, filename=f"<{module_name}>", mode="exec")
    exec(compiled, globs, locs)  # sandboxed builtins only
    mod = types.ModuleType(module_name)
    for k, v in locs.items():
        setattr(mod, k, v)
    # Require required symbols
    for required in ("solve_instance", "generate_instances"):
        if not hasattr(mod, required):
            raise SafeCodeError(f"Module missing required symbol: {required}")
    return mod

# ==========================
# LLM Adapter (stub — replace with your LLM)
# ==========================
class LLMAdapter:
    """
    Swap this stub for your real LLM calls.
    Responsibilities:
      - identify_problem(spec) -> maybe refine description
      - propose_initial_solver(spec) -> Patch (module_code)
      - analyze_failure(spec, logs) -> Patch (module_code)
      - code_review(patch) -> safety/style comments
      - detect_counterexample(logs) -> (bool, reason)
    """
    def identify_problem(self, spec: ProblemSpec) -> ProblemSpec:
        return spec

    def _toy_initial_solver(self, spec: ProblemSpec) -> str:
        """
        Minimal baseline:
        - generate_instances: list of dicts with "n"
        - solve_instance: quick 'YES' for n <= 2000; None beyond (budget trigger)
        """
        return r'''
def generate_instances(seed, sizes):
    return [{"n": int(s)} for s in sizes]

def solve_instance(instance):
    n = int(instance["n"])
    # Quick for small n; inconclusive for big n (simulate budget)
    if n <= 2000:
        ops = n * n
        return ("YES", {"ops": ops, "n": n, "reason": "baseline-heuristic"})
    return (None, {"n": n, "reason": "budget-exhausted"})
'''

    def propose_initial_solver(self, spec: ProblemSpec) -> Patch:
        code = self._toy_initial_solver(spec)
        pid = hashlib.sha256((spec.name + "|init|" + str(time.time())).encode()).hexdigest()[:12]
        return Patch(patch_id=pid, module_code=code, reason="Initial baseline solver")

    def analyze_failure(self, spec: ProblemSpec, logs: List[Dict[str, Any]]) -> Patch:
        """
        Inspect logs and propose a refined module with wider 'YES' range.
        """
        code = r'''
def generate_instances(seed, sizes):
    return [{"n": int(s)} for s in sizes]

def solve_instance(instance):
    n = int(instance["n"])
    # Improved up to ~10000; None beyond
    if n <= 10000:
        ops = n * (n // 2 + 1)
        return ("YES", {"ops": ops, "n": n, "reason": "improved-heuristic"})
    return (None, {"n": n, "reason": "budget-exhausted"})
'''
        pid = hashlib.sha256((spec.name + "|patch|" + str(time.time())).encode()).hexdigest()[:12]
        return Patch(patch_id=pid, module_code=code, reason="Improve heuristic & capacity")

    def code_review(self, patch: Patch) -> str:
        return "Style OK. Budget-cooperative. No obvious hazards."

    def detect_counterexample(self, logs: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        If a solver flags the question as ill-formed (stats['flag']=="ill_formed_question"),
        collapse.
        """
        for row in logs:
            stats = row.get("stats") or {}
            if stats.get("flag") == "ill_formed_question":
                return True, "Solver flagged the question as ill-formed."
        return False, ""
# ==========================
# Budget helpers
# ==========================
class BudgetTimer:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.start = time.time()
    def exhausted(self) -> bool:
        return (time.time() - self.start) >= self.seconds

# ==========================
# PEACE Meta Engine
# ==========================
class PeaceMetaEngine:
    def __init__(self, cfg: EngineConfig | None = None, llm: LLMAdapter | None = None):
        self.cfg = cfg or EngineConfig()
        self.llm = llm or LLMAdapter()
        self.confidence = Truth.both()
        self.logs: List[Dict[str, Any]] = []
        self.current_module: Optional[types.ModuleType] = None
        self.current_patch_id: Optional[str] = None

    # ---------- confidence plumbing ----------
    def record_success(self, n: int) -> None:
        alpha = self.cfg.alpha_success_large if n >= self.cfg.large_threshold_n else self.cfg.alpha_success_small
        self.confidence.support(alpha)

    def collapse(self, reason: str) -> None:
        if self.cfg.collapse_on_counterexample:
            self.confidence.value = max(1e-12, self.cfg.collapse_value)
        self.logs.append({"event": "collapse", "reason": reason, "confidence": self.confidence.value})

    # ---------- solver loading ----------
    def load_solver_patch(self, patch: Patch) -> None:
        mod = load_module_safely(patch.module_code, f"solver_{patch.patch_id}", self.cfg)
        self.current_module = mod
        self.current_patch_id = patch.patch_id

    # ---------- single round attempt ----------
    def _attempt_round(self, spec: ProblemSpec, sizes: List[int], seed: int, timer: BudgetTimer) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        if not self.current_module:
            return {"round_ok": False, "reason": "no_module"}

        gen = getattr(self.current_module, "generate_instances", None)
        solve = getattr(self.current_module, "solve_instance", None)
        if not (callable(gen) and callable(solve)):
            return {"round_ok": False, "reason": "bad_module"}

        try:
            instances = gen(seed, sizes)
        except Exception as e:
            return {"round_ok": False, "reason": f"gen_error: {e}"}

        for inst in instances:
            if timer.exhausted():
                results.append({"n": inst.get("n"), "decision": None, "stats": {"reason": "budget-exhausted"}})
                break
            try:
                decision, stats = solve(inst)
            except Exception as e:
                decision, stats = None, {"error": str(e), "trace": traceback.format_exc()}

            row = {"n": inst.get("n"), "decision": decision, "stats": stats, "patch_id": self.current_patch_id}
            results.append(row)
            self.logs.append(row)

            # confidence dynamics: reward successes
            if decision in ("YES", "NO"):
                self.record_success(int(inst.get("n") or 0))

        return {"round_ok": True, "results": results}

    # ---------- main solve loop with self-edits ----------
    def solve_problem_class(self, spec: ProblemSpec, seed: int = 2025) -> Dict[str, Any]:
        # Step 0: LLM refines problem (stubbed)
        spec = self.llm.identify_problem(spec)

        # Step 1: initial solver patch
        init_patch = self.llm.propose_initial_solver(spec)
        self.load_solver_patch(init_patch)
        self.logs.append({"event": "patch_loaded", "patch_id": init_patch.patch_id, "reason": init_patch.reason})

        # Rounds of attempts + self-edits
        for round_idx in range(self.cfg.patch_rounds_max + 1):
            # Sizes to test this round
            sizes = spec.input_size_hint[: self.cfg.samples_per_round]
            timer = BudgetTimer(self.cfg.time_budget_sec)

            round_out = self._attempt_round(spec, sizes, seed + round_idx, timer)

            # Check for counterexample (ill-formed question)
            hit, why = self.llm.detect_counterexample(self.logs)
            if hit:
                self.collapse(why)
                return {
                    "ok": False,
                    "collapsed": True,
                    "reason": why,
                    "confidence": self.confidence.value,
                    "label": self.confidence.label(),
                    "round": round_idx,
                    "logs": self.logs[-10:],
                }

            # If round succeeded for all instances (YES/NO), we consider it progress.
            if round_out.get("round_ok"):
                decisions = [r.get("decision") for r in round_out.get("results", [])]
                # If we made decisions for every tested size, keep going or finish
                if decisions and all(d in ("YES", "NO") for d in decisions):
                    # successful round: if confidence is high enough, we can stop
                    if self.confidence.value > 0.85 or round_idx == self.cfg.patch_rounds_max:
                        return {
                            "ok": True,
                            "collapsed": False,
                            "confidence": self.confidence.value,
                            "label": self.confidence.label(),
                            "round": round_idx,
                            "results": round_out["results"],
                        }
                    # else continue to next round with same module (maybe larger sizes later)
                    continue

            # If we’re here, we need a self-edit (ask LLM for a patch)
            if round_idx < self.cfg.patch_rounds_max:
                patch = self.llm.analyze_failure(spec, self.logs)
                # optional: code_review
                _review = self.llm.code_review(patch)
                try:
                    self.load_solver_patch(patch)
                    self.logs.append({"event": "patch_loaded", "patch_id": patch.patch_id, "reason": patch.reason})
                except SafeCodeError as e:
                    # If patch unsafe, record and keep previous module (or stop if none)
                    self.logs.append({"event": "patch_rejected", "patch_id": patch.patch_id, "error": str(e)})
                    if not self.current_module:
                        return {
                            "ok": False,
                            "collapsed": False,
                            "reason": "no_safe_patch_available",
                            "confidence": self.confidence.value,
                            "label": self.confidence.label(),
                        }
            else:
                # No more patch rounds; finish neutrally
                break

        # If we exit loop without collapse or high confidence, remain epistemically humble
        return {
            "ok": True,
            "collapsed": False,
            "confidence": self.confidence.value,
            "label": self.confidence.label(),
            "reason": "budget_exhausted_or_partial_progress",
            "logs": self.logs[-10:],
        }

# ==========================
# CLI (flexible runtime control)
# ==========================
import argparse, sys

def _build_cfg_from_args(args: argparse.Namespace) -> EngineConfig:
    cfg = EngineConfig(
        time_budget_sec=args.time_budget,
        patch_rounds_max=args.patch_rounds,
        samples_per_round=args.samples_per_round,
        alpha_success_small=args.alpha_small,
        alpha_success_large=args.alpha_large,
        large_threshold_n=args.large_threshold,
        collapse_on_counterexample=not args.no_collapse_on_ce,
        collapse_value=args.collapse_value,
    )

    # Optional builtins adjustments (whitelist/blacklist)
    if args.add_builtin:
        safe = list(cfg.safe_builtins)
        for name in args.add_builtin:
            if name not in safe:
                safe.append(name)
        cfg.safe_builtins = tuple(safe)

    if args.ban_builtin:
        banned = set(cfg.banned_builtins)
        for name in args.ban_builtin:
            banned.add(name)
        cfg.banned_builtins = tuple(sorted(banned))

    return cfg

def _parse_sizes(csv_text: str) -> List[int]:
    parts = [p.strip() for p in csv_text.split(",") if p.strip()]
    sizes: List[int] = []
    for p in parts:
        try:
            sizes.append(int(p))
        except ValueError:
            raise SystemExit(f"Bad size value: {p!r}. Use comma-separated integers, e.g. 200,2000,10000")
    if not sizes:
        raise SystemExit("No sizes provided after parsing.")
    return sizes

def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PEACE Meta Engine — Dialetheic, Budgeted, Self-editing, LLM-driven")
    # Budgets & dynamics
    parser.add_argument("--time-budget", type=float, default=1.0, help="Per-round wall-clock seconds (default: 1.0)")
    parser.add_argument("--patch-rounds", type=int, default=2, help="Max self-edit rounds (default: 2)")
    parser.add_argument("--samples-per-round", type=int, default=3, help="Instances per round (default: 3)")
    parser.add_argument("--alpha-small", type=float, default=0.05, help="Confidence boost for small n")
    parser.add_argument("--alpha-large", type=float, default=0.15, help="Confidence boost for large n")
    parser.add_argument("--large-threshold", type=int, default=5000, help="Boundary between small/large n")

    # Collapse policy
    parser.add_argument("--no-collapse-on-ce", action="store_true", help="Disable collapse on counterexample flag")
    parser.add_argument("--collapse-value", type=float, default=1e-6, help="Confidence after collapse (default: 1e-6)")

    # Problem spec
    parser.add_argument("--name", type=str, default="generic_p_vs_np_demo", help="Problem family name")
    parser.add_argument("--desc", type=str, default="Toy family standing in for a decision class; LLM supplies solvers.", help="Description")
    parser.add_argument("--sizes", type=str, default="200,2000,10000,20000",
                        help="Comma-separated input sizes to test (e.g., 200,2000,10000)")

    # Optional: tweak builtin whitelist/blacklist for sandbox
    parser.add_argument("--add-builtin", action="append", default=[],
                        help="Add a builtin to the allowed whitelist (repeatable). Example: --add-builtin round")
    parser.add_argument("--ban-builtin", action="append", default=[],
                        help="Ban a builtin in addition to defaults (repeatable). Example: --ban-builtin sum")

    args = parser.parse_args(argv)

    cfg = _build_cfg_from_args(args)

    # Build the problem spec from flags
    spec = ProblemSpec(
        name=args.name,
        description=args.desc,
        input_size_hint=_parse_sizes(args.sizes),
    )

    # Create engine and run
    engine = PeaceMetaEngine(cfg=cfg, llm=LLMAdapter())
    out = engine.solve_problem_class(spec, seed=2025)

    # Pretty print results
    print("ok:", out.get("ok"))
    print("collapsed:", out.get("collapsed"))
    print("confidence:", f"{out.get('confidence', 0):.6f}", "-", out.get("label"))
    if "reason" in out:
        print("reason:", out["reason"])
    if "results" in out:
        for r in out["results"][:10]:
            print("  n=", r.get("n"), "decision=", r.get("decision"), "stats=", r.get("stats"))
    if "logs" in out:
        print("  ...logs tail shown; total logs:", len(out.get("logs", [])))

    return 0

if __name__ == "__main__":
    raise SystemExit(main_cli())
