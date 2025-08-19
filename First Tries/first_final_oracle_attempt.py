from __future__ import annotations

# ==========================
# Standard Library
# ==========================
import ast
import time
import sqlite3
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import threading

# ==========================
# Logging
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("peace_oracle.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PEACE")

# ==========================
# Truth values (discrete, no UNKNOWN)
# ==========================
class TV(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2       # Paraconsistent: both true and/or false (includes “unknown”)

# ==========================
# PEACE fuzzy truth tracker (continuous)
# ==========================
@dataclass
class TruthFuzzy:
    """Continuous truth/confidence in (0,1); 0≈False-ish, 0.5≈Both-ish, 1≈True-ish."""
    value: float = 0.5

    def clamp(self) -> None:
        eps = 1e-12
        if self.value < eps:
            self.value = eps
        if self.value > 1 - eps:
            self.value = 1 - eps

    def label(self) -> str:
        v = self.value
        if v < 0.33:
            return "False-ish"
        if v > 0.67:
            return "True-ish"
        return "Both-ish"

    # Nudge toward True
    def support(self, strength: float) -> None:
        self.value = self.value + strength * (1.0 - self.value)
        self.clamp()

    # Nudge toward False
    def doubt(self, strength: float) -> None:
        self.value = self.value * (1.0 - strength)
        self.clamp()

# ==========================
# Epistemic state for one problem family
# ==========================
@dataclass
class EpistemicState:
    truth: TruthFuzzy = field(default_factory=TruthFuzzy)  # continuous truth/confidence
    verdict: TV = TV.BOTH                                  # discrete verdict (BOTH subsumes unknown)
    collapsed_reason: Optional[str] = None                 # record UP/DOWN collapse cause

    def collapse_up(self, reason: str, to_value: float = 1.0 - 1e-6) -> None:
        """Solution found (for this framing): commit to ~True."""
        self.truth.value = max(self.truth.value, to_value)
        self.truth.clamp()
        self.verdict = TV.TRUE
        self.collapsed_reason = reason

    def collapse_down(self, reason: str, to_value: float = 1e-6) -> None:
        """Ill-posed / counterexample: commit to ~False."""
        self.truth.value = min(self.truth.value, to_value)
        self.truth.clamp()
        self.verdict = TV.FALSE
        self.collapsed_reason = reason

# ==========================
# Engine policy / configuration
# ==========================
@dataclass
class EnginePolicy:
    # Budgeting
    time_budget_sec: float = 2.0            # per attempt / round wall-clock (cooperative)
    samples_per_round: int = 3
    patch_rounds_max: int = 3

    # Confidence dynamics
    alpha_small: float = 0.02               # reward for decided small instances
    alpha_large: float = 0.10               # reward for decided large instances
    large_threshold_n: int = 10_000

    # Collapses
    collapse_on_solution: bool = True
    solution_value: float = 1.0 - 1e-6
    collapse_on_counterexample: bool = True
    counterexample_value: float = 1e-6

    # Safety (static checks only; not a full sandbox)
    banned_modules: Tuple[str, ...] = ("os", "sys", "subprocess", "pickle", "marshal")
    banned_builtins: Tuple[str, ...] = ("eval", "exec", "__import__", "open", "compile")

# ==========================
# Core data models
# ==========================
@dataclass(frozen=True)
class MathematicalProblem:
    """
    A general problem/conjecture; the engine asks:
        “Under current framing and budgets, can we produce decisive answers
         (YES/NO) and/or a solution/witness?”
    """
    name: str
    description: str
    complexity_score: int            # 1-10 rough scale
    computational_bound: int         # feasible direct computation bound
    problem_type: str                # "number_theory", "combinatorics", "crypto", ...
    verification_function: Optional[Callable[[int], Any]] = None

    def __hash__(self) -> int:
        return hash((self.name, self.description, self.complexity_score))

@dataclass
class CodeModification:
    modification_id: str
    target_module: str
    target_function: str
    original_code: str
    modified_code: str
    modification_type: str            # "add", "modify", "delete"
    safety_score: float
    mathematical_soundness: float
    reasoning: str
    timestamp: float

    def __hash__(self) -> int:
        return hash(self.modification_id)

# ==========================
# PEACE perspectives (pluggable evaluators)
# ==========================
@dataclass
class PEACEPerspective:
    name: str
    evaluate_fn: Callable[[Any], TV]
    confidence_fn: Callable[[Any], float]
    memory: Dict[str, Tuple[TV, float]] = field(default_factory=dict)
    stability_score: float = 1.0

    def evaluate(self, statement: Any) -> Tuple[TV, float]:
        key = str(statement)
        if key not in self.memory:
            verdict = self.evaluate_fn(statement)
            confidence = self.confidence_fn(statement)
            self.memory[key] = (verdict, confidence)
        return self.memory[key]

# ==========================
# Versioned cache (SQLite) — evaluations & code mod history
# ==========================
class VersionedPEACECache:
    def __init__(self, db_path: str = "peace_cache.db"):
        self.cache: Dict[str, List[Tuple[TV, float, str]]] = defaultdict(list)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY,
                    statement_hash TEXT,
                    statement TEXT,
                    verdict TEXT,
                    confidence REAL,
                    perspective TEXT,
                    version TEXT,
                    timestamp REAL
                );

                CREATE TABLE IF NOT EXISTS code_modifications (
                    id INTEGER PRIMARY KEY,
                    modification_id TEXT UNIQUE,
                    target_module TEXT,
                    target_function TEXT,
                    original_code TEXT,
                    modified_code TEXT,
                    safety_verdict TEXT,
                    safety_confidence REAL,
                    mathematical_soundness REAL,
                    reasoning TEXT,
                    timestamp REAL
                );

                CREATE INDEX IF NOT EXISTS idx_statement_hash ON evaluations(statement_hash);
                CREATE INDEX IF NOT EXISTS idx_modification_id ON code_modifications(modification_id);
            """)
            self.conn.commit()

    def record_evaluation(
        self, statement: Any, verdict: TV, confidence: float, perspective: str, version: str = "1.0"
    ):
# ==========================
# Safety: AST checks + sandboxed module loader
# ==========================
import types
import builtins as _builtins

class SafeCodeError(Exception):
    pass

class SafeASTChecker(ast.NodeVisitor):
    """Reject dangerous imports, attributes, and builtins usage (static check)."""
    def __init__(self, policy: EnginePolicy):
        self.policy = policy
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self.policy.banned_modules:
                self.errors.append(f"Import of banned module: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in self.policy.banned_modules:
                self.errors.append(f"Import-from banned module: {node.module}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.policy.banned_builtins:
            self.errors.append(f"Use of banned builtin: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block dunder spelunking like obj.__dict__ / __class__
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute access: {node.attr}")
        self.generic_visit(node)

SAFE_BUILTINS: Tuple[str, ...] = (
    # arithmetic / containers / iteration
    "abs","all","any","bool","enumerate","filter","float","int","len","list","max","min",
    "pow","range","reversed","round","set","slice","sorted","str","sum","tuple","zip",
    # conversion / handy
    "chr","ord","bin","hex","oct","hash","divmod","isinstance",
)

def check_code_safety(code: str, policy: EnginePolicy) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SafeCodeError(f"Syntax error: {e}") from e
    checker = SafeASTChecker(policy)
    checker.visit(tree)
    if checker.errors:
        raise SafeCodeError(" ; ".join(checker.errors))

def build_safe_globals() -> Dict[str, Any]:
    allowed: Dict[str, Any] = {}
    for name in SAFE_BUILTINS:
        if hasattr(_builtins, name):
            allowed[name] = getattr(_builtins, name)
    # Do NOT expose __import__ or other banned names
    return {"__builtins__": allowed}

def load_module_safely(code: str, module_name: str, policy: EnginePolicy) -> types.ModuleType:
    check_code_safety(code, policy)
    globs = build_safe_globals()
    locs: Dict[str, Any] = {}
    compiled = compile(code, filename=f"<{module_name}>", mode="exec")
    exec(compiled, globs, locs)  # sandboxed builtins only
    mod = types.ModuleType(module_name)
    for k, v in locs.items():
        setattr(mod, k, v)
    # required API
    for required in ("generate_instances", "solve_instance"):
        if not hasattr(mod, required):
            raise SafeCodeError(f"Solver module missing required symbol: {required}")
    return mod

# ==========================
# LLM Adapter (stub; swap with your real LLM)
# ==========================
@dataclass
class SolverPatch:
    patch_id: str
    module_code: str
    reason: str
    timestamp: float = field(default_factory=time.time)

class LLMAdapter:
    """
    Real implementation should call your LLM to:
      • identify_problem(problem) -> problem (optional refinement)
      • propose_initial_solver(problem) -> SolverPatch (module with generate_instances/solve_instance)
      • analyze_failure(problem, logs) -> SolverPatch (targeted improvements)
      • code_review(patch) -> str (optional)
      • detect_flags(results/logs) -> (solution_found:bool, counterexample:bool, reason:str)
    The stub below simulates behavior for demonstration.
    """

    # Minimal deterministic ID helper
    def _pid(self, *parts: str, n: int = 12) -> str:
        h = hashlib.sha256(("||".join(parts) + f"|{time.time()}").encode()).hexdigest()
        return h[:n]

    def identify_problem(self, problem: MathematicalProblem) -> MathematicalProblem:
        return problem  # no-op in stub

    def propose_initial_solver(self, problem: MathematicalProblem) -> SolverPatch:
        # Baseline: decides only for "small" size (budget cooperative), else returns None.
        code = r'''
def _lcg(seed):
    a=1664525; c=1013904223; m=2**32
    return (a*seed + c) % m

def generate_instances(seed, sizes):
    # Produce one randomized instance per size using a tiny LCG (no imports).
    out=[]
    s=int(seed)
    for n in sizes:
        s=_lcg(s)
        out.append({"size": int(n), "seed": int(s)})
    return out

def solve_instance(instance):
    """
    Contract:
      return ("YES"|"NO"|None, stats_dict)
      - return None for budget/inconclusive (never busy-loop)
      - set stats["flag"]="solution_found" when you have a decisive witness/certificate
      - set stats["flag"]="ill_formed_question" for category errors / counterexample to question
      - include "size" in stats for the engine to update confidence properly
    """
    n = int(instance.get("size", 0))
    # Baseline heuristic: handle small n, abstain otherwise.
    if n <= 5000:
        # pretend we derived a valid decision quickly
        return ("YES", {"size": n, "ops": n, "reason": "baseline-heuristic"})
    return (None, {"size": n, "reason": "budget-exhausted"})
'''
        return SolverPatch(self._pid(problem.name, "init"), code, "Initial budget-cooperative baseline")

    def analyze_failure(self, problem: MathematicalProblem, logs: List[Dict[str, Any]]) -> SolverPatch:
        # Read last abstentions and widen the 'small' cutoff.
        code = r'''
def _lcg(seed):
    a=1664525; c=1013904223; m=2**32
    return (a*seed + c) % m

def generate_instances(seed, sizes):
    out=[]
    s=int(seed)
    for n in sizes:
        s=_lcg(s)
        out.append({"size": int(n), "seed": int(s)})
    return out

def solve_instance(instance):
    n = int(instance.get("size", 0))
    # Improved heuristic range; still abstain beyond the new bound.
    if n <= 20000:
        # pretend we produce a certificate for the decision
        return ("YES", {"size": n, "ops": n//2+1, "flag": "solution_found", "reason": "improved-heuristic"})
    return (None, {"size": n, "reason": "budget-exhausted"})
'''
        return SolverPatch(self._pid(problem.name, "patch"), code, "Improve heuristic capacity & add solution flag")

    def code_review(self, patch: SolverPatch) -> str:
        return "Looks budget-cooperative; uses only safe builtins; returns None when unsure."

    def detect_flags(self, rows: List[Dict[str, Any]]) -> Tuple[bool, bool, str]:
        sol = any(((r.get("stats") or {}).get("flag") == "solution_found") for r in rows)
        bad = any(((r.get("stats") or {}).get("flag") == "ill_formed_question") for r in rows)
        reason = "solver-flagged-solution" if sol else ("ill-formed-question" if bad else "")
        return sol, bad, reason

# ==========================
# Budgeting & RNG helpers
# ==========================
class BudgetTimer:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.start = time.time()
    def exhausted(self) -> bool:
        return (time.time() - self.start) >= self.seconds

def _extract_size(instance: Dict[str, Any], stats: Dict[str, Any]) -> int:
    # Prefer explicit "size"; fall back to "n" if provided; else 0.
    if isinstance(stats, dict) and "size" in stats:
        return int(stats["size"])
    if isinstance(instance, dict):
        if "size" in instance: return int(instance["size"])
        if "n" in instance: return int(instance["n"])
    return 0

# ==========================
# PEACE Meta Engine (general)
# ==========================
class PeaceMetaEngine:
    def __init__(self, policy: EnginePolicy | None = None, llm: LLMAdapter | None = None):
        self.policy = policy or EnginePolicy()
        self.llm = llm or LLMAdapter()
        self.state = EpistemicState()
        self.cache = VersionedPEACECache()
        self.current_module: Optional[types.ModuleType] = None
        self.current_patch_id: Optional[str] = None
        self.logs: List[Dict[str, Any]] = []

    # ----- confidence dynamics -----
    def _reward(self, size: int) -> None:
        alpha = self.policy.alpha_large if size >= self.policy.large_threshold_n else self.policy.alpha_small
        self.state.truth.support(alpha)

    def collapse_up(self, reason: str) -> None:
        if self.policy.collapse_on_solution:
            self.state.collapse_up(reason, to_value=self.policy.solution_value)
        self.logs.append({"event": "collapse_up", "reason": reason, "confidence": self.state.truth.value})

    def collapse_down(self, reason: str) -> None:
        if self.policy.collapse_on_counterexample:
            self.state.collapse_down(reason, to_value=self.policy.counterexample_value)
        self.logs.append({"event": "collapse_down", "reason": reason, "confidence": self.state.truth.value})

    # ----- solver loading -----
    def _load_patch(self, patch: SolverPatch) -> None:
        mod = load_module_safely(patch.module_code, f"solver_{patch.patch_id}", self.policy)
        self.current_module = mod
        self.current_patch_id = patch.patch_id
        self.logs.append({"event": "patch_loaded", "patch_id": patch.patch_id, "reason": patch.reason})

    # ----- one attempt round -----
    def _attempt_round(self, sizes: List[int], seed: int) -> Dict[str, Any]:
        if not self.current_module:
            return {"round_ok": False, "reason": "no_module"}

        gen = getattr(self.current_module, "generate_instances", None)
        solve = getattr(self.current_module, "solve_instance", None)
        if not (callable(gen) and callable(solve)):
            return {"round_ok": False, "reason": "bad_module"}

        timer = BudgetTimer(self.policy.time_budget_sec)

        # Randomize order each round (reproducible by seed)
        sizes = list(sizes)
        # simple shuffle using LCG (no random import)
        def _lcg_shuffle(arr: List[int], s: int) -> None:
            for i in range(len(arr)-1, 0, -1):
                s = (1664525*s + 1013904223) % (2**32)
                j = s % (i+1)
                arr[i], arr[j] = arr[j], arr[i]
        _lcg_shuffle(sizes, seed)

        try:
            instances = gen(seed, sizes)
        except Exception as e:
            return {"round_ok": False, "reason": f"gen_error: {e}"}

        rows: List[Dict[str, Any]] = []
        for inst in instances:
            if timer.exhausted():
                rows.append({"instance": inst, "decision": None, "stats": {"size": _extract_size(inst, {}), "reason": "budget-exhausted"}})
                break
            try:
                decision, stats = solve(inst)
            except Exception as e:
                decision, stats = None, {"error": str(e)}
            size = _extract_size(inst, stats)
            row = {"patch_id": self.current_patch_id, "decision": decision, "stats": stats, "size": size}
            rows.append(row)
            self.logs.append(row)
            if decision in ("YES","NO"):
                self._reward(size)

        # After round, check flags
        solution_flag, bad_flag, why = self.llm.detect_flags(rows)
        return {"round_ok": True, "results": rows, "solution_flag": solution_flag, "bad_flag": bad_flag, "reason": why}

    # ----- main solve loop -----
    def solve(self, problem: MathematicalProblem, seed: int = 2025) -> Dict[str, Any]:
        prob = self.llm.identify_problem(problem)

        # Initial patch
        init = self.llm.propose_initial_solver(prob)
        self._load_patch(init)

        # Build a size plan from computational_bound (generic heuristic)
        bound = max(10, int(prob.computational_bound))
        candidate_sizes = sorted({max(10, bound//10), bound, min(bound*10, bound*50), max(100, bound*2)})
        per_round = max(1, self.policy.samples_per_round)

        for round_idx in range(self.policy.patch_rounds_max + 1):
            sizes = candidate_sizes[:per_round]
            out = self._attempt_round(sizes, seed + round_idx)

            # Check collapses
            if out.get("bad_flag"):
                self.collapse_down(out.get("reason","ill-posed"))
                return {
                    "ok": False, "collapsed": "down",
                    "confidence": self.state.truth.value, "label": self.state.truth.label(),
                    "round": round_idx, "reason": out.get("reason",""),
                    "logs": self.logs[-10:], "results": out.get("results",[])
                }

            results = out.get("results", [])
            decisions = [r.get("decision") for r in results]

            if out.get("solution_flag") or (decisions and all(d in ("YES","NO") for d in decisions)):
                self.collapse_up(out.get("reason","solution_found" if out.get("solution_flag") else "all_decided"))
                return {
                    "ok": True, "collapsed": "up", "solved": True,
                    "confidence": self.state.truth.value, "label": self.state.truth.label(),
                    "round": round_idx, "results": results
                }

            # Not decisive: request a patch if we have budgeted rounds left
            if round_idx < self.policy.patch_rounds_max:
                patch = self.llm.analyze_failure(prob, self.logs)
                _ = self.llm.code_review(patch)
                try:
                    self._load_patch(patch)
                except SafeCodeError as e:
                    self.logs.append({"event":"patch_rejected","error":str(e),"patch_id":patch.patch_id})
                    # continue with previous module if any; else stop neutrally
                    if not self.current_module:
                        break
            else:
                break

        # Neutral finish (budget exhausted / partial progress)
        return {
            "ok": True, "collapsed": None, "solved": False,
            "confidence": self.state.truth.value, "label": self.state.truth.label(),
            "reason": "budget_exhausted_or_partial_progress",
            "logs": self.logs[-10:]
        }
# ==========================
# CLI + Demo runner
# ==========================
import argparse
import sys

def _build_policy_from_args(args: argparse.Namespace) -> EnginePolicy:
    return EnginePolicy(
        time_budget_sec=args.time_budget,
        samples_per_round=args.samples,
        patch_rounds_max=args.patch_rounds,
        alpha_small=args.alpha_small,
        alpha_large=args.alpha_large,
        large_threshold_n=args.large_threshold,
        collapse_on_solution=not args.no_solution_collapse,
        solution_value=args.solution_value,
        collapse_on_counterexample=not args.no_counterexample_collapse,
        counterexample_value=args.counterexample_value,
    )

def _make_preset_problem(preset: str) -> MathematicalProblem:
    preset = (preset or "").lower().strip()
    if preset in ("goldbach", "goldbach_demo", "gb"):
        return MathematicalProblem(
            name="goldbach_demo",
            description="Demo family standing in for Goldbach-style instances; LLM supplies solver strategy.",
            complexity_score=8,
            computational_bound=20_000,   # engine will try sizes around this
            problem_type="number_theory",
            verification_function=None,
        )
    if preset in ("sat", "sat_demo"):
        return MathematicalProblem(
            name="sat_demo",
            description="Demo family for SAT-like decision tasks with budget-cooperative solvers.",
            complexity_score=9,
            computational_bound=10_000,
            problem_type="combinatorics",
            verification_function=None,
        )
    if preset in ("crypto", "crypto_demo"):
        return MathematicalProblem(
            name="crypto_demo",
            description="Demo family for crypto-hardness style instances (factor-like toy).",
            complexity_score=9,
            computational_bound=15_000,
            problem_type="crypto",
            verification_function=None,
        )
    # default
    return MathematicalProblem(
        name="generic_p_vs_np_demo",
        description="Generic family; the LLM adapter will provide a baseline & patches.",
        complexity_score=7,
        computational_bound=12_000,
        problem_type="general",
        verification_function=None,
    )

def _make_custom_problem(args: argparse.Namespace) -> MathematicalProblem:
    return MathematicalProblem(
        name=args.name or "custom_demo",
        description=args.desc or "User-specified family; the LLM adapter supplies solvers.",
        complexity_score=max(1, min(10, int(args.complexity))),
        computational_bound=max(10, int(args.bound)),
        problem_type=args.ptype or "general",
        verification_function=None,
    )

def _pretty_print(out: Dict[str, Any]) -> None:
    print("ok:", out.get("ok"))
    print("collapsed:", out.get("collapsed"))
    if out.get("solved") is not None:
        print("solved:", out.get("solved"))
    print("confidence:", f"{out.get('confidence', 0.0):.6f}", "-", out.get("label"))
    if "reason" in out and out["reason"]:
        print("reason:", out["reason"])
    results = out.get("results") or []
    if results:
        print("results (up to 10 rows):")
        for r in results[:10]:
            print("  decision=", r.get("decision"),
                  " size=", r.get("size"),
                  " stats=", r.get("stats"))
        if len(results) > 10:
            print(f"  ... ({len(results)-10} more)")
    logs = out.get("logs") or []
    if logs:
        print("logs tail (up to 10):")
        for row in logs[-10:]:
            print("  ", row)

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PEACE Generalized Meta Engine — Dialetheic, Budgeted, Self-editing, LLM-guided"
    )

    # Problem selection
    g_prob = parser.add_argument_group("Problem")
    g_prob.add_argument("--preset", type=str, default="goldbach_demo",
                        help="Preset: goldbach_demo|sat_demo|crypto_demo|generic")
    g_prob.add_argument("--name", type=str, default=None, help="Custom problem name (overrides preset)")
    g_prob.add_argument("--desc", type=str, default=None, help="Custom problem description")
    g_prob.add_argument("--ptype", type=str, default=None, help="Custom problem type (e.g., number_theory, crypto)")
    g_prob.add_argument("--complexity", type=int, default=7, help="1..10 (custom only)")
    g_prob.add_argument("--bound", type=int, default=12000, help="computational_bound (custom only)")

    # Policy / budgets
    g_pol = parser.add_argument_group("Policy/Budget")
    g_pol.add_argument("--time-budget", type=float, default=1.0, help="Per-round wall-clock seconds")
    g_pol.add_argument("--samples", type=int, default=3, help="Instances per round")
    g_pol.add_argument("--patch-rounds", type=int, default=2, help="Max self-edit rounds")
    g_pol.add_argument("--alpha-small", type=float, default=0.05, help="Confidence bump for small size")
    g_pol.add_argument("--alpha-large", type=float, default=0.15, help="Confidence bump for large size")
    g_pol.add_argument("--large-threshold", type=int, default=5000, help="Boundary between small/large")

    # Collapse tuning
    g_col = parser.add_argument_group("Collapses")
    g_col.add_argument("--no-solution-collapse", action="store_true", help="Disable collapse-up on solution")
    g_col.add_argument("--solution-value", type=float, default=1.0 - 1e-6, help="Truth value after solution-collapse")
    g_col.add_argument("--no-counterexample-collapse", action="store_true", help="Disable collapse-down on counterexample")
    g_col.add_argument("--counterexample-value", type=float, default=1e-6, help="Truth value after counterexample-collapse")

    # Misc
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed (deterministic shuffles)")
    parser.add_argument("--use-custom", action="store_true", help="Use custom problem fields instead of --preset")

    args = parser.parse_args(argv)

    # Problem
    problem = _make_custom_problem(args) if args.use_custom else _make_preset_problem(args.preset)
    if args.name:  # allow overriding preset's name/desc/ptype quickly
        problem = MathematicalProblem(
            name=args.name,
            description=args.desc or problem.description,
            complexity_score=problem.complexity_score if args.complexity is None else max(1, min(10, int(args.complexity))),
            computational_bound=problem.computational_bound if args.bound is None else max(10, int(args.bound)),
            problem_type=args.ptype or problem.problem_type,
            verification_function=problem.verification_function,
        )

    # Policy
    policy = _build_policy_from_args(args)

    # Engine
    engine = PeaceMetaEngine(policy=policy, llm=LLMAdapter())

    # Solve
    out = engine.solve(problem, seed=args.seed)

    # Report
    print("\n=== PEACE Meta Result ===")
    print("problem:", problem.name, "| type:", problem.problem_type)
    _pretty_print(out)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
