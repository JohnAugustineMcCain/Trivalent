# Requires peace.py in the same folder.
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import ast, textwrap

from peace import (
    TV, Formula, Var, Not, And, Or, Implies, TruePred, FalseStar, SelfRef,
    Valuation, Evidence, evidence_to_tv, Perspective, PerspectiveResult,
    Context, select_verdict, eval_formula, DESIGNATED, pretty
)

# ----------------------------
# Claim wrapper for termination
# ----------------------------

@dataclass(frozen=True)
class HaltingClaim(Formula):
    """
    Represents: 'Program P halts (on given inputs/assumptions)'.
    `code` is a Python snippet or function source. `inputs` optional.
    `assumptions` may include invariants, pre/post conditions, or environment constraints.
    """
    code: str
    inputs: Optional[Dict[str, Any]] = None
    assumptions: Optional[Dict[str, Any]] = None

# ----------------------------
# Utils: lightweight static inspection
# ----------------------------

@dataclass
class LoopInfo:
    kind: str              # 'for' or 'while'
    header: str
    body_text: str
    heuristic: str         # classification reason

def _extract_loops(py_src: str) -> List[LoopInfo]:
    src = textwrap.dedent(py_src)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [LoopInfo("syntax-error", "", "", f"SyntaxError: {e}")]
    loops: List[LoopInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            target = ast.unparse(node.target) if hasattr(ast, "unparse") else "iter"
            it = ast.unparse(node.iter) if hasattr(ast, "unparse") else "iter"
            body = ast.get_source_segment(src, node) or ""
            # Heuristic: for over range with finite bounds looks terminating
            finite = "range(" in it and not any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "iter" for n in ast.walk(node.iter))
            loops.append(LoopInfo("for", f"for {target} in {it}", body, "finite-range" if finite else "unknown-iterator"))
        if isinstance(node, ast.While):
            test = ast.unparse(node.test) if hasattr(ast, "unparse") else "cond"
            body = ast.get_source_segment(src, node) or ""
            # Heuristic: while True without break is suspicious
            pure_true = isinstance(node.test, ast.Constant) and node.test.value is True
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
            if pure_true and not has_break:
                kind = "while"
                loops.append(LoopInfo(kind, f"while True", body, "infinite-without-break"))
            else:
                # Look for monotone counter update toward a linear inequality
                # (very rough heuristic)
                assignments = [n for n in ast.walk(node) if isinstance(n, ast.AugAssign)]
                monotone = any(isinstance(a.op, (ast.Sub,)) or isinstance(a.op, (ast.Add,)) for a in assignments)
                loops.append(LoopInfo("while", f"while {test}", body, "monotone-counter" if monotone else "unknown-condition"))
    return loops

# ----------------------------
# Perspectives for halting
# ----------------------------

class StaticAnalysisPerspective(Perspective):
    """
    Admissible if the claim is a HaltingClaim.
    Load-bearing if code contains loops/recursion we can classify.
    Verdict:
      - T for clearly finite loops (for-range with concrete finite bounds; while with obvious decreasing counter)
      - F for 'while True' with no break/return/raise
      - B otherwise
    """
    name = "static"

    def is_admissible(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        return (isinstance(phi, HaltingClaim), "targets program text")

    def is_load_bearing(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        if not isinstance(phi, HaltingClaim):
            return (False, "not a halting claim")
        loops = _extract_loops(phi.code)
        return (len(loops) > 0 and loops[0].kind != "syntax-error", f"found {len(loops)} loop(s)")

    def evaluate(self, phi: Formula, val: Valuation, context: Dict) -> PerspectiveResult:
        adm, awhy = self.is_admissible(phi, context)
        if not adm:
            base = TV.B
            return PerspectiveResult(base, adm, False, reason=f"{awhy}")
        loops = _extract_loops(phi.code)
        if any(li.kind == "syntax-error" for li in loops):
            return PerspectiveResult(TV.F, True, False, "syntax error: cannot run")
        # Simple rules
        if loops and all(li.kind == "for" and li.heuristic == "finite-range" for li in loops):
            return PerspectiveResult(TV.T, True, True, "all loops finite over range(...)")
        if any(li.kind == "while" and li.heuristic == "infinite-without-break" for li in loops):
            return PerspectiveResult(TV.F, True, True, "while True without break")
        if any(li.kind == "while" and li.heuristic == "monotone-counter" for li in loops):
            return PerspectiveResult(TV.T, True, True, "monotone counter heuristic")
        return PerspectiveResult(TV.B, True, True, "structure ambiguous; default B")

class EmpiricalBoundedPerspective(Perspective):
    """
    Admissible if HaltingClaim and sandbox/run allowance is present in context.
    Load-bearing if we can run with bounded steps (e.g., through an interpreter you provide).
    Verdict:
      - T if program halts within step/time budget for provided inputs
      - F if exceeds budget with observed non-decreasing state (heuristic)
      - B otherwise
    NOTE: This class does not execute code by default (safety). You can inject a runner.
    """
    name = "empirical"

    def is_admissible(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        return (isinstance(phi, HaltingClaim), "targets runtime behavior")

    def is_load_bearing(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        runner = context.get("runner")
        budget = context.get("budget_steps", 100000)
        return (runner is not None and budget > 0, "runner and budget provided" if runner else "no runner; decorative")

    def evaluate(self, phi: Formula, val: Valuation, context: Dict) -> PerspectiveResult:
        adm, awhy = self.is_admissible(phi, context)
        lb, lwhy = self.is_load_bearing(phi, context)
        if not (adm and lb):
            return PerspectiveResult(TV.B, adm, lb, reason=f"{awhy}; {lwhy}")
        runner = context["runner"]
        budget = context.get("budget_steps", 100000)
        inputs = (phi.inputs or {})
        try:
            outcome = runner(code=phi.code, inputs=inputs, budget=budget)
        except Exception as e:
            return PerspectiveResult(TV.B, True, True, reason=f"runner error: {e}")
        if outcome == "halt":
            return PerspectiveResult(TV.T, True, True, reason=f"halted within {budget} steps")
        if outcome == "nonhalt":
            return PerspectiveResult(TV.F, True, True, reason=f"exceeded budget with non-decreasing trace")
        return PerspectiveResult(TV.B, True, True, reason=f"inconclusive runner outcome: {outcome}")

# ----------------------------
# LLM perspective (stubbed)
# ----------------------------

class LLMClient:
    """
    Implement `complete(prompt: str) -> str` for your provider.
    This is a stub to keep the module provider-agnostic.
    """
    def complete(self, prompt: str) -> str:
        raise NotImplementedError("Plug in your LLM provider here.")

class LLMInvariantPerspective(Perspective):
    """
    Uses an LLM to propose a ranking function and loop invariant.
    Admissible for HaltingClaim; load-bearing if loops detected.
    Verdict:
      - T if proposed ranking function is syntactically decreasing toward a well-founded set
      - F if LLM identifies explicit non-termination pattern (e.g., oscillation with no measure)
      - B otherwise
    """
    name = "llm-invariant"

    def __init__(self, client: LLMClient):
        self.client = client

    def is_admissible(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        return (isinstance(phi, HaltingClaim), "targets invariants/ranking")

    def is_load_bearing(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        if not isinstance(phi, HaltingClaim):
            return (False, "not a halting claim")
        loops = _extract_loops(phi.code)
        return (len(loops) > 0, f"found {len(loops)} loop(s)")

    def evaluate(self, phi: Formula, val: Valuation, context: Dict) -> PerspectiveResult:
        adm, awhy = self.is_admissible(phi, context)
        lb, lwhy = self.is_load_bearing(phi, context)
        if not adm:
            return PerspectiveResult(TV.B, adm, lb, reason=f"{awhy}")
        # Build a prompt that asks for invariant + ranking function + justification
        prompt = (
            "You are checking loop termination.\n"
            "Return JSON with fields: {invariant, ranking_function, well_founded_set, verdict, justification}.\n"
            "Verdict must be one of: T (halts), F (does not halt), B (unknown).\n\n"
            f"Code:\n{phi.code}\n\n"
            f"Assumptions: {phi.assumptions or {}}\n"
        )
        try:
            raw = self.client.complete(prompt)
        except Exception as e:
            return PerspectiveResult(TV.B, True, lb, reason=f"LLM error: {e}")
        # naive parse; caller can ensure valid JSON in practice
        import json
        try:
            data = json.loads(raw)
        except Exception:
            return PerspectiveResult(TV.B, True, lb, reason="LLM non-JSON / unparsable")
        vmap = {"T": TV.T, "F": TV.F, "B": TV.B}
        verdict = vmap.get(str(data.get("verdict", "B")).strip().upper(), TV.B)
        # sanity check: if ranking function and well-founded set present, nudge T unless justification says otherwise
        if verdict is TV.B and data.get("ranking_function") and data.get("well_founded_set"):
            verdict = TV.T
        reason = f"inv={data.get('invariant')}; rank={data.get('ranking_function')} on {data.get('well_founded_set')}; just={data.get('justification')}"
        return PerspectiveResult(verdict, True, lb, reason=reason)

# ----------------------------
# Orchestrator
# ----------------------------

@dataclass
class HaltingResult:
    final: TV
    per_perspective: Dict[str, PerspectiveResult]

def analyze_halting(code: str,
                    context: Context,
                    perspectives: List[Perspective],
                    inputs: Optional[Dict[str, Any]] = None,
                    assumptions: Optional[Dict[str, Any]] = None,
                    set_valued: bool = False) -> HaltingResult:
    claim = HaltingClaim(code=code, inputs=inputs, assumptions=assumptions)
    v, results = select_verdict(claim, {}, context, perspectives, set_valued=set_valued)
    return HaltingResult(final=v, per_perspective=results)

# ----------------------------
# Example usage
# ----------------------------

class DummyLLM(LLMClient):
    """Example no-API LLM stub that 'recognizes' common patterns."""
    def complete(self, prompt: str) -> str:
        # Extremely naive: if it sees while True without break, call F.
        import json, re
        if "while True" in prompt and "break" not in prompt:
            return json.dumps({
                "invariant":"True",
                "ranking_function": None,
                "well_founded_set": None,
                "verdict":"F",
                "justification":"Unconditional loop with no break/return."
            })
        if "range(" in prompt:
            return json.dumps({
                "invariant":"i in [start,end)",
                "ranking_function":"end - i",
                "well_founded_set":"N (>=0)",
                "verdict":"T",
                "justification":"For-loop with finite bounds decreases measure to 0."
            })
        return json.dumps({
            "invariant": None,
            "ranking_function": None,
            "well_founded_set": None,
            "verdict":"B",
            "justification":"Insufficient structure."
        })

def _demo():
    code1 = """
def f(n):
    s=0
    for i in range(n):
        s += i
    return s
"""
    code2 = """
def spin():
    while True:
        pass
"""
    ctx = Context(
        name="HaltingAnalysis",
        relevance_order=["static","llm-invariant","empirical","meta","figurative","pragmatic"],
        C_c=0.8,
        meta={"runner": None, "budget_steps": 10000}  # no runner by default
    )
    perspectives = [StaticAnalysisPerspective(), LLMInvariantPerspective(DummyLLM()), EmpiricalBoundedPerspective()]
    r1 = analyze_halting(code1, ctx, perspectives)
    r2 = analyze_halting(code2, ctx, perspectives)
    def summarize(hr: HaltingResult):
        return {
            "final": pretty(hr.final),
            "perspectives": {k: {"value": pretty(v.value), "admissible": v.admissible, "load_bearing": v.load_bearing, "reason": v.reason}
                             for k,v in hr.per_perspective.items()}
        }
    return {"finite_for": summarize(r1), "while_true": summarize(r2)}

if __name__ == "__main__":
    import json
    print(json.dumps(_demo(), indent=2))
