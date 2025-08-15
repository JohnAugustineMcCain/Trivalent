from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, List, Tuple, Union, Callable

# ----------------------------
# Truth values and operations
# ----------------------------

class TV(Enum):
    T = auto()
    F = auto()
    B = auto()  # both true and false (meta-dialetheic default)

def t(tv: TV) -> int:
    return 1 if tv in (TV.T, TV.B) else 0

def f(tv: TV) -> int:
    return 1 if tv in (TV.F, TV.B) else 0

def neg(a: TV) -> TV:
    # t(¬A) = f(A); f(¬A) = t(A)
    if a is TV.T: return TV.F
    if a is TV.F: return TV.T
    return TV.B  # B stays B

def conj(a: TV, b: TV) -> TV:
    # t(A∧B)=t(A)∧t(B); f(A∧B)=f(A)∨f(B)
    t_val = t(a) & t(b)
    f_val = f(a) | f(b)
    if t_val and not f_val: return TV.T
    if f_val and not t_val: return TV.F
    if t_val and f_val: return TV.B
    return TV.B

def disj(a: TV, b: TV) -> TV:
    # t(A∨B)=t(A)∨t(B); f(A∨B)=f(A)∧f(B)
    t_val = t(a) | t(b)
    f_val = f(a) & f(b)
    if t_val and not f_val: return TV.T
    if f_val and not t_val: return TV.F
    if t_val and f_val: return TV.B
    return TV.B

def impl(a: TV, b: TV) -> TV:
    # A→B := ¬A ∨ B
    return disj(neg(a), b)

DESIGNATED = {TV.T, TV.B}

# ----------------------------
# Formula AST
# ----------------------------

class Formula:
    pass

@dataclass(frozen=True)
class Var(Formula):
    name: str

@dataclass(frozen=True)
class Not(Formula):
    phi: Formula

@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

# Transparent truth predicate True(·)
@dataclass(frozen=True)
class TruePred(Formula):
    phi: Formula

# For modeling self-reference patterns explicitly (e.g., L ≡ ¬True(L))
@dataclass(frozen=True)
class SelfRef(Formula):
    # represents a fixed-point name bound to a body formula using the same SelfRef object
    name: str
    body: Formula

# Optional figurative operator False*(·): true iff phi has any false-content
@dataclass(frozen=True)
class FalseStar(Formula):
    phi: Formula

# ----------------------------
# Valuation and evaluation
# ----------------------------

Valuation = Dict[str, TV]

def eval_formula(phi: Formula, val: Valuation) -> TV:
    if isinstance(phi, Var):
        return val.get(phi.name, TV.B)  # meta-dialetheic default
    if isinstance(phi, Not):
        return neg(eval_formula(phi.phi, val))
    if isinstance(phi, And):
        return conj(eval_formula(phi.left, val), eval_formula(phi.right, val))
    if isinstance(phi, Or):
        return disj(eval_formula(phi.left, val), eval_formula(phi.right, val))
    if isinstance(phi, Implies):
        return impl(eval_formula(phi.left, val), eval_formula(phi.right, val))
    if isinstance(phi, TruePred):
        inner = eval_formula(phi.phi, val)
        # transparent truth: t(True(phi))=t(phi), f(True(phi))=f(phi)
        if inner is TV.T: return TV.T
        if inner is TV.F: return TV.F
        return TV.B
    if isinstance(phi, FalseStar):
        inner = eval_formula(phi.phi, val)
        # true iff phi has any false-content; otherwise false
        if f(inner) == 1:
            return TV.T
        return TV.F
    if isinstance(phi, SelfRef):
        # Minimal fixed-point semantics: liar-like forms stabilize at B from the neutral start.
        # A fuller implementation could iterate; for PEACE we adopt B as the neutral fixed point.
        return TV.B
    raise TypeError(f"Unsupported formula type: {type(phi)}")

# ----------------------------
# Evidence model
# ----------------------------

@dataclass
class Evidence:
    pos: float = 0.0  # support for truth
    neg: float = 0.0  # support for falsity

def evidence_to_tv(e: Evidence, theta: float = 0.7, theta_gap: float = 0.3) -> TV:
    # simple mapping; tweak thresholds as needed
    if e.pos >= theta and e.neg < theta_gap:
        return TV.T
    if e.neg >= theta and e.pos < theta_gap:
        return TV.F
    if e.pos >= theta and e.neg >= theta:
        return TV.B
    return TV.B

# ----------------------------
# Perspectives
# ----------------------------

@dataclass
class PerspectiveResult:
    value: TV
    admissible: bool
    load_bearing: bool
    reason: str = ""

class Perspective:
    name: str = "base"

    def is_admissible(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        # default: admissible
        return True, "default admissible"

    def is_load_bearing(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        # default heuristic: targets truth predicate
        targets_truth = contains_truth_predicate(phi)
        return targets_truth, "targets truth predicate" if targets_truth else "decorative"

    def evaluate(self, phi: Formula, val: Valuation, context: Dict) -> PerspectiveResult:
        v = eval_formula(phi, val)
        adm, awhy = self.is_admissible(phi, context)
        lb, lwhy = self.is_load_bearing(phi, context)
        return PerspectiveResult(value=v, admissible=adm, load_bearing=lb, reason=f"{awhy}; {lwhy}")

class MetaPerspective(Perspective):
    name = "meta"
    # transparent truth; base engine already encodes it

class PragmaticPerspective(Perspective):
    name = "pragmatic"
    def is_load_bearing(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        # load-bearing for self-falsifying schemas: phi ≡ ¬True(phi) or explicit SelfRef with Not(TruePred(SelfRef))
        has_liar_form = detects_liar(phi)
        return has_liar_form, "self-falsifying assertion" if has_liar_form else "not a self-falsifier"

    def evaluate(self, phi: Formula, val: Valuation, context: Dict) -> PerspectiveResult:
        adm, awhy = self.is_admissible(phi, context)
        lb, lwhy = self.is_load_bearing(phi, context)
        if adm and lb:
            # treat assertion as defective → F
            return PerspectiveResult(TV.F, adm, lb, reason=f"{awhy}; {lwhy}; assertional defect -> F")
        base = eval_formula(phi, val)
        return PerspectiveResult(base, adm, lb, reason=f"{awhy}; {lwhy}; conservative return")

class FigurativePerspective(Perspective):
    name = "figurative"
    def is_load_bearing(self, phi: Formula, context: Dict) -> Tuple[bool, str]:
        # we treat figurative perspective as re-reading True(·) as False*(·)
        has_truth = contains_truth_predicate(phi)
        return has_truth, "reinterprets truth as False*"

    def evaluate(self, phi: Formula, val: Valuation, context: Dict) -> PerspectiveResult:
        adm, awhy = self.is_admissible(phi, context)
        lb, lwhy = self.is_load_bearing(phi, context)
        def rewrite(p: Formula) -> Formula:
            if isinstance(p, TruePred):
                return FalseStar(rewrite(p.phi))
            if isinstance(p, Not): return Not(rewrite(p.phi))
            if isinstance(p, And): return And(rewrite(p.left), rewrite(p.right))
            if isinstance(p, Or): return Or(rewrite(p.left), rewrite(p.right))
            if isinstance(p, Implies): return Implies(rewrite(p.left), rewrite(p.right))
            if isinstance(p, (Var, SelfRef, FalseStar)): return p
            return p
        value = eval_formula(rewrite(phi), val) if lb else eval_formula(phi, val)
        return PerspectiveResult(value, adm, lb, reason=f"{awhy}; {lwhy}; figurative rewrite applied")

# ----------------------------
# Utilities: structure checks
# ----------------------------

def contains_truth_predicate(phi: Formula) -> bool:
    if isinstance(phi, TruePred): return True
    if isinstance(phi, Var): return False
    if isinstance(phi, Not): return contains_truth_predicate(phi.phi)
    if isinstance(phi, (And, Or, Implies)):
        return contains_truth_predicate(phi.left) or contains_truth_predicate(phi.right)
    if isinstance(phi, SelfRef):
        return contains_truth_predicate(phi.body)
    if isinstance(phi, FalseStar):
        return contains_truth_predicate(phi.phi)
    return False

def detects_liar(phi: Formula) -> bool:
    # heuristic: Not(TruePred(X)) where X syntactically equals the whole formula (SelfRef)
    if isinstance(phi, SelfRef):
        return _body_is_liar(phi.body, phi)
    return _body_is_liar(phi, phi)

def _body_is_liar(body: Formula, self_node: Formula) -> bool:
    if isinstance(body, Not) and isinstance(body.phi, TruePred):
        target = body.phi.phi
        return target == self_node
    if isinstance(body, Not): return _body_is_liar(body.phi, self_node)
    if isinstance(body, (And, Or, Implies)):
        return _body_is_liar(body.left, self_node) or _body_is_liar(body.right, self_node)
    if isinstance(body, TruePred):
        return _body_is_liar(body.phi, self_node)
    if isinstance(body, SelfRef):
        return _body_is_liar(body.body, body)
    return False

# ----------------------------
# Context & selection
# ----------------------------

@dataclass
class Context:
    name: str
    relevance_order: List[str]  # list of perspective names from most to least relevant
    meta: Dict = None
    C_c: float = 0.5  # context completeness score

def select_verdict(phi: Formula,
                   val: Valuation,
                   context: Context,
                   perspectives: List[Perspective],
                   set_valued: bool = False) -> Tuple[TV, Dict[str, PerspectiveResult]]:
    results: Dict[str, PerspectiveResult] = {}
    influential: Dict[str, PerspectiveResult] = {}
    for p in perspectives:
        res = p.evaluate(phi, val, context.meta or {})
        results[p.name] = res
        if res.admissible and res.load_bearing:
            influential[p.name] = res
    if not influential:
        base = eval_formula(phi, val)
        return base, results
    order = context.relevance_order
    if set_valued:
        ranks = {name: order.index(name) if name in order else len(order) for name in influential.keys()}
        min_rank = min(ranks.values())
        winners = [name for name, r in ranks.items() if r == min_rank]
        vals = {influential[w].value for w in winners}
        final = TV.B if vals == {TV.T, TV.F} else (list(vals)[0] if len(vals)==1 else TV.B)
        return final, {k:results[k] for k in winners}
    else:
        for name in order:
            if name in influential:
                return influential[name].value, results
        any_name = next(iter(influential))
        return influential[any_name].value, results

# ----------------------------
# Convenience constructors
# ----------------------------

def liar() -> Formula:
    # L ≡ ¬True(L) modeled via SelfRef
    L = SelfRef("L", None)  # temporary
    object.__setattr__(L, "body", Not(TruePred(L)))
    return L

def pretty(tv: TV) -> str:
    return {TV.T:"T", TV.F:"F", TV.B:"B"}[tv]

# ----------------------------
# Demo helpers
# ----------------------------

def demo_liar():
    L = liar()
    ctx_sem = Context("Seminar", ["meta","pragmatic","figurative"], C_c=0.9, meta={})
    ctx_debate = Context("Debate", ["pragmatic","meta","figurative"], C_c=0.9, meta={})
    perspectives = [MetaPerspective(), PragmaticPerspective(), FigurativePerspective()]
    val: Valuation = {}
    base = eval_formula(L, val)
    v_sem, res_sem = select_verdict(L, val, ctx_sem, perspectives)
    v_deb, res_deb = select_verdict(L, val, ctx_debate, perspectives)
    return {
        "base": pretty(base),
        "seminar": pretty(v_sem),
        "debate": pretty(v_deb),
        "details_seminar": {k:(pretty(r.value), r.reason) for k,r in res_sem.items()},
        "details_debate": {k:(pretty(r.value), r.reason) for k,r in res_deb.items()},
    }
