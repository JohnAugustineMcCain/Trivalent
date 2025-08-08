# Trivalent
A self-editing, dialetheic-perspectival AI foundation for Artificial Life
import os
import json
import time
import uuid
import random
import datetime
import ast
from functools import reduce
from enum import Enum
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field

# ============================
#  TRI-VALUED LOGIC PRIMITIVES
# ============================

class Tvalue(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2

    def __invert__(self):  # NOT
        lookup = {
            Tvalue.FALSE: Tvalue.TRUE,
            Tvalue.TRUE: Tvalue.FALSE,
            Tvalue.BOTH: Tvalue.BOTH
        }
        return lookup[self]

    def __and__(self, other):
        lookup = {
            (Tvalue.TRUE, Tvalue.TRUE): Tvalue.TRUE,
            (Tvalue.TRUE, Tvalue.BOTH): Tvalue.BOTH,
            (Tvalue.BOTH, Tvalue.TRUE): Tvalue.BOTH,
            (Tvalue.BOTH, Tvalue.BOTH): Tvalue.BOTH,
        }
        return lookup.get((self, other), Tvalue.FALSE)

    def __or__(self, other):
        lookup = {
            (Tvalue.FALSE, Tvalue.FALSE): Tvalue.FALSE,
            (Tvalue.FALSE, Tvalue.BOTH): Tvalue.BOTH,
            (Tvalue.BOTH, Tvalue.FALSE): Tvalue.BOTH,
            (Tvalue.BOTH, Tvalue.BOTH): Tvalue.BOTH,
        }
        return lookup.get((self, other), Tvalue.TRUE)


def parse_logical_expression(expr: str) -> Callable[[Dict[str, Tvalue]], Tvalue]:
    """
    Parse pythonic boolean expression over variable names using tri-valued logic.
    Operators: and, or, not
    Unknown vars evaluate to BOTH.
    """
    expr_ast = ast.parse(expr, mode='eval')

    def make_eval(node):
        if isinstance(node, ast.BoolOp):
            op = Tvalue.__and__ if isinstance(node.op, ast.And) else Tvalue.__or__
            children = [make_eval(v) for v in node.values]
            return lambda context: reduce(op, [child(context) for child in children])
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            child = make_eval(node.operand)
            return lambda context: ~child(context)
        elif isinstance(node, ast.Name):
            return lambda context: context.get(node.id, Tvalue.BOTH)
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    return make_eval(expr_ast.body)


# ============================
#  PERSPECTIVES
# ============================

@dataclass
class Perspective:
    name: str
    evaluate_statement_fn: Callable[[str], Tvalue]
    evaluate_patch_fn: Optional[Callable[['Patch'], Tvalue]] = None
    memory: Dict[str, Tvalue] = field(default_factory=dict)
    reliability: float = 0.5  # online-learned weight in [0,1]
    protected: bool = False   # if True, a FALSE vote can block deployment

    def evaluate(self, statement: str) -> Tvalue:
        if statement not in self.memory:
            self.memory[statement] = self.evaluate_statement_fn(statement)
        return self.memory[statement]

    def evaluate_patch(self, patch: 'Patch') -> Tvalue:
        if self.evaluate_patch_fn is None:
            return Tvalue.BOTH
        return self.evaluate_patch_fn(patch)

    def merge_with(self, other: 'Perspective', new_name: str) -> 'Perspective':
        def composed_eval(statement: str) -> Tvalue:
            v1 = self.evaluate(statement)
            v2 = other.evaluate(statement)
            return v1 if v1 == v2 else Tvalue.BOTH
        return Perspective(name=new_name, evaluate_statement_fn=composed_eval)

    def update_reliability(self, was_correct: bool, lr: float = 0.05):
        if was_correct:
            self.reliability = min(1.0, self.reliability + lr * (1 - self.reliability))
        else:
            self.reliability = max(0.0, self.reliability - lr * self.reliability)


# ============================
#  PERSPECTIVE MANAGER (dynamic)
# ============================

@dataclass
class PerspectiveStats:
    name: str
    wins: int = 0          # agreed with non-BOTH consensus
    losses: int = 0        # disagreed with non-BOTH consensus
    abstains: int = 0      # BOTH when consensus not BOTH
    last_seen: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    @property
    def samples(self) -> int:
        return self.wins + self.losses + self.abstains

    def as_dict(self):
        return {
            "wins": self.wins, "losses": self.losses, "abstains": self.abstains,
            "last_seen": self.last_seen.isoformat()
        }

class PerspectiveManager:
    def __init__(self, perspectives: List[Perspective]):
        self._persps: Dict[str, Perspective] = {p.name: p for p in perspectives}
        self._stats: Dict[str, PerspectiveStats] = {p.name: PerspectiveStats(p.name) for p in perspectives}

        # knobs
        self.reliability_floor = 0.15         # retire below this (after warmup)
        self.reliability_ceiling = 0.95       # optional cap
        self.min_samples_for_retire = 50      # don't retire too soon
        self.decay_rate = 0.002               # reliability decay per tick (prevents lock-in)
        self.max_pool = 24                    # hard cap
        self.min_pool = 6                     # don't drop below this

        # inject callbacks so Proposition can notify us
        for p in self._persps.values():
            def _mk_cb(name):
                def _cb(was_correct: bool, abstained: bool):
                    self.record_outcome(name, was_correct, abstained)
                return _cb
            p._mgr_record = _mk_cb(p.name)  # type: ignore[attr-defined]

    # ---- CRUD ----
    def list(self) -> List[Perspective]:
        return list(self._persps.values())

    def get(self, name: str) -> Perspective:
        return self._persps[name]

    def add(self, p: Perspective):
        if p.name in self._persps:
            raise ValueError(f"Perspective {p.name} already exists")
        self._persps[p.name] = p
        self._stats[p.name] = PerspectiveStats(p.name)
        # inject callback
        def _mgr_record(was_correct: bool, abstained: bool):
            self.record_outcome(p.name, was_correct, abstained)
        p._mgr_record = _mgr_record  # type: ignore[attr-defined]

    def retire(self, name: str):
        self._persps.pop(name, None)
        self._stats.pop(name, None)

    # ---- lifecycle & stats ----
    def record_outcome(self, name: str, was_correct: bool, abstained: bool):
        s = self._stats[name]
        s.last_seen = datetime.datetime.utcnow()
        if abstained:
            s.abstains += 1
        elif was_correct:
            s.wins += 1
        else:
            s.losses += 1

    def periodic_decay(self):
        # small reliability decay to avoid stale dominance
        for p in self._persps.values():
            p.reliability = max(0.0, p.reliability - self.decay_rate * p.reliability)
            if self.reliability_ceiling is not None:
                p.reliability = min(self.reliability_ceiling, p.reliability)

    def auto_prune(self):
        """Retire persistently-bad perspectives, but keep a minimum pool."""
        if len(self._persps) <= self.min_pool:
            return
        bad = []
        for name, p in list(self._persps.items()):
            s = self._stats[name]
            if s.samples < self.min_samples_for_retire:
                continue
            if p.reliability < self.reliability_floor:
                bad.append((p.reliability, name))
        bad.sort()  # lowest first
        to_retire = min(len(bad), max(0, len(self._persps) - self.max_pool) + 2)
        for _, name in bad[:to_retire]:
            self.retire(name)

    # ---- bandit-ish selection (exploit + explore) ----
    def select_subset(self, k: int) -> List[Perspective]:
        all_ps = self.list()
        k = min(k, len(all_ps))
        sorted_ps = sorted(all_ps, key=lambda p: p.reliability, reverse=True)
        top = sorted_ps[:max(1, k // 2)]
        rest = sorted_ps[max(1, k // 2):]
        explore = random.sample(rest, k - len(top)) if rest else []
        return top + explore

    def snapshot(self) -> Dict[str, Dict]:
        return {name: {"reliability": self._persps[name].reliability,
                       "protected": self._persps[name].protected,
                       **self._stats[name].as_dict()}
                for name in self._persps}

    # ---- persistence ----
    def save(self, path: str):
        blob = {
            "perspectives": [
                {"name": p.name, "reliability": p.reliability, "protected": p.protected}
                for p in self._persps.values()
            ],
            "stats": {k: v.as_dict() for k,v in self._stats.items()}
        }
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)

    def load_reliabilities(self, path: str):
        try:
            with open(path) as f:
                blob = json.load(f)
            rel = {p["name"]: p["reliability"] for p in blob.get("perspectives", [])}
            for name, p in self._persps.items():
                if name in rel:
                    p.reliability = rel[name]
        except Exception:
            pass

    # ---- optional: spawn new perspectives (e.g., LLM) ----
    def spawn_if_needed(self, factory: Callable[[], Perspective] = None) -> Optional[str]:
        if len(self._persps) < self.min_pool and factory is not None:
            try:
                p = factory()
                p.reliability = 0.4   # must earn trust
                p.protected = False
                self.add(p)
                return p.name
            except Exception:
                return None
        return None


# ============================
#  COLLAPSE CACHE
# ============================

class CollapseCache:
    def __init__(self):
        # key -> list[(value, weight)]
        self.cache: Dict[str, List[Tuple[Tvalue, float]]] = defaultdict(list)

    def record(self, key: str, value: Tvalue, weight: float = 1.0):
        self.cache[key].append((value, weight))

    def get_weighted_score(self, key: str) -> float:
        entries = self.cache.get(key, [])
        if not entries:
            return 0.0
        total_weight, net_score = 0.0, 0.0
        for val, weight in entries:
            if val == Tvalue.TRUE:
                net_score += weight
            elif val == Tvalue.FALSE:
                net_score -= weight
            total_weight += weight
        return net_score / total_weight if total_weight > 0 else 0.0

    def get_stability(self, key: str) -> float:
        return abs(self.get_weighted_score(key))

    def clear(self, key: str):
        self.cache.pop(key, None)


# ============================
#  PROPOSITIONS
# ============================

@dataclass
class Proposition:
    statement: str
    contextual_value: Tvalue = Tvalue.BOTH
    perspective_values: Dict[str, Tvalue] = field(default_factory=dict)
    collapse_history: List[Tuple[datetime.datetime, Tvalue]] = field(default_factory=list)
    locked: bool = False  # stop updating when stability is high

    def _compute_dynamic_threshold(self, perspectives: List[Perspective]) -> float:
        count = len(perspectives)
        return min(max(0.5 / (1 + 0.1 * count), 0.1), 0.5)

    def evaluate(self, perspectives: List[Perspective], cache: CollapseCache,
                 confidence_threshold: float = 0.95, key: Optional[str] = None) -> Tvalue:
        if self.locked:
            return self.contextual_value

        key = key or f"PROP::{self.statement}"
        values: Set[Tvalue] = set()
        for p in perspectives:
            value = p.evaluate(self.statement)
            self.perspective_values[p.name] = value
            cache.record(key, value, weight=p.reliability)
            values.add(value)

        threshold = self._compute_dynamic_threshold(perspectives)
        score = cache.get_weighted_score(key)

        if Tvalue.TRUE in values and Tvalue.FALSE in values:
            self.contextual_value = Tvalue.BOTH
        elif score > threshold:
            self.contextual_value = Tvalue.TRUE
        elif score < -threshold:
            self.contextual_value = Tvalue.FALSE
        else:
            self.contextual_value = Tvalue.BOTH

        self.collapse_history.append((datetime.datetime.now(), self.contextual_value))

        # Reliability update based on non-BOTH consensus + notify manager if present
        if self.contextual_value != Tvalue.BOTH:
            for p in perspectives:
                pv = p.evaluate(self.statement)
                was_correct = (pv == self.contextual_value)
                p.update_reliability(was_correct)
                if hasattr(p, "_mgr_record"):
                    p._mgr_record(was_correct=was_correct, abstained=False)  # type: ignore[attr-defined]
        else:
            for p in perspectives:
                if hasattr(p, "_mgr_record"):
                    p._mgr_record(was_correct=False, abstained=True)  # type: ignore[attr-defined]

        # Early stopping
        if cache.get_stability(key) >= confidence_threshold:
            self.locked = True

        return self.contextual_value

    def compute_stability(self, cache: CollapseCache, key: Optional[str] = None) -> float:
        key = key or f"PROP::{self.statement}"
        return cache.get_stability(key)

    def try_solve(self, perspectives: List[Perspective]) -> Tuple[bool, Tvalue]:
        """Simulate solver-verifier loop for hard problems."""
        candidate_solution = random.choice([Tvalue.TRUE, Tvalue.FALSE])
        votes = [p.evaluate(self.statement) for p in perspectives]
        if votes.count(candidate_solution) > len(votes) / 2:
            return True, candidate_solution
        return False, Tvalue.BOTH


# ============================
#  PATCH & SELF-UPDATER
# ============================

@dataclass
class Patch:
    description: str
    diff: Dict[str, Any] = field(default_factory=dict)   # generic changes (e.g., params/configs)
    risk: float = 0.2                                    # 0..1 subjective risk score
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

@dataclass
class PatchDecision:
    patch_id: str
    decision: str                 # ACCEPT / REJECT / AMBIGUOUS
    stability: float
    blocked_by: List[str]
    votes: List[Tuple[str, Tvalue, float]]  # (perspective name, vote, weight)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    notes: str = ""

class SelfUpdater:
    def __init__(self,
                 perspectives: List[Perspective],
                 cache: Optional[CollapseCache] = None,
                 accept_threshold: float = 0.9,
                 lock_threshold: float = 0.97,
                 reject_threshold: float = 0.2,
                 manager: 'PerspectiveManager' = None):
        self.perspectives = perspectives
        self.manager = manager
        self.cache = cache or CollapseCache()
        self.accept_threshold = accept_threshold
        self.lock_threshold = lock_threshold
        self.reject_threshold = reject_threshold
        self.audit_log: List[PatchDecision] = []
        self.deployed: Dict[str, Patch] = {}

    def evaluate_patch(self, patch: Patch) -> PatchDecision:
        key = f"PATCH::{patch.id}"
        votes: List[Tuple[str, Tvalue, float]] = []
        blocked_by: List[str] = []

        ps = self.manager.select_subset(k=min(10, len(self.manager.list()))) if self.manager else self.perspectives

        for p in ps:
            v = p.evaluate_patch(patch)
            self.cache.record(key, v, weight=p.reliability)
            votes.append((p.name, v, p.reliability))
            if p.protected and v == Tvalue.FALSE:
                blocked_by.append(p.name)

        stability = self.cache.get_stability(key)

        if blocked_by:
            decision = "REJECT"
        elif stability >= self.accept_threshold:
            decision = "ACCEPT"
        elif stability <= self.reject_threshold:
            decision = "REJECT"
        else:
            decision = "AMBIGUOUS"

        dec = PatchDecision(
            patch_id=patch.id,
            decision=decision,
            stability=stability,
            blocked_by=blocked_by,
            votes=votes,
            notes=patch.description
        )
        self.audit_log.append(dec)
        return dec

    # Hooks for deployment; here we simulate canary + promote/rollback
    def deploy_canary(self, patch: Patch) -> bool:
        # Simulate by flipping a coin weighted by (1 - risk)
        return random.random() < (1.0 - patch.risk)

    def post_deploy_monitor(self, patch: Patch) -> float:
        # Simulated post-deploy stability reading
        key = f"POST::{patch.id}"
        sim_val = random.choice([Tvalue.TRUE, Tvalue.TRUE, Tvalue.BOTH, Tvalue.FALSE])
        self.cache.record(key, sim_val, weight=1.0)
        return self.cache.get_stability(key)

    def process_patch(self, patch: Patch) -> PatchDecision:
        dec = self.evaluate_patch(patch)
        if dec.decision != "ACCEPT":
            return dec

        # Canary
        if not self.deploy_canary(patch):
            dec.decision = "REJECT"
            dec.notes += " | Canary failed"
            return dec

        # Post-deploy monitoring
        post_stability = self.post_deploy_monitor(patch)
        if post_stability >= self.lock_threshold:
            self.deployed[patch.id] = patch
            dec.notes += f" | Promoted to prod (post stability={post_stability:.2f})"
        else:
            dec.decision = "REJECT"
            dec.notes += f" | Rolled back (post stability={post_stability:.2f})"
        return dec


# ============================
#  LLM CLIENT + INTEGRATIONS
# ============================

class LLMClient:
    """
    Thin wrapper so you can swap backends later.
    Supports OpenAI Chat Completions by default.
    Env vars:
      OPENAI_API_KEY (required to use)
      OPENAI_BASE_URL (optional)
      OPENAI_MODEL (optional; default 'gpt-4o')
    """
    def __init__(self, model: str = None, max_retries: int = 3, timeout_s: int = 30):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self.timeout_s = timeout_s

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI  # lazy import
            self._OpenAI = OpenAI
        except Exception as e:
            raise RuntimeError("openai python package not installed. `pip install openai`") from e

        self._client = self._OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_json(self, system: str, user: str) -> dict:
        """
        Ask model to return strict JSON. Retries on transient errors.
        """
        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise last_err


def make_llm_perspective(name: str = "LLM",
                         model: str = None,
                         protected: bool = False) -> Perspective:
    """
    LLM returns: {"vote": "TRUE"|"FALSE"|"BOTH", "rationale": "..."} for a proposition.
    Treated like any other perspective with reliability learning.
    """
    client = LLMClient(model=model)

    def eval_stmt(statement: str) -> Tvalue:
        sys = (
            "You are a cautious truth evaluator using a tri-valued logic: TRUE, FALSE, BOTH. "
            "Interpret the user's statement literally. "
            "Return strict JSON with keys: vote, rationale. No extra keys."
        )
        usr = (
            f"Statement: {statement}\n\n"
            "Rules:\n"
            "- If clearly true in general reality, vote TRUE.\n"
            "- If clearly false in general reality, vote FALSE.\n"
            "- If paradoxical/ambiguous/context-dependent, vote BOTH.\n"
            "Output JSON ONLY, like: {\"vote\":\"TRUE\", \"rationale\":\"...\"}"
        )
        try:
            res = client.chat_json(system=sys, user=usr)
            vote = str(res.get("vote", "BOTH")).upper()
            return Tvalue.TRUE if vote == "TRUE" else Tvalue.FALSE if vote == "FALSE" else Tvalue.BOTH
        except Exception:
            return Tvalue.BOTH

    return Perspective(name=name, evaluate_statement_fn=eval_stmt, evaluate_patch_fn=None, protected=protected)


class LLMPatchGenerator:
    """
    Ask the LLM to propose small, testable changes (patches) to improve solve rates/perf.
    Output is validated into a Patch. Use with SelfUpdater.process_patch().
    """
    def __init__(self, model: str = None):
        self.client = LLMClient(model=model)

    def propose(self, problem_summary: str, constraints: Dict[str, Any] = None) -> Patch:
        constraints = constraints or {}
        sys = (
            "You propose safe, incremental patches for a reasoning/solving engine. "
            "Patches MUST be small, testable, and revertible. "
            "Return strict JSON with keys: description, diff, risk."
        )
        usr = json.dumps({
            "problem_summary": problem_summary,
            "constraints": constraints,
            "examples": [
                {"description":"Tune restart policy", "diff":{"module":"sat_solver","param.restart":"Luby(10)"},"risk":0.2},
                {"description":"Increase LNS neighborhood", "diff":{"module":"cp","param.lns_neighborhood":256},"risk":0.25}
            ]
        })
        try:
            res = self.client.chat_json(system=sys, user=usr)
            desc = str(res.get("description", "LLM-proposed patch"))
            diff = res.get("diff", {}) or {}
            risk = float(res.get("risk", 0.3))
            risk = max(0.0, min(1.0, risk))
            return Patch(description=desc, diff=diff, risk=risk)
        except Exception:
            return Patch(description="Fallback: lower cost, safe tweak",
                         diff={"param.safe_mode": True, "cost_delta": -0.05}, risk=0.15)


# ============================
#  DEFAULT PERSPECTIVES (stubs)
# ============================

def make_reality_perspective(ground_truth: Dict[str, Tvalue]) -> Perspective:
    def eval_stmt(s: str) -> Tvalue:
        return ground_truth.get(s, Tvalue.BOTH)
    return Perspective("Reality", eval_stmt, protected=False)

def make_static_analysis_perspective(strict: bool = True) -> Perspective:
    def eval_stmt(_: str) -> Tvalue:
        return Tvalue.BOTH
    def eval_patch(patch: Patch) -> Tvalue:
        if patch.risk >= 0.8:
            return Tvalue.FALSE if strict else Tvalue.BOTH
        return Tvalue.TRUE if patch.risk <= 0.3 else Tvalue.BOTH
    return Perspective("StaticAnalysis", eval_stmt, eval_patch_fn=eval_patch, protected=True)

def make_unit_test_perspective(pass_rate: float = 0.9) -> Perspective:
    def eval_stmt(_: str) -> Tvalue:
        return Tvalue.BOTH
    def eval_patch(_: Patch) -> Tvalue:
        r = random.random()
        if r < pass_rate:
            return Tvalue.TRUE
        elif r < (pass_rate + 0.05):
            return Tvalue.FALSE
        else:
            return Tvalue.BOTH
    return Perspective("UnitTests", eval_stmt, eval_patch_fn=eval_patch, protected=True)

def make_performance_perspective(latency_budget_ms: int = 200) -> Perspective:
    def eval_stmt(_: str) -> Tvalue:
        return Tvalue.BOTH
    def eval_patch(patch: Patch) -> Tvalue:
        if patch.risk > 0.6 and random.random() < 0.6:
            return Tvalue.FALSE
        return random.choice([Tvalue.TRUE, Tvalue.BOTH])
    return Perspective("Performance", eval_stmt, eval_patch_fn=eval_patch, protected=False)

def make_safety_perspective() -> Perspective:
    def eval_stmt(_: str) -> Tvalue:
        return Tvalue.BOTH
    def eval_patch(patch: Patch) -> Tvalue:
        sensitive = patch.diff.get("touches", [])
        if any(x in {"security", "auth", "safety"} for x in sensitive):
            return Tvalue.FALSE
        return Tvalue.BOTH
    return Perspective("Safety", eval_stmt, eval_patch_fn=eval_patch, protected=True)

def make_cost_perspective(target_cost: float = 1.0) -> Perspective:
    def eval_stmt(_: str) -> Tvalue:
        return Tvalue.BOTH
    def eval_patch(patch: Patch) -> Tvalue:
        delta = patch.diff.get("cost_delta", 0.0)
        if delta <= 0:
            return Tvalue.TRUE
        if delta/target_cost > 0.25:
            return Tvalue.FALSE
        return Tvalue.BOTH
    return Perspective("Cost", eval_stmt, eval_patch_fn=eval_patch, protected=False)


# ============================
#  DEMO (run as a script)
# ============================

def demo():
    random.seed(7)
    ground_truth = {
        "Sky is blue": Tvalue.TRUE,
        "Grass is red": Tvalue.FALSE,
        "This statement is false": Tvalue.BOTH
    }

    cache = CollapseCache()
    prop_true = Proposition("Sky is blue")
    prop_false = Proposition("Grass is red")
    prop_liar = Proposition("This statement is false")

    perspectives = [
        make_reality_perspective(ground_truth),
        make_static_analysis_perspective(),
        make_unit_test_perspective(),
        make_performance_perspective(),
        make_safety_perspective(),
        make_cost_perspective(),
    ]

    # --- Manager wraps perspectives for dynamic lifecycle ---
    mgr = PerspectiveManager(perspectives)

    # Optionally spawn a fresh LLM perspective if pool thin (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        spawned = mgr.spawn_if_needed(lambda: make_llm_perspective(name=f"LLM_{random.randint(1000,9999)}"))
        if spawned:
            print(f"[Mgr] Spawned new perspective: {spawned}")

    # Evaluate propositions over multiple rounds (dynamic subset each tick)
    for _ in range(100):
        ps = mgr.select_subset(k=min(8, len(mgr.list())))
        prop_true.evaluate(ps, cache)
        prop_false.evaluate(ps, cache)
        prop_liar.evaluate(ps, cache)
        mgr.periodic_decay()
        mgr.auto_prune()

    print("[Propositions]")
    print("Sky is blue ->", prop_true.contextual_value.name, "stability:", f"{prop_true.compute_stability(cache):.2f}")
    print("Grass is red ->", prop_false.contextual_value.name, "stability:", f"{prop_false.compute_stability(cache):.2f}")
    print("Liar ->", prop_liar.contextual_value.name, "stability:", f"{prop_liar.compute_stability(cache):.2f}")

    # Self-updater with manager
    updater = SelfUpdater(mgr.list(), cache=cache, manager=mgr)
    patches = [
        Patch("Tune search heuristic for planner", diff={"module": "planner", "cost_delta": -0.1}, risk=0.15),
        Patch("Refactor core reasoning loop", diff={"module": "core", "touches": ["safety"]}, risk=0.65),
        Patch("Swap model to larger LLM", diff={"module": "nlp", "cost_delta": 0.4}, risk=0.55),
        Patch("Improve caching layer", diff={"module": "infra", "cost_delta": -0.2}, risk=0.25),
        Patch("Experimental self-modifying kernel", diff={"module": "kernel"}, risk=0.9),
    ]

    print("\n[Self-Updater Decisions]")
    for p in patches:
        dec = updater.process_patch(p)
        print(f"Patch {p.description!r} -> {dec.decision} | stability={dec.stability:.2f} | blocked={dec.blocked_by} | notes={dec.notes}")

    # Optional: Add a primary LLM perspective and get an LLM-proposed patch
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm_p = make_llm_perspective(name="LLM_Main", protected=False)
            mgr.add(llm_p)
            for _ in range(10):
                ps = mgr.select_subset(k=min(8, len(mgr.list())))
                prop_true.evaluate(ps, cache)
                prop_false.evaluate(ps, cache)
                prop_liar.evaluate(ps, cache)

            print("\n[After Adding LLM_Main]")
            print("Sky is blue ->", prop_true.contextual_value.name, "stability:", f"{prop_true.compute_stability(cache):.2f}")
            print("Grass is red ->", prop_false.contextual_value.name, "stability:", f"{prop_false.compute_stability(cache):.2f}")
            print("Liar ->", prop_liar.contextual_value.name, "stability:", f"{prop_liar.compute_stability(cache):.2f}")

            gen = LLMPatchGenerator()
            patch = gen.propose(
                problem_summary="Improve SAT+MILP portfolio solve-rate under 30s timeout.",
                constraints={"avoid_modules":["security","auth","safety"], "max_cost_delta": 0.2}
            )
            dec = updater.process_patch(patch)
            print(f"\n[LLM Patch] {patch.description} -> {dec.decision} | stability={dec.stability:.2f} | blocked={dec.blocked_by} | notes={dec.notes}")
        except Exception as e:
            print("[LLM] Integration skipped due to error:", e)


if __name__ == "__main__":
    demo()
    
## License
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0).
You may not use the material for commercial purposes without my permission.
See the LICENSE file for full legal text.
