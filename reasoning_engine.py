from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
import time, json, hashlib, random, statistics, copy, uuid
from collections import deque, defaultdict

# ---------- Paraconsistent primitives (Priest's LP; NO 'NEITHER') ----------
TruthPair = Tuple[float, float]  # (t, f) in [0,1]
LP_TRUE:  TruthPair = (1.0, 0.0)
LP_FALSE: TruthPair = (0.0, 1.0)
LP_BOTH:  TruthPair = (1.0, 1.0)

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)

def lp_and(a: TruthPair, b: TruthPair) -> TruthPair:
    # meet for truth, join for falsity
    return (min(a[0], b[0]), max(a[1], b[1]))

def lp_or(a: TruthPair, b: TruthPair) -> TruthPair:
    # join for truth, meet for falsity
    return (max(a[0], b[0]), min(a[1], b[1]))

def lp_not(a: TruthPair) -> TruthPair:
    return (a[1], a[0])

def lp_imp(a: TruthPair, b: TruthPair) -> TruthPair:
    # a → b ≡ ¬a ∨ b
    return lp_or(lp_not(a), b)

def pair_to_scalar(tp: TruthPair) -> float:
    # summarize a pair to a pragmatic decision scalar in [0,1]
    t, f = tp
    # prefer pure truth > both > pure false
    return t * (1 - f) + 0.5 * (t * f)

def scalar_to_pair(x: float, both_band: Tuple[float, float]) -> TruthPair:
    """
    Map a scalar belief into LP truth-pair WITHOUT EVER PRODUCING (0,0).
    Policy:
      - if x >= 0.85 -> TRUE
      - if x <= 0.15 -> FALSE
      - if x in BOTH band -> BOTH
      - otherwise interpolate toward BOTH but clamp away from (0,0)
    """
    lo, hi = both_band
    x = clamp01(x)
    if x >= 0.85:
        return LP_TRUE
    if x <= 0.15:
        return LP_FALSE
    if lo <= x <= hi:
        return LP_BOTH
    # interpolate toward BOTH; avoid (0,0) by epsilon floor
    d = abs(x - 0.5) * 2.0              # 0 at center, 1 at extremes
    eps = 0.05                          # never allow (0,0)
    v = max(eps, 1.0 - d)               # shrinks toward BOTH, not past eps
    return (v, v)
    # ---------- Data structures ----------
@dataclass
class Perspective:
    value_scalar: float
    rationale: str
    confidence: float
    perspective_type: str
    key_factors: List[str]
    source: str
    provenance: Dict[str, Any]
    value_pair: TruthPair = field(init=False)
    def __post_init__(self):
        self.value_scalar = clamp01(self.value_scalar)
        self.confidence   = clamp01(self.confidence)
        # temporary; engine remaps with its active BOTH band
        self.value_pair   = scalar_to_pair(self.value_scalar, (0.45, 0.55))

@dataclass
class Consensus:
    raw_scalar: float
    avg_conf: float
    total_weight: float
    reliability_weights: Dict[str, float]

@dataclass
class EvaluationResult:
    value_pair: TruthPair
    value_scalar: float
    decision: str            # "TRUE" | "FALSE" | "BOTH"
    reasoning: str
    stability: float
    consensus: Consensus
    contradictions: Dict[str, Any]
    perspectives: List[Perspective]

# ---------- Reliability / calibration ----------
class ReliabilityBook:
    """Per-source rolling Brier and bounded weight in [0.5, 2.0]."""
    def __init__(self, window: int = 200):
        self.window = window
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    def update(self, source: str, predicted: float, realized: float):
        err = (predicted - realized) ** 2
        self.history[source].append(err)
    def weight(self, source: str) -> float:
        hist = self.history[source]
        if not hist:
            return 1.0
        score = sum(hist) / len(hist)  # lower is better
        return clamp01(1.5 - score) + 0.5
        # ---------- Safe tunables (for self-modification with guardrails) ----------
@dataclass
class Tunables:
    both_band_lo: float = 0.40
    both_band_hi: float = 0.60
    strong_contradiction_gap: float = 0.40
    truth_hi: float = 0.85
    truth_lo: float = 0.15
    min_high_conf: float = 0.70
    def as_dict(self): return asdict(self)

class SafeUpdater:
    def __init__(self, tunables: Tunables):
        self.tunables = tunables
        self.versions: List[Dict[str, Any]] = []
    def propose(self, patch: Dict[str, float], tests: List[Callable[[], None]]) -> bool:
        # copy & validate
        new_state = copy.deepcopy(self.tunables.as_dict())
        for k, v in patch.items():
            if k not in new_state:
                raise KeyError(f"Unknown tunable: {k}")
            if k in ("both_band_lo","both_band_hi","truth_hi","truth_lo","min_high_conf") and not (0<=v<=1):
                raise ValueError(f"{k} must be within [0,1]")
            if k=="strong_contradiction_gap" and not (0<=v<=1):
                raise ValueError("strong_contradiction_gap must be within [0,1]")
            new_state[k] = float(v)
        if new_state["both_band_lo"] > new_state["both_band_hi"]:
            raise ValueError("both_band_lo must be <= both_band_hi")

        # dry-run tests with rollback
        bak = copy.deepcopy(self.tunables)
        for k, v in new_state.items(): setattr(self.tunables, k, v)
        try:
            for t in tests: t()
        except Exception:
            # rollback on any failure
            for k, v in bak.as_dict().items(): setattr(self.tunables, k, v)
            return False
        # commit meta
        self.versions.append({"id": str(uuid.uuid4())[:8], "time": time.time(), "state": new_state})
        return True
        # ---------- Engine (perspectival collapse; default BOTH) ----------
class PPCPlusEngine:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tun = Tunables()
        self.safe = SafeUpdater(self.tun)
        self.reliability = ReliabilityBook()
        self.history: Dict[str, List[EvaluationResult]] = defaultdict(list)
        self.global_context: Dict[str, Any] = {}
        self.adapters: Dict[str, Callable[[str, Dict[str, Any]], Perspective]] = {}

    def set_global_context(self, ctx: Dict[str, Any]):
        self.global_context = dict(ctx or {})

    def register_adapter(self, name: str,
                         adapter: Callable[[str, Dict[str, Any]], Dict[str, Any]],
                         meta: Dict[str, Any]):
        def wrapper(statement: str, context: Dict[str, Any]) -> Perspective:
            raw = adapter(statement, context)
            prov = {
                "name": name,
                "meta": meta,
                "ctx_hash": hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:12]
            }
            p = Perspective(
                value_scalar=float(raw["value"]),
                rationale=str(raw.get("rationale", "")),
                confidence=float(raw.get("confidence", 0.5)),
                perspective_type=str(raw.get("perspective_type", "general")),
                key_factors=list(raw.get("key_factors", [])),
                source=name,
                provenance=prov,
            )
            # remap using current BOTH band
            p.value_pair = scalar_to_pair(p.value_scalar, (self.tun.both_band_lo, self.tun.both_band_hi))
            return p
        self.adapters[name] = wrapper

    def evaluate(self, statement: str, context: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        ctx = {**self.global_context, **(context or {})}

        # Collect perspectives (if none, we still return BOTH by design)
        perspectives: List[Perspective] = []
        for name, fn in self.adapters.items():
            try:
                perspectives.append(fn(statement, ctx))
            except Exception as e:
                # graceful degradation still counts as a (low-weight) BOTH-lean
                perspectives.append(Perspective(
                    value_scalar=0.5, rationale=f"Adapter error: {e}", confidence=0.1,
                    perspective_type="error_fallback", key_factors=["adapter_error"],
                    source=name, provenance={"name": name, "error": True, "meta": {}}
                ))

        if not perspectives:
            # Epistemic default: BOTH
            empty_cons = Consensus(0.5, 0.0, 0.0, {})
            result = EvaluationResult(
                value_pair=LP_BOTH,
                value_scalar=0.5,
                decision="BOTH",
                reasoning="No perspectives available; defaulting to BOTH (perspectival prior).",
                stability=1.0,
                consensus=empty_cons,
                contradictions={"count": 0, "avg_gap": 0.0, "max_gap": 0.0, "strong_pairs": []},
                perspectives=[]
            )
            self.history[statement].append(result)
            return result

        # Reliability-weighted scalar consensus
        weights, wsum, scalar_sum = {}, 0.0, 0.0
        for p in perspectives:
            w = p.confidence * self.reliability.weight(p.source)
            weights[p.source] = w
            wsum += w
            scalar_sum += p.value_scalar * w
        raw = scalar_sum / wsum if wsum > 0 else 0.5
        avg_conf = sum(p.confidence for p in perspectives) / len(perspectives)
        consensus = Consensus(raw, avg_conf, wsum, weights)

        # Contradiction analysis
        diffs, strong = [], []
        for i in range(len(perspectives)):
            for j in range(i + 1, len(perspectives)):
                d = abs(perspectives[i].value_scalar - perspectives[j].value_scalar)
                diffs.append(d)
                if d > self.tun.strong_contradiction_gap:
                    strong.append((perspectives[i].source, perspectives[j].source, d))
        contradictions = {
            "count": len(strong),
            "avg_gap": statistics.mean(diffs) if diffs else 0.0,
            "max_gap": max(diffs) if diffs else 0.0,
            "strong_pairs": strong
        }

        # BOTH band adapts to local contradiction rate (per statement)
        recent = self.history[statement][-10:] if self.history[statement] else []
        recent_contra = [r.contradictions.get("count", 0) for r in recent]
        contra_rate = (sum(recent_contra) / (len(recent_contra) or 1)) / max(1, len(perspectives) - 1)
        width = max(0.2, min(0.6, 0.2 + 0.4 * contra_rate))  # keep within [0.2, 0.6]
        center = 0.5
        self.tun.both_band_lo = center - width / 2
        self.tun.both_band_hi = center + width / 2

        # Make the perspectival decision (collapse only with sufficient push)
        pair = scalar_to_pair(raw, (self.tun.both_band_lo, self.tun.both_band_hi))
        decision = self._name_from_scalar(raw, avg_conf)

        # Stability from recent variance
        scalars = [r.value_scalar for r in recent] + [raw]
        if len(scalars) <= 1:
            stability = 1.0
        else:
            var = statistics.pvariance(scalars)
            stability = max(0.0, 1.0 - 4.0 * var)

        result = EvaluationResult(
            value_pair=pair,
            value_scalar=raw,
            decision=decision,
            reasoning=self._reasoning_text(raw, avg_conf, contradictions),
            stability=stability,
            consensus=consensus,
            contradictions=contradictions,
            perspectives=perspectives
        )
        self.history[statement].append(result)

        # Update reliability after the fact (consensus-as-proxy target)
        for p in perspectives:
            self.reliability.update(p.source, p.value_scalar, raw)

        return result

    def _name_from_scalar(self, raw: float, conf: float) -> str:
        # Only three outcomes: TRUE / FALSE / BOTH (default)
        if self.tun.both_band_lo <= raw <= self.tun.both_band_hi:
            return "BOTH"
        if raw >= self.tun.truth_hi and conf >= self.tun.min_high_conf:
            return "TRUE"
        if raw <= self.tun.truth_lo and conf >= self.tun.min_high_conf:
            return "FALSE"
        # borderline cases remain BOTH (don’t force classical bivalence)
        return "BOTH"

    def _reasoning_text(self, raw: float, conf: float, contra: Dict[str, Any]) -> str:
        return (f"Consensus={raw:.3f}, avg_conf={conf:.2f}, "
                f"contradictions={contra['count']}, both_band=[{self.tun.both_band_lo:.2f},{self.tun.both_band_hi:.2f}]")

# ---------- Example adapters (swap these with real LLM calls) ----------
def mock_adapter_factory(name: str, bias: float = 0.0):
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    def adapter(statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        s = statement.lower()
        if any(k in s for k in ["this statement", "liar", "paradox", "unprovable"]):
            base, conf, ptype, rationale = 0.5, 0.85, "paradox_aware", "Self-referential/paradoxical; preserve BOTH."
        elif any(k in s for k in ["2 + 2", "sky is blue", "earth orbits", "gravity"]):
            base, conf, ptype, rationale = (0.9, 0.9, "empirical", "Empirically grounded.") if "not" not in s else (0.1, 0.9, "empirical", "Contradicts empirical regularities.")
        else:
            base, conf, ptype, rationale = (0.6 if "will" in s else 0.5, 0.6, "speculative", "Plausible yet uncertain.")
        jitter = (rng.random() - 0.5) * 0.1
        val = clamp01(base + bias + jitter)
        return {"value": val, "rationale": rationale, "confidence": conf,
                "perspective_type": ptype, "key_factors": ["content","heuristics"]}
    return adapter
    from ppc_plus import PPCPlusEngine, mock_adapter_factory

def main():
    eng = PPCPlusEngine(seed=123)
    eng.set_global_context({
        "logic_system": "Perspectivistic Paraconsistent Contextualism",
        "goal": "preserve contradiction; collapse only with sufficient perspectival/contextual push"
    })

    # Register “personas” (replace with real LLM adapters)
    eng.register_adapter("LogicalAnalyst",     mock_adapter_factory("LogicalAnalyst", +0.05), {"style":"logical"})
    eng.register_adapter("SkepticalCritic",    mock_adapter_factory("SkepticalCritic", -0.05), {"style":"skeptical"})
    eng.register_adapter("ParadoxSpecialist",  mock_adapter_factory("ParadoxSpecialist", 0.00), {"style":"paradox"})
    eng.register_adapter("PragmaticEvaluator", mock_adapter_factory("PragmaticEvaluator", +0.02), {"style":"pragmatic"})
    eng.register_adapter("PhilosophicalThinker", mock_adapter_factory("PhilosophicalThinker", -0.02), {"style":"philosophical"})

    tests = [
        "This statement is false",
        "This statement is false and the sky is blue",
        "2 + 2 = 4",
        "If the sky is blue then grass is green",
        "Artificial intelligence will surpass human intelligence",
        "All statements are either true or false",
        "Some truths cannot be proven"
    ]

    for s in tests:
        res = eng.evaluate(s)
        print(f"{s!r} -> {res.decision} (scalar={res.value_scalar:.3f}, stability={res.stability:.2f})")
        print("  ", res.reasoning)

    # Safe self-modification example (guarded)
    ok = eng.safe.propose(
        {"strong_contradiction_gap": 0.35, "both_band_lo": 0.42, "both_band_hi": 0.58},
        tests=[lambda: None]  # plug in your invariants / smoke tests here
    )
    print("Patch applied:", ok)
    print("Done.")

if __name__ == "__main__":
    main()
