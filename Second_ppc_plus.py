# ppc_synth_engine.py
# Synthesized Perspectival Paraconsistent Contextual Engine
# - Scalar truth (False=0.0, Both=0.5, True=1.0); no "unknown".
# - Cache ONLY perspectives (per-claim), with asymmetric decay → snap-to-0.5 → eviction.
# - Paraconsistent, centered fusion with stability (K) + dissent cap before T/F.
# - LP projection (t,f) on demand; formal lp_* ops included.
# - ReliabilityBook to reweight sources (rolling Brier-ish).
# - Tunables + SafeUpdater for guarded self-tuning.
# - Adaptive BOTH-band width using rolling contradiction rate + fused-value variance.
# - Mock LLM/adapters; demo.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Iterable, Optional, Any, DefaultDict, Callable
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
import math, uuid, re, random, statistics, hashlib, json

# =============================================================================
# 0) Time helpers
# =============================================================================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def days_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 86400.0

# =============================================================================
# 1) Scalar truth, centered fuzzy ops (Both=0.5 substrate)
# =============================================================================
FALSE, BOTH, TRUE = 0.0, 0.5, 1.0

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)

def delta(v: float) -> float:
    return v - 0.5

def neg(v: float) -> float:
    return 1.0 - v

def conj(v: float, u: float) -> float:
    return 0.5 + min(delta(v), delta(u))

def disj(v: float, u: float) -> float:
    return 0.5 + max(delta(v), delta(u))

def impl(v: float, u: float) -> float:
    return max(neg(v), u)

# =============================================================================
# 2) LP truth-pair projection + LP connectives
# =============================================================================
TruthPair = Tuple[float, float]  # (t, f) ∈ [0,1]
LP_TRUE: TruthPair  = (1.0, 0.0)
LP_FALSE: TruthPair = (0.0, 1.0)
LP_BOTH: TruthPair  = (1.0, 1.0)

def lp_and(a: TruthPair, b: TruthPair) -> TruthPair:
    return (min(a[0], b[0]), max(a[1], b[1]))

def lp_or(a: TruthPair, b: TruthPair) -> TruthPair:
    return (max(a[0], b[0]), min(a[1], b[1]))

def lp_not(a: TruthPair) -> TruthPair:
    return (a[1], a[0])

def lp_imp(a: TruthPair, b: TruthPair) -> TruthPair:
    return lp_or(lp_not(a), b)

def pair_to_scalar(tp: TruthPair) -> float:
    t, f = tp
    return t * (1.0 - f) + 0.5 * (t * f)  # pure T > both > pure F

def scalar_to_pair(x: float, both_band: Tuple[float, float]) -> TruthPair:
    lo, hi = both_band
    x = clamp01(x)
    if x >= 0.85:
        return LP_TRUE
    if x <= 0.15:
        return LP_FALSE
    if lo <= x <= hi:
        return LP_BOTH
    # interpolate toward BOTH away from extremes, avoid (0,0)
    d = abs(x - 0.5) * 2.0  # 0 center, 1 extremes
    eps = 0.05
    v = max(eps, 1.0 - d)   # 1 at center, → eps at extremes
    return (v, v)

# =============================================================================
# 3) IDs & core data model
# =============================================================================
def new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex[:10]}"

@dataclass(frozen=True)
class Claim:
    claim_id: str
    canonical_text: str
    tags: Tuple[str, ...] = ()

@dataclass(frozen=True)
class PerspectiveHeader:
    perspective_id: str
    kind: str                 # {"cognitive","affective","preference","social","meta"}
    owner: str                # {"system","session","community","user:<uid>"}
    traits: Tuple[str, ...] = ()
    authority_hint: float = 1.0  # base multiplier for fusion weight

@dataclass
class PerspectiveValuePoint:
    value: float                  # THIS IS the epistemic truth/confidence (0..1)
    event_time: datetime
    context_tags: Tuple[str, ...] = ()
    justification: str = ""

@dataclass
class ClaimSeries:
    claim_id: str
    points: List[PerspectiveValuePoint] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    status: str = "active"        # {"active","inactive","evicted","new"}

    def append(self, p: PerspectiveValuePoint) -> None:
        self.points.append(p)
        self.last_updated = p.event_time
        self.status = "active"

    def latest(self) -> Optional[PerspectiveValuePoint]:
        return self.points[-1] if self.points else None

@dataclass
class PerspectiveRecord:
    header: PerspectiveHeader
    claims: Dict[str, ClaimSeries] = field(default_factory=dict)
    last_active: Optional[datetime] = None

    def touch(self, t: Optional[datetime] = None) -> None:
        self.last_active = t or utcnow()

# =============================================================================
# 4) Decay policy (asymmetric), snap-to-0.5, eviction
# =============================================================================
@dataclass
class DecayPolicy:
    half_life_true_days: float = 7.0
    half_life_false_days: float = 1.0
    snap_to_both_band: Tuple[float, float] = (0.4, 0.6)
    evict_after_days_at_exact_both: float = 14.0
    half_life_affective_true_days: Optional[float] = 3.0
    half_life_preference_true_days: Optional[float] = 21.0

    def _half_life_for(self, header: PerspectiveHeader, last_v: float) -> float:
        if header.kind == "affective" and last_v > 0.5 and self.half_life_affective_true_days:
            return self.half_life_affective_true_days
        if header.kind == "preference" and last_v > 0.5 and self.half_life_preference_true_days:
            return self.half_life_preference_true_days
        if last_v > 0.5:
            return self.half_life_true_days
        elif last_v < 0.5:
            return self.half_life_false_days
        return 0.0

    def apply_decay(self, header: PerspectiveHeader, v0: float, t0: datetime, now: datetime) -> float:
        if v0 == 0.5:
            return 0.5
        hl = self._half_life_for(header, v0)
        if hl <= 0:  # already at center or no half-life specified
            return 0.5
        dt_days = days_between(t0, now)
        vt = 0.5 + (v0 - 0.5) * math.pow(0.5, dt_days / hl)
        return clamp01(vt)

    def snap_or_status(self, v: float, last_update: datetime, now: datetime) -> Tuple[float, str]:
        lo, hi = self.snap_to_both_band
        status = "active"
        vv = v
        if lo <= v <= hi:
            vv = 0.5
            status = "inactive"
            if days_between(last_update, now) >= self.evict_after_days_at_exact_both:
                status = "evicted"
        return vv, status

# =============================================================================
# 5) In-memory PerspectiveCache (only perspectives, no prompts)
# =============================================================================
class PerspectiveCache:
    def __init__(self, policy: Optional[DecayPolicy] = None):
        self._perspectives: Dict[str, PerspectiveRecord] = {}
        self.policy = policy or DecayPolicy()

    def upsert_perspective(self, header: PerspectiveHeader) -> PerspectiveRecord:
        rec = self._perspectives.get(header.perspective_id)
        if rec is None:
            rec = PerspectiveRecord(header=header)
            self._perspectives[header.perspective_id] = rec
        else:
            rec.header = header
        rec.touch()
        return rec

    def get_perspective(self, perspective_id: str) -> Optional[PerspectiveRecord]:
        return self._perspectives.get(perspective_id)

    def list_perspectives(self) -> List[PerspectiveRecord]:
        return list(self._perspectives.values())

    def write_point(
        self,
        header: PerspectiveHeader,
        claim: Claim,
        value: float,
        event_time: Optional[datetime] = None,
        context_tags: Iterable[str] = (),
        justification: str = "",
    ) -> None:
        rec = self.upsert_perspective(header)
        series = rec.claims.get(claim.claim_id)
        if series is None:
            series = ClaimSeries(claim_id=claim.claim_id, status="new")
            rec.claims[claim.claim_id] = series
        point = PerspectiveValuePoint(
            value=clamp01(value),
            event_time=event_time or utcnow(),
            context_tags=tuple(sorted(set(context_tags))),
            justification=justification,
        )
        series.append(point)
        rec.touch(point.event_time)

    def read_current_value(
        self, header: PerspectiveHeader, claim_id: str, now: Optional[datetime] = None
    ) -> Tuple[float, str, Optional[datetime]]:
        now = now or utcnow()
        rec = self._perspectives.get(header.perspective_id)
        if rec is None:
            return BOTH, "new", None
        series = rec.claims.get(claim_id)
        if series is None or not series.points:
            return BOTH, "new", None
        last = series.latest()
        v = self.policy.apply_decay(rec.header, last.value, last.event_time, now)
        v, status = self.policy.snap_or_status(v, last.event_time, now)
        return v, status, series.last_updated

    def maintenance(self, now: Optional[datetime] = None) -> None:
        now = now or utcnow()
        to_evict: List[Tuple[str, str]] = []
        for pid, rec in list(self._perspectives.items()):
            rec.touch(now)
            for cid, series in list(rec.claims.items()):
                if not series.points:
                    continue
                last = series.latest()
                v = self.policy.apply_decay(rec.header, last.value, last.event_time, now)
                v2, status = self.policy.snap_or_status(v, last.event_time, now)
                if status == "evicted":
                    to_evict.append((pid, cid))
                else:
                    if v2 == 0.5 and (series.status != "inactive" or last.value != 0.5):
                        series.points.append(PerspectiveValuePoint(value=0.5, event_time=now, justification="snap-to-both"))
                        series.last_updated = now
                    series.status = status
        for pid, cid in to_evict:
            rec = self._perspectives.get(pid)
            if rec and cid in rec.claims:
                del rec.claims[cid]

# =============================================================================
# 6) Fusion & dissent profiling
# =============================================================================
@dataclass
class DissentProfile:
    mass_above: float
    mass_below: float
    n_perspectives: int
    notes: str = ""

def fuse_centered(values_and_weights: Iterable[Tuple[float, float]]) -> Tuple[float, DissentProfile]:
    eps = 1e-9
    num = den = 0.0
    mass_above = mass_below = 0.0
    n = 0
    for v, w in values_and_weights:
        dv = delta(v)
        num += w * dv
        den += abs(w)
        if v > 0.5:   mass_above += w
        elif v < 0.5: mass_below += w
        n += 1
    V = 0.5 + (num / (den + eps)) if n > 0 else 0.5
    profile = DissentProfile(
        mass_above=mass_above, mass_below=mass_below, n_perspectives=n,
        notes="Opposition exists" if (mass_above > 0 and mass_below > 0) else "Unanimous or empty"
    )
    return clamp01(V), profile

# =============================================================================
# 7) Tunables + SafeUpdater + ReliabilityBook
# =============================================================================
@dataclass
class Tunables:
    both_band_lo: float = 0.40
    both_band_hi: float = 0.60
    strong_contradiction_gap: float = 0.40
    truth_hi: float = 0.85
    truth_lo: float = 0.15
    min_high_conf: float = 0.70
    stability_k: int = 2
    dissent_cap: float = 0.20

    # decay policy mirrors (so we can self-tune if desired)
    half_life_true_days: float = 7.0
    half_life_false_days: float = 1.0
    evict_after_days_at_exact_both: float = 14.0

    def as_dict(self): return asdict(self)

class SafeUpdater:
    def __init__(self, tunables: Tunables):
        self.tunables = tunables
        self.versions: List[Dict[str, Any]] = []

    def propose(self, patch: Dict[str, float], tests: List[Callable[[], None]]) -> bool:
        new_state = self.tunables.as_dict()
        for k, v in patch.items():
            if k not in new_state:
                raise KeyError(f"Unknown tunable: {k}")
            if isinstance(new_state[k], float):
                if not (0.0 <= float(v) <= 1.0) and "half_life" not in k and "evict_after" not in k:
                    raise ValueError(f"{k} must be within [0,1]")
            new_state[k] = type(new_state[k])(v)
        if new_state["both_band_lo"] > new_state["both_band_hi"]:
            raise ValueError("both_band_lo must be <= both_band_hi")

        # dry run
        bak = self.tunables.as_dict()
        for k, v in new_state.items(): setattr(self.tunables, k, v)
        try:
            for t in tests: t()
        except Exception:
            # rollback
            for k, v in bak.items(): setattr(self.tunables, k, v)
            return False
        self.versions.append({"id": str(uuid.uuid4())[:8], "time": utcnow().isoformat(), "state": new_state})
        return True

class ReliabilityBook:
    """Per-source rolling Brier-ish error; returns weight in [0.5, 2.0]."""
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
        # Map error → [0.5, 2.0] conservatively
        return clamp01(1.5 - score) + 0.5

# =============================================================================
# 8) Verdict policy, humility, stability
# =============================================================================
def humility_from_scalar(V: float) -> float:
    return clamp01(1.0 - 2.0 * abs(delta(V)))

@dataclass
class StabilityTrack:
    last_verdict: Optional[str] = None
    streak: int = 0
    def update(self, verdict: str) -> None:
        if verdict == self.last_verdict and verdict in ("TRUE","FALSE"):
            self.streak += 1
        else:
            self.streak = 1 if verdict in ("TRUE","FALSE") else 0
        self.last_verdict = verdict
    def is_stable(self, k: int) -> bool:
        return self.streak >= k if self.last_verdict in ("TRUE","FALSE") else False

# =============================================================================
# 9) Epistemic Engine (synth)
# =============================================================================
@dataclass
class EngineConfig:
    tun: Tunables = field(default_factory=Tunables)
    run_maintenance_each_turn: bool = True
    adaptive_both_band: bool = True  # widen band with contradiction rate + variance

class EpistemicEngine:
    def __init__(self, cache: Optional[PerspectiveCache] = None, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        # Keep decay policy in cache aligned with tunables:
        decay = DecayPolicy(
            half_life_true_days=self.config.tun.half_life_true_days,
            half_life_false_days=self.config.tun.half_life_false_days,
            evict_after_days_at_exact_both=self.config.tun.evict_after_days_at_exact_both,
        )
        self.cache = cache or PerspectiveCache(decay)
        self.stability: Dict[str, StabilityTrack] = defaultdict(StabilityTrack)
        self.reliability = ReliabilityBook()
        self.safe = SafeUpdater(self.config.tun)

        # Rolling history for adaptive BOTH-band
        self._recent_by_claim: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))  # fused_V
        self._contra_hist: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))      # contradiction rates

    # ---------- external API ----------
    def process_turn(
        self,
        claims_msg: Dict[str, Any],
        perspectives_msg: Dict[str, Any],
        now: Optional[datetime] = None
    ) -> Dict[str, Any]:
        now = now or utcnow()
        user_id = claims_msg.get("user_id")
        session_id = claims_msg.get("session_id")

        claims = self._parse_claims(claims_msg)
        items_by_claim: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

        # 1) Update cache per perspective with conservative learning
        for item in perspectives_msg.get("items", []):
            claim_obj, header, v_reading, w_ctx, justification, ctx_tags = self._parse_perspective_item(item)
            v_current, _, _ = self.cache.read_current_value(header, claim_obj.claim_id, now=now)
            alpha = self._compute_alpha(w_ctx, header)
            v_new = clamp01(0.5 + alpha*(v_reading-0.5) + (1.0-alpha)*(v_current-0.5))
            self.cache.write_point(header, claim_obj, v_new, event_time=now, context_tags=ctx_tags, justification=justification or "update")
            w_eff = self._effective_weight(w_ctx, header)
            items_by_claim[claim_obj.claim_id].append({"header": header, "claim": claim_obj, "value": v_new, "weight": w_eff})

        if self.config.run_maintenance_each_turn:
            self.cache.maintenance(now=now)

        # 2) Fuse per-claim, verdicts, humility, LP projection, contradiction audit, reliability updates
        verdicts_payload: List[Dict[str, Any]] = []
        for c in claims:
            fused_V, profile, audit = self._fuse_and_audit(c, items_by_claim[c.claim_id])

            # --- Update rolling histories for adaptive band ---
            denom_pairs = max(1, len(items_by_claim[c.claim_id]) - 1)
            cur_rate = (audit["count"] / denom_pairs) if denom_pairs > 0 else 0.0
            self._contra_hist[c.claim_id].append(cur_rate)
            self._recent_by_claim[c.claim_id].append(fused_V)

            # adaptive BOTH band (uses rolling stats + current rate)
            both_band = (self.config.tun.both_band_lo, self.config.tun.both_band_hi)
            if self.config.adaptive_both_band:
                both_band = self._adapt_both_band(c.claim_id, fused_V, cur_rate)

            raw_verdict = self._verdict_from_scalar(fused_V, both_band)
            st = self.stability[c.claim_id]
            st.update(raw_verdict)
            verdict_final, stable, explanations = self._apply_stability_and_dissent(raw_verdict, st, profile, both_band)

            # LP projection of fused scalar (for formal consumers)
            lp_pair = scalar_to_pair(fused_V, both_band)

            # reliability updates (consensus-as-proxy target)
            for it in items_by_claim[c.claim_id]:
                src = it["header"].perspective_id
                self.reliability.update(src, it["value"], fused_V)

            verdicts_payload.append({
                "claim_id": c.claim_id,
                "canonical_text": c.canonical_text,
                "fused_value": fused_V,
                "lp_pair": {"t": lp_pair[0], "f": lp_pair[1]},
                "both_band": {"lo": both_band[0], "hi": both_band[1]},
                "verdict": verdict_final,
                "humility": humility_from_scalar(fused_V),
                "stable": stable,
                "dissent_profile": {
                    "mass_above": profile.mass_above,
                    "mass_below": profile.mass_below,
                    "n_perspectives": profile.n_perspectives,
                    "notes": profile.notes
                },
                "contradictions": audit,
                "explanations": explanations
            })

        style_directives = self._style_from_context(verdicts_payload)
        return {
            "version": "v1",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": now.isoformat(),
            "verdicts": verdicts_payload,
            "style_directives": style_directives
        }

    # ---------- internals ----------
    def _parse_claims(self, claims_msg: Dict[str, Any]) -> List[Claim]:
        out: List[Claim] = []
        for c in claims_msg.get("claims", []):
            claim_id = c.get("claim_id") or f"c:auto:{abs(hash(c.get('canonical_text',''))) % (10**10)}"
            text = c.get("canonical_text") or ""
            tags = tuple(c.get("tags", []) or [])
            out.append(Claim(claim_id=claim_id, canonical_text=text, tags=tags))
        return out

    def _parse_perspective_item(self, item: Dict[str, Any]) -> Tuple[Claim, PerspectiveHeader, float, float, str, Tuple[str, ...]]:
        c = item.get("claim", {})
        p = item.get("perspective", {})
        claim_obj = Claim(
            claim_id = c.get("claim_id") or f"c:auto:{abs(hash(c.get('canonical_text',''))) % (10**10)}",
            canonical_text = c.get("canonical_text") or "",
            tags = tuple(c.get("tags", []) or [])
        )
        header = PerspectiveHeader(
            perspective_id = p.get("perspective_id") or f"p:auto:{abs(hash(str(p))) % (10**10)}",
            kind = p.get("kind", "cognitive"),
            owner = p.get("owner", "system"),
            traits = tuple(p.get("traits", []) or []),
            authority_hint = float(p.get("authority_hint", 1.0)),
        )
        v_reading = clamp01(float(item.get("value", 0.5)))
        w_ctx = float(item.get("weight", 1.0))
        justification = item.get("justification", "")
        ctx_tags = tuple(item.get("context_tags", []) or [])
        return claim_obj, header, v_reading, w_ctx, justification, ctx_tags

    def _compute_alpha(self, context_weight: float, header: PerspectiveHeader) -> float:
        base = max(0.05, min(1.0, context_weight))
        if header.kind == "affective":  return min(1.0, base * 1.3)
        if header.kind == "preference": return min(1.0, base * 0.8)
        return base

    def _effective_weight(self, item_weight: float, header: PerspectiveHeader) -> float:
        # Reliability multiplier
        rel = self.reliability.weight(header.perspective_id)
        return max(0.0, item_weight) * max(0.0, header.authority_hint) * rel

    def _fuse_and_audit(self, claim: Claim, items: List[Dict[str, Any]]) -> Tuple[float, DissentProfile, Dict[str, Any]]:
        vals = [(it["value"], it["weight"]) for it in items]
        V, profile = fuse_centered(vals)
        # contradiction audit (scalar gaps between contributing perspectives)
        diffs, strong = [], []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = abs(items[i]["value"] - items[j]["value"])
                diffs.append(d)
                if d > self.config.tun.strong_contradiction_gap:
                    strong.append((items[i]["header"].perspective_id, items[j]["header"].perspective_id, d))
        audit = {
            "count": len(strong),
            "avg_gap": statistics.mean(diffs) if diffs else 0.0,
            "max_gap": max(diffs) if diffs else 0.0,
            "strong_pairs": strong
        }
        return V, profile, audit

    # --------- UPDATED: rolling, variance-aware BOTH-band adaptation ---------
    def _adapt_both_band(self, claim_id: str, fused_V: float, cur_rate: float) -> Tuple[float, float]:
        """
        Blend current contradiction rate with rolling history and fused-value variance to
        set BOTH band width. Clamped to [0.20, 0.60].
        """
        # Rolling contradiction rate (mean over window)
        hist = self._contra_hist.get(claim_id)
        mean_rate = (sum(hist) / len(hist)) if hist and len(hist) > 0 else cur_rate

        # Fused-value variance over recent window (0.. ~0.25), scaled to [0,1]
        rec = self._recent_by_claim.get(claim_id)
        var = statistics.pvariance(rec) if rec and len(rec) >= 2 else 0.0
        var_term = min(1.0, 4.0 * var)  # gentle influence of volatility

        # Blend: base 0.20 + 0.30*mean_rate + 0.10*cur_rate + 0.10*var_term
        width = 0.20 + 0.30 * mean_rate + 0.10 * cur_rate + 0.10 * var_term
        width = max(0.20, min(0.60, width))

        center = 0.5
        lo, hi = center - width / 2, center + width / 2

        # Reflect into tunables for visibility/logging
        self.config.tun.both_band_lo, self.config.tun.both_band_hi = lo, hi
        return (lo, hi)

    def _verdict_from_scalar(self, V: float, both_band: Tuple[float, float]) -> str:
        lo, hi = both_band
        if V <= lo: return "FALSE"
        if V >= hi: return "TRUE"
        return "BOTH"

    def _apply_stability_and_dissent(
        self, raw_verdict: str, st: StabilityTrack, profile: DissentProfile, both_band: Tuple[float, float]
    ) -> Tuple[str, bool, List[str]]:
        ex: List[str] = []
        verdict_final = raw_verdict
        stable = False
        def opposite_mass(v: str) -> float:
            if v == "TRUE":  return profile.mass_below
            if v == "FALSE": return profile.mass_above
            return 0.0

        if raw_verdict in ("TRUE","FALSE"):
            if not st.is_stable(self.config.tun.stability_k):
                verdict_final = "BOTH"
                ex.append(f"Insufficient stability (streak {st.streak}/{self.config.tun.stability_k}).")
            elif opposite_mass(raw_verdict) > self.config.tun.dissent_cap:
                verdict_final = "BOTH"
                ex.append(f"Dissent too high (opposition mass {opposite_mass(raw_verdict):.2f} > cap {self.config.tun.dissent_cap:.2f}).")
            else:
                stable = True
                ex.append("Stable assertion with low dissent.")
        else:
            ex.append("Within BOTH band; contradiction/ambiguity retained.")
        return verdict_final, stable, ex

    def _style_from_context(self, verdicts_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not verdicts_payload:
            return {"tone":"pragmatic","verbosity":"low","include_counterpoints":False,"ask_clarifying_questions":True}
        mean_humility = sum(v["humility"] for v in verdicts_payload) / len(verdicts_payload)
        any_both = any(v["verdict"]=="BOTH" for v in verdicts_payload)
        any_unstable = any((not v["stable"]) and v["verdict"]!="BOTH" for v in verdicts_payload)
        if mean_humility > 0.6 or any_both or any_unstable:
            return {"tone":"exploratory","verbosity":"medium","include_counterpoints":True,"ask_clarifying_questions":True,
                    "user_specific_notes":"Retain contradictions explicitly when present."}
        return {"tone":"pragmatic","verbosity":"low","include_counterpoints":True,"ask_clarifying_questions":False,
                "user_specific_notes":"Be direct; include brief rationale."}

# =============================================================================
# 10) Mock LLM / Adapters
# =============================================================================
class MockLLM:
    def __init__(self, user_id: str = "user:john", seed: int = 123):
        self.user_id = user_id
        self.rng = random.Random(seed)

    def extract_claims(self, prompt: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        parts = [p.strip() for p in re.split(r'[.?!;\n]+', prompt) if p.strip()]
        claims = []
        for text in parts:
            canon = re.sub(r"\s+", " ", text).strip()
            claim_id = f"c:{abs(hash(canon)) % (10**10)}"
            claims.append({"claim_id": claim_id, "canonical_text": canon, "tags": tags or []})
        return {"version":"v1","user_id":self.user_id,"session_id":new_id("sess"),"timestamp":utcnow().isoformat(),"claims":claims}

    def perspectives_for_claims(self, claims_msg: Dict[str, Any]) -> Dict[str, Any]:
        items = []
        for c in claims_msg.get("claims", []):
            text = c["canonical_text"].lower()
            p_obs = {"perspective_id":"p:observation","kind":"cognitive","owner":"system","traits":["evidence"],"authority_hint":1.2}
            p_contra = {"perspective_id":"p:contrarian","kind":"cognitive","owner":"system","traits":["skeptical"],"authority_hint":0.9}
            p_empathy = {"perspective_id":"p:empathy","kind":"affective","owner":"system","traits":["supportive"],"authority_hint":0.8}
            p_user = {"perspective_id":f"p:{self.user_id}:prefers-pragmatic","kind":"preference","owner":self.user_id,"traits":["concise"],"authority_hint":1.0}

            def w(x): return round(max(0.2, min(1.0, x)), 2)

            if any(k in text for k in ["always","never","impossible"]):
                v_obs, v_contra = 0.35, 0.65
            elif any(k in text for k in ["is","are","was","were","has","have","contains","equals"]) and re.search(r"\d", text):
                v_obs, v_contra = 0.82, 0.42
            else:
                v_obs, v_contra = 0.56, 0.44

            v_empathy = 0.52 if any(k in text for k in ["concern","worry","help","sad"]) else 0.5
            v_user = 0.62  # style preference persistence

            for (p, v, wt, just, tags) in [
                (p_obs, v_obs, 1.0, "Observation perspective reading", ["default"]),
                (p_contra, v_contra, 0.8, "Contrarian counter-reading", ["default"]),
                (p_empathy, v_empathy, 0.6, "Affective balance", ["affective"]),
                (p_user, v_user, 0.5, "User style preference (pragmatic)", ["preference"]),
            ]:
                items.append({
                    "claim": c, "perspective": p, "value": v, "weight": w(wt),
                    "justification": just, "context_tags": tags
                })
        return {"version":"v1","user_id":claims_msg.get("user_id"),"session_id":claims_msg.get("session_id"),
                "timestamp":utcnow().isoformat(),"items":items}

# =============================================================================
# 11) Demo
# =============================================================================
def demo():
    print("\n=== PPC Synth Engine Demo ===")
    engine = EpistemicEngine()
    llm = MockLLM(user_id="user:john")

    prompt1 = "AI always replaces human jobs. The 2024 report shows 8% productivity gains with AI tools."
    claims1 = llm.extract_claims(prompt1, tags=["ai","jobs","report"])
    pers1 = llm.perspectives_for_claims(claims1)

    print("\n-- Turn 1: Claims --")
    for c in claims1["claims"]:
        print("  ", c["canonical_text"])

    out1 = engine.process_turn(claims1, pers1)
    print("\n-- Turn 1: Verdicts --")
    for v in out1["verdicts"]:
        print(f"  {v['canonical_text']!r} -> {v['verdict']}  V={v['fused_value']:.3f}  LP={v['lp_pair']}  band=[{v['both_band']['lo']:.2f},{v['both_band']['hi']:.2f}]")
        if v["explanations"]:
            print("     reasons:", "; ".join(v["explanations"]))
    print("  style:", out1["style_directives"])

    # After two days (to see decay/asymmetry and reinforcement behavior)
    future = utcnow() + timedelta(days=2)
    prompt2 = "AI can displace some roles, but it also augments human work. Workers with training were 12% faster last quarter."
    claims2 = llm.extract_claims(prompt2, tags=["ai","work","training"])
    pers2 = llm.perspectives_for_claims(claims2)

    print("\n-- Turn 2: Claims (after 2 days) --")
    for c in claims2["claims"]:
        print("  ", c["canonical_text"])

    out2 = engine.process_turn(claims2, pers2, now=future)
    print("\n-- Turn 2: Verdicts --")
    for v in out2["verdicts"]:
        print(f"  {v['canonical_text']!r} -> {v['verdict']}  V={v['fused_value']:.3f}  LP={v['lp_pair']}  band=[{v['both_band']['lo']:.2f},{v['both_band']['hi']:.2f}]")
        if v["explanations"]:
            print("     reasons:", "; ".join(v["explanations"]))
    print("  style:", out2["style_directives"])

    # Show reliability weights learned so far (per perspective/persona)
    print("\n-- Reliability weights --")
    for pid in ["p:observation","p:contrarian","p:empathy",f"p:{llm.user_id}:prefers-pragmatic"]:
        print(f"  {pid}: {engine.reliability.weight(pid):.2f}")

if __name__ == "__main__":
    demo()
