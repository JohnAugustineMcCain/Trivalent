# perspectival_engine.py
# Complete, single-file implementation of the Perspectival Paraconsistent Contextual Logic engine
# Philosophy: Both (0.5) is the default substrate; "confidence" IS the truth value per perspective.
# Cache ONLY perspectives (per-claim scalars + time). Prompts are not cached.
# Engine is sovereign about truth; LLM handles rhetoric (tone via humility).
#
# Sections:
#   0) Time helpers
#   1) Truth scalar + centered logic ops
#   2) Data model (Claim, PerspectiveHeader, cache records)
#   3) Decay policy (True stable ~1 week; False fragile ~1 day) + snap/evict
#   4) In-memory PerspectiveCache
#   5) Fusion + dissent profiling
#   6) Verdict policy, humility, stability
#   7) EpistemicEngine (consumes LLM messages, updates cache, fuses, outputs verdicts + style)
#   8) Mock LLM + glue: claim extraction, perspective generation
#   9) Demo (two turns to showcase reinforcement/decay + adaptive style)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Any, DefaultDict
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import math
import uuid
import re
import pprint

# =========================
# 0) Time helpers
# =========================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def days_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 86400.0

# =========================
# 1) Truth scalar & centered ops
# =========================
FALSE: float = 0.0
BOTH:  float = 0.5  # default epistemic substrate
TRUE:  float = 1.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def delta(v: float) -> float:
    """Centered deviation from BOTH=0.5."""
    return v - 0.5

def neg(v: float) -> float:
    """Negation: fixed point at 0.5, symmetric."""
    return 1.0 - v

def conj(v: float, u: float) -> float:
    """AND: centered min (keeps 0.5 neutral)."""
    return 0.5 + min(delta(v), delta(u))

def disj(v: float, u: float) -> float:
    """OR: centered max (keeps 0.5 neutral)."""
    return 0.5 + max(delta(v), delta(u))

def impl(v: float, u: float) -> float:
    """Implication: max(¬v, u) lifted to the scalar domain."""
    return max(neg(v), u)

# =========================
# 2) IDs & basic types
# =========================

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
    authority_hint: float = 1.0  # used as a default weight in fusion

# One time-stamped reading for a (perspective, claim)
@dataclass
class PerspectiveValuePoint:
    value: float                  # 0.0=F, 0.5=BOTH, 1.0=T  (this IS the epistemic confidence/truth)
    event_time: datetime
    context_tags: Tuple[str, ...] = ()
    justification: str = ""

# Collection of readings for one claim under a given perspective
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

# A perspective with its per-claim time series
@dataclass
class PerspectiveRecord:
    header: PerspectiveHeader
    claims: Dict[str, ClaimSeries] = field(default_factory=dict)
    last_active: Optional[datetime] = None

    def touch(self, t: Optional[datetime] = None) -> None:
        self.last_active = t or utcnow()

# =========================
# 3) Decay, snap-to-both, eviction
# =========================

@dataclass
class DecayPolicy:
    # Defaults from your spec:
    half_life_true_days: float = 7.0        # True-leaning should be more stable
    half_life_false_days: float = 1.0       # False-leaning should be more fragile
    snap_to_both_band: Tuple[float, float] = (0.4, 0.6)  # band for snapping unreinforced values to 0.5
    evict_after_days_at_exact_both: float = 14.0         # eviction window after snap

    # Optional: category-specific half-lives (affect/preference)
    half_life_affective_true_days: Optional[float] = 3.0
    half_life_preference_true_days: Optional[float] = 21.0

    def _half_life_for(self, header: PerspectiveHeader, last_v: float) -> float:
        """Pick half-life by perspective kind + which side of center it's on."""
        if header.kind == "affective" and last_v > 0.5 and self.half_life_affective_true_days:
            return self.half_life_affective_true_days
        if header.kind == "preference" and last_v > 0.5 and self.half_life_preference_true_days:
            return self.half_life_preference_true_days
        if last_v > 0.5:
            return self.half_life_true_days
        elif last_v < 0.5:
            return self.half_life_false_days
        else:
            return 0.0  # already at 0.5

    def apply_decay(self, header: PerspectiveHeader, v0: float, t0: datetime, now: datetime) -> float:
        """Exponential pull toward 0.5 with asymmetric half-lives (True slower, False faster)."""
        if v0 == 0.5:
            return 0.5
        hl = self._half_life_for(header, v0)
        if hl <= 0:
            return 0.5
        dt_days = days_between(t0, now)
        vt = 0.5 + (v0 - 0.5) * math.pow(0.5, dt_days / hl)
        return clamp01(vt)

    def snap_or_status(self, v: float, last_update: datetime, now: datetime) -> Tuple[float, str]:
        """Snap to exact 0.5 inside the snap band when unreinforced; evict after N days at 0.5."""
        lo, hi = self.snap_to_both_band
        status = "active"
        vv = v
        if lo <= v <= hi:
            vv = 0.5
            status = "inactive"
            if days_between(last_update, now) >= self.evict_after_days_at_exact_both:
                status = "evicted"
        return vv, status

# =========================
# 4) In-memory cache (perspectives only)
# =========================

class PerspectiveCache:
    """
    Stores only perspectives and their per-claim time series. Prompts are NOT stored.
    Provides decay, snap-to-both, and eviction logic on read/maintenance.
    """

    def __init__(self, policy: Optional[DecayPolicy] = None):
        self._perspectives: Dict[str, PerspectiveRecord] = {}
        self.policy = policy or DecayPolicy()

    # -- Perspective management --

    def upsert_perspective(self, header: PerspectiveHeader) -> PerspectiveRecord:
        rec = self._perspectives.get(header.perspective_id)
        if rec is None:
            rec = PerspectiveRecord(header=header)
            self._perspectives[header.perspective_id] = rec
        else:
            rec.header = header  # allow updates of traits/authority/kind/owner
        rec.touch()
        return rec

    def get_perspective(self, perspective_id: str) -> Optional[PerspectiveRecord]:
        return self._perspectives.get(perspective_id)

    def list_perspectives(self) -> List[PerspectiveRecord]:
        return list(self._perspectives.values())

    # -- Read/Write points --

    def write_point(
        self,
        header: PerspectiveHeader,
        claim: Claim,
        value: float,
        event_time: Optional[datetime] = None,
        context_tags: Iterable[str] = (),
        justification: str = "",
    ) -> None:
        """Append a new reading; resets decay clock for that perspective+claim."""
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
        """
        Return the decayed/snapshotted value and status for this (perspective, claim).
        If no series exists, return (0.5, "new", None).
        """
        now = now or utcnow()
        rec = self._perspectives.get(header.perspective_id)
        if rec is None:
            return BOTH, "new", None
        series = rec.claims.get(claim_id)
        if series is None or not series.points:
            return BOTH, "new", None

        last = series.latest()
        assert last is not None
        v = self.policy.apply_decay(rec.header, last.value, last.event_time, now)
        v, status = self.policy.snap_or_status(v, last.event_time, now)
        return v, status, series.last_updated

    # -- Maintenance (batch decay/snap/evict) --

    def maintenance(self, now: Optional[datetime] = None) -> None:
        now = now or utcnow()
        to_evict: List[Tuple[str, str]] = []  # (perspective_id, claim_id)
        for pid, rec in list(self._perspectives.items()):
            rec.touch(now)
            for cid, series in list(rec.claims.items()):
                if not series.points:
                    continue
                last = series.latest()
                if last is None:
                    continue
                v = self.policy.apply_decay(rec.header, last.value, last.event_time, now)
                v2, status = self.policy.snap_or_status(v, last.event_time, now)

                if status == "evicted":
                    to_evict.append((pid, cid))
                else:
                    if v2 == 0.5 and (series.status != "inactive" or last.value != 0.5):
                        series.points.append(
                            PerspectiveValuePoint(value=0.5, event_time=now, justification="snap-to-both")
                        )
                        series.last_updated = now
                    series.status = status

        for pid, cid in to_evict:
            rec = self._perspectives.get(pid)
            if rec and cid in rec.claims:
                del rec.claims[cid]
            # Keep empty perspective shells (identity), or GC if desired.

# =========================
# 5) Fusion & dissent helpers
# =========================

@dataclass
class DissentProfile:
    mass_above: float
    mass_below: float
    n_perspectives: int
    notes: str = ""

def fuse_centered(values_and_weights: Iterable[Tuple[float, float]]) -> Tuple[float, DissentProfile]:
    """
    Paraconsistent fusion:
    V = 0.5 + sum_i w_i*(v_i-0.5) / (sum_i |w_i| + eps)
    Also compute a dissent profile: weighted mass above/below 0.5.
    """
    eps = 1e-9
    num = 0.0
    den = 0.0
    mass_above = 0.0
    mass_below = 0.0
    n = 0
    for v, w in values_and_weights:
        dv = delta(v)
        num += w * dv
        den += abs(w)
        if v > 0.5:
            mass_above += w
        elif v < 0.5:
            mass_below += w
        n += 1
    V = 0.5 + (num / (den + eps)) if n > 0 else 0.5
    profile = DissentProfile(
        mass_above=mass_above,
        mass_below=mass_below,
        n_perspectives=n,
        notes="Opposition exists" if (mass_above > 0 and mass_below > 0) else "Unanimous or empty"
    )
    return clamp01(V), profile

# =========================
# 6) Verdict bands & stability
# =========================

@dataclass
class VerdictPolicy:
    both_band: Tuple[float, float] = (0.2, 0.8)  # Both if V ∈ [0.2, 0.8]
    deassert_band: Tuple[float, float] = (0.25, 0.75)  # hysteresis (optional)
    stability_k: int = 2            # consecutive evaluations to assert T/F
    dissent_cap: float = 0.20       # opposite-side mass cap to allow assertion

    def verdict_from_scalar(self, V: float) -> str:
        lo, hi = self.both_band
        if V <= lo:
            return "FALSE"
        if V >= hi:
            return "TRUE"
        return "BOTH"

def humility_from_scalar(V: float) -> float:
    """h = 1 - 2*|V-0.5| (near 0.5 => humble)"""
    return clamp01(1.0 - 2.0 * abs(delta(V)))

@dataclass
class StabilityTrack:
    last_verdict: Optional[str] = None
    streak: int = 0

    def update(self, verdict: str) -> None:
        if verdict == self.last_verdict and verdict in ("TRUE", "FALSE"):
            self.streak += 1
        else:
            self.streak = 1 if verdict in ("TRUE", "FALSE") else 0
        self.last_verdict = verdict

    def is_stable(self, required_k: int) -> bool:
        return self.streak >= required_k if self.last_verdict in ("TRUE", "FALSE") else False

# =========================
# 7) Epistemic Engine
# =========================

def compute_alpha(context_weight: float, header: PerspectiveHeader) -> float:
    """
    Contextual learning rate in [0,1].
    Start with base proportional to the LLM-supplied 'weight' and modulate by kind.
    """
    base = max(0.05, min(1.0, context_weight))
    if header.kind == "affective":
        return min(1.0, base * 1.3)
    if header.kind == "preference":
        return min(1.0, base * 0.8)
    return base

def effective_weight(item_weight: float, header: PerspectiveHeader) -> float:
    """Fusion weight per perspective reading on this turn."""
    return max(0.0, item_weight) * max(0.0, header.authority_hint)

@dataclass
class EngineConfig:
    verdict_policy: VerdictPolicy = field(default_factory=VerdictPolicy)
    decay_policy: DecayPolicy = field(default_factory=DecayPolicy)
    run_maintenance_each_turn: bool = True

class EpistemicEngine:
    def __init__(self, cache: Optional[PerspectiveCache] = None, config: Optional[EngineConfig] = None):
        self.cache = cache or PerspectiveCache()
        self.config = config or EngineConfig()
        self._stability: Dict[str, StabilityTrack] = defaultdict(StabilityTrack)

    def process_turn(
        self,
        claims_msg: Dict[str, Any],
        perspectives_msg: Dict[str, Any],
        now: Optional[datetime] = None
    ) -> Dict[str, Any]:
        now = now or utcnow()
        vp = self.config.verdict_policy

        user_id = claims_msg.get("user_id")
        session_id = claims_msg.get("session_id")

        # 1) Extract claims
        claims = self._parse_claims(claims_msg)

        # 2) Apply perspective updates and collect (v, w) for fusion
        items_by_claim: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in perspectives_msg.get("items", []):
            claim_obj, header, v_reading, w_ctx, justification, ctx_tags = self._parse_perspective_item(item)

            # conservative learning
            v_current, _, _ = self.cache.read_current_value(header, claim_obj.claim_id, now=now)
            alpha = compute_alpha(w_ctx, header)
            v_new = 0.5 + alpha * (v_reading - 0.5) + (1.0 - alpha) * (v_current - 0.5)
            v_new = clamp01(v_new)

            self.cache.write_point(
                header=header,
                claim=claim_obj,
                value=v_new,
                event_time=now,
                context_tags=ctx_tags,
                justification=justification or "update"
            )

            w_eff = effective_weight(w_ctx, header)
            items_by_claim[claim_obj.claim_id].append({
                "header": header,
                "claim": claim_obj,
                "value": v_new,
                "weight": w_eff
            })

        if self.config.run_maintenance_each_turn:
            self.cache.maintenance(now=now)

        # 3) Fuse, determine verdicts & humility
        verdicts_payload: List[Dict[str, Any]] = []
        for c in claims:
            V, profile = self._fuse_for_claim(c, items_by_claim[c.claim_id], now=now)
            verdict, stable, explanations = self._apply_verdict_rules(c.claim_id, V, profile, vp)
            verdicts_payload.append({
                "claim_id": c.claim_id,
                "fused_value": V,
                "verdict": verdict,
                "humility": humility_from_scalar(V),
                "stable": stable,
                "dissent_profile": {
                    "mass_above": profile.mass_above,
                    "mass_below": profile.mass_below,
                    "n_perspectives": profile.n_perspectives,
                    "notes": profile.notes
                },
                "explanations": explanations
            })

        # 4) Style directives (pragmatic defaults)
        style_directives = self._style_from_context(verdicts_payload)

        return {
            "version": "v1",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": now.isoformat(),
            "verdicts": verdicts_payload,
            "style_directives": style_directives
        }

    # ----- internals -----

    def _parse_claims(self, claims_msg: Dict[str, Any]) -> List[Claim]:
        out: List[Claim] = []
        for c in claims_msg.get("claims", []):
            claim_id = c.get("claim_id")
            text = c.get("canonical_text") or ""
            tags = tuple(c.get("tags", []) or [])
            if not claim_id:
                claim_id = f"c:auto:{abs(hash(text)) % (10**10)}"
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

    def _values_for_claim_from_items(self, items: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        return [(it["value"], it["weight"]) for it in items]

    def _fuse_for_claim(self, claim: Claim, items: List[Dict[str, Any]], now: datetime) -> Tuple[float, DissentProfile]:
        vals = self._values_for_claim_from_items(items)
        V, profile = fuse_centered(vals)
        return V, profile

    def _apply_verdict_rules(
        self, claim_id: str, V: float, profile: DissentProfile, vp: VerdictPolicy
    ) -> Tuple[str, bool, List[str]]:
        explanations: List[str] = []
        raw_verdict = vp.verdict_from_scalar(V)

        st = self._stability[claim_id]
        st.update(raw_verdict)

        def opposite_mass(v: str) -> float:
            if v == "TRUE":   return profile.mass_below
            if v == "FALSE":  return profile.mass_above
            return 0.0

        verdict_final = raw_verdict
        stable_flag = False

        if raw_verdict in ("TRUE", "FALSE"):
            opp = opposite_mass(raw_verdict)
            if not st.is_stable(vp.stability_k):
                verdict_final = "BOTH"
                explanations.append(f"Insufficient stability (streak {st.streak}/{vp.stability_k}).")
            elif opp > vp.dissent_cap:
                verdict_final = "BOTH"
                explanations.append(f"Dissent too high (opposition mass {opp:.2f} > cap {vp.dissent_cap:.2f}).")
            else:
                stable_flag = True
                explanations.append("Stable assertion with low dissent.")
        else:
            explanations.append("Within BOTH band; contradiction/ambiguity retained.")

        return verdict_final, stable_flag, explanations

    def _style_from_context(self, verdicts_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not verdicts_payload:
            return {"tone": "pragmatic", "verbosity": "low", "include_counterpoints": False, "ask_clarifying_questions": True}

        mean_humility = sum(v["humility"] for v in verdicts_payload) / len(verdicts_payload)
        any_both = any(v["verdict"] == "BOTH" for v in verdicts_payload)
        any_unstable = any(not v["stable"] and v["verdict"] != "BOTH" for v in verdicts_payload)

        if mean_humility > 0.6 or any_both or any_unstable:
            return {
                "tone": "exploratory",
                "verbosity": "medium",
                "include_counterpoints": True,
                "ask_clarifying_questions": True,
                "user_specific_notes": "Retain contradictions explicitly when present."
            }
        else:
            return {
                "tone": "pragmatic",
                "verbosity": "low",
                "include_counterpoints": True,
                "ask_clarifying_questions": False,
                "user_specific_notes": "Be direct; include brief rationale."
            }

# =========================
# 8) Mock LLM & glue
# =========================

class MockLLM:
    """
    - Extracts evaluable claims from a raw prompt (simple heuristic).
    - Generates perspective readings (value v in [0,1]) with context weights and justifications.
    """

    def __init__(self, user_id: str = "user:john"):
        self.user_id = user_id

    # ---- Claim extraction ----
    def extract_claims(self, prompt: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        # Naive sentence/phrase splitter; real system would use proper NLP.
        parts = [p.strip() for p in re.split(r'[.?!;\n]+', prompt) if p.strip()]
        claims = []
        for i, text in enumerate(parts):
            # canonicalize: lowercase & strip extra spaces (but store original text)
            canon = re.sub(r"\s+", " ", text).strip()
            claim_id = f"c:{abs(hash(canon)) % (10**10)}"
            claims.append({"claim_id": claim_id, "canonical_text": canon, "tags": tags or []})
        return {
            "version": "v1",
            "user_id": self.user_id,
            "session_id": new_id("sess"),
            "timestamp": utcnow().isoformat(),
            "claims": claims
        }

    # ---- Perspective generation ----
    def perspectives_for_claims(self, claims_msg: Dict[str, Any]) -> Dict[str, Any]:
        items = []
        for c in claims_msg.get("claims", []):
            text = c["canonical_text"].lower()

            # Base perspectives: observation (evidence-seeking), contrarian, empathy (affective), user preference persona
            p_obs = {
                "perspective_id": "p:observation",
                "kind": "cognitive", "owner": "system",
                "traits": ["evidence-seeking","cautious"], "authority_hint": 1.2
            }
            p_contra = {
                "perspective_id": "p:contrarian",
                "kind": "cognitive", "owner": "system",
                "traits": ["skeptical"], "authority_hint": 0.9
            }
            p_empathy = {
                "perspective_id": "p:empathy",
                "kind": "affective", "owner": "system",
                "traits": ["supportive","calm"], "authority_hint": 0.8
            }
            p_user_pref = {
                "perspective_id": f"p:{self.user_id}:prefers-pragmatic",
                "kind": "preference", "owner": self.user_id,
                "traits": ["concise","pragmatic"], "authority_hint": 1.0
            }

            # Heuristic readings for demo:
            # - If claim looks like a universal ("always", "never"), observation leans skeptical (False-ish), contrarian may lean True-ish.
            # - If claim is plainly factual-looking (numbers, dates), observation leans True-ish.
            # - Empathy prefers balance (near 0.5) unless sentiment words present.
            # - User preference doesn't assert truth about the world; it gently biases style, so keep it near center but >0.5 to persist.
            def w(x): return round(max(0.2, min(1.0, x)), 2)

            if any(k in text for k in ["always", "never", "impossible"]):
                v_obs = 0.35
                v_contra = 0.65
            elif any(k in text for k in ["is", "are", "was", "were", "has", "have", "contains", "equals"]) and re.search(r"\d", text):
                v_obs = 0.8
                v_contra = 0.4
            else:
                v_obs = 0.55
                v_contra = 0.45

            v_empathy = 0.52 if any(k in text for k in ["concern", "worry", "help", "sad"]) else 0.5
            v_pref = 0.62  # gentle push toward "pragmatic/concise" style as a preference-perspective

            items.extend([
                {
                    "claim": c, "perspective": p_obs,
                    "value": v_obs, "weight": w(1.0),
                    "justification": "Observation perspective reading", "context_tags": ["default"]
                },
                {
                    "claim": c, "perspective": p_contra,
                    "value": v_contra, "weight": w(0.8),
                    "justification": "Contrarian counter-reading", "context_tags": ["default"]
                },
                {
                    "claim": c, "perspective": p_empathy,
                    "value": v_empathy, "weight": w(0.6),
                    "justification": "Affective balance", "context_tags": ["affective"]
                },
                {
                    "claim": c, "perspective": p_user_pref,
                    "value": v_pref, "weight": w(0.5),
                    "justification": "User style preference (pragmatic)", "context_tags": ["preference"]
                }
            ])

        return {
            "version": "v1",
            "user_id": claims_msg.get("user_id"),
            "session_id": claims_msg.get("session_id"),
            "timestamp": utcnow().isoformat(),
            "items": items
        }

# =========================
# 9) Demo
# =========================

def demo():
    print("\n=== Perspectival Engine Demo ===")
    engine = EpistemicEngine()
    llm = MockLLM(user_id="user:john")

    # Turn 1: user asks something with a universal claim and a factual-looking claim
    prompt1 = (
        "AI always replaces human jobs. "
        "The 2024 report shows 8% productivity gains with AI tools."
    )
    claims_msg1 = llm.extract_claims(prompt1, tags=["ai","jobs","report"])
    perspectives_msg1 = llm.perspectives_for_claims(claims_msg1)

    print("\n-- Turn 1: Input Claims --")
    pprint.pprint(claims_msg1["claims"])

    out1 = engine.process_turn(claims_msg1, perspectives_msg1)
    print("\n-- Turn 1: EpistemicVerdicts --")
    pprint.pprint(out1)

    # Simulate time passing to see decay/maintenance effects
    future_time = utcnow() + timedelta(days=2)

    # Turn 2: user follows up with a more nuanced statement (reinforcement chance)
    prompt2 = (
        "AI can displace some roles, but it also augments human work. "
        "Workers with training were 12% faster last quarter."
    )
    claims_msg2 = llm.extract_claims(prompt2, tags=["ai","work","training"])
    perspectives_msg2 = llm.perspectives_for_claims(claims_msg2)

    print("\n-- Turn 2: Input Claims --")
    pprint.pprint(claims_msg2["claims"])

    out2 = engine.process_turn(claims_msg2, perspectives_msg2, now=future_time)
    print("\n-- Turn 2: EpistemicVerdicts (after 2 days) --")
    pprint.pprint(out2)

    # Show some cache internals for one perspective to illustrate identity building
    print("\n-- Cache peek: perspective 'p:observation' series sizes --")
    p_obs = engine.cache.get_perspective("p:observation")
    if p_obs:
        for cid, series in p_obs.claims.items():
            print(f"  claim {cid}: {len(series.points)} points; last status={series.status}")

if __name__ == "__main__":
    demo()