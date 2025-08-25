#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized PEACE Oracle — Trivalent, Problem-Agnostic, Calibration-First

Key properties
- Truth values: FALSE / TRUE / BOTH  (no UNKNOWN; BOTH subsumes undecidable/unknown/contradiction)
- Problem-agnostic adapters produce bounded probes & discrete observations (binary or counts)
- Mixture-of-experts calibration with randomized PIT for discrete distributions
- EarnedConfidence meter bumps only on agreement (forecast ~ observation)
- WAL-backed SQLite provenance for runs, probes, bumps, verdicts
- Self-modification interfaces preserved (stubs still, safety gates intact)

Run demo (toy “count up to N” adapter just to exercise the pipeline):
    python3 peace_oracle.py --trials 200 --seed 2025
"""

from __future__ import annotations

import ast, argparse, json, math, random, sqlite3, threading, time, hashlib, logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from contextlib import contextmanager

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("peace_oracle.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------
# Truth values (strictly trivalent)
# -------------------------
class TV(Enum):
    FALSE = 0
    TRUE  = 1
    BOTH  = 2   # includes: undecidable/unknown/inconsistent

# -------------------------
# Core dataclasses
# -------------------------
@dataclass
class Verdict:
    tv: TV
    confidence: float
    reason: Dict[str, Any]
    provenance: Dict[str, Any]
    witness: Optional[Any] = None  # required for existential TRUE

class EarnedConfidence:
    def __init__(self): self.cc = 0.0
    def bump(self, eps: float = 0.002): self.cc = min(1.0, self.cc + eps)

@dataclass(frozen=True)
class MathematicalProblem:
    name: str
    description: str
    complexity_score: int      # 1-10
    computational_bound: int   # budget for small checks
    problem_type: str          # "number_theory", "combinatorics", "analysis", ...
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
    modification_type: str      # "add" | "modify" | "delete"
    safety_score: float
    mathematical_soundness: float
    reasoning: str
    timestamp: float
    def __hash__(self) -> int: return hash(self.modification_id)

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
            conf = self.confidence_fn(statement)
            self.memory[key] = (verdict, conf)
        return self.memory[key]

# -------------------------
# WAL-backed provenance store
# -------------------------
def _canon(x: Any) -> str:
    try: return json.dumps(x, sort_keys=True, separators=(",",":"))
    except Exception: return str(x)

def _sid(*parts: str, prefix: str="id_", n: int=12) -> str:
    h = hashlib.sha256("||".join(parts).encode()).hexdigest()
    return f"{prefix}{h[:n]}"

class VersionedPEACECache:
    def __init__(self, db_path: str = "peace_cache.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
        self._initialize_db()

    def _initialize_db(self):
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created REAL,
                    seed_core INTEGER,
                    code_hash TEXT
                );
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
                CREATE TABLE IF NOT EXISTS probes (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT,
                    problem TEXT,
                    trial TEXT,
                    forecast TEXT,
                    observed TEXT,
                    pit REAL,
                    ci REAL,
                    pmf_at_y REAL,
                    bump INTEGER,
                    elapsed REAL,
                    created REAL
                );
                CREATE TABLE IF NOT EXISTS bumps (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT,
                    cc REAL,
                    eps REAL,
                    created REAL
                );
                CREATE TABLE IF NOT EXISTS verdicts (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT,
                    claim TEXT,
                    verdict_tv TEXT,
                    confidence REAL,
                    reason TEXT,
                    provenance TEXT,
                    witness TEXT,
                    created REAL
                );
                CREATE INDEX IF NOT EXISTS idx_statement_hash ON evaluations(statement_hash);
                CREATE INDEX IF NOT EXISTS idx_modification_id ON code_modifications(modification_id);
            """)
            self.conn.commit()

    def log_run(self, run_id: str, seed: int, code_hash: str):
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO runs(run_id,created,seed_core,code_hash) VALUES(?,?,?,?)",
                              (run_id, time.time(), seed, code_hash))
            self.conn.commit()

    def record_evaluation(self, statement: Any, verdict: TV, confidence: float, perspective: str, version: str="1.0"):
        s = _canon(statement); sh = hashlib.sha256(s.encode()).hexdigest()
        with self._lock:
            self.conn.execute("""
                INSERT INTO evaluations(statement_hash,statement,verdict,confidence,perspective,version,timestamp)
                VALUES (?,?,?,?,?,?,?)""", (sh, s, verdict.name, confidence, perspective, version, time.time()))
            self.conn.commit()

    def record_modification(self, modification: CodeModification):
        with self._lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO code_modifications
                (modification_id,target_module,target_function,original_code,modified_code,
                 safety_verdict,safety_confidence,mathematical_soundness,reasoning,timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (modification.modification_id, modification.target_module, modification.target_function,
                 modification.original_code, modification.modified_code,
                 "PENDING", modification.safety_score, modification.mathematical_soundness,
                 modification.reasoning, modification.timestamp))
            self.conn.commit()

    def get_modification_history(self, target_function: str) -> List[CodeModification]:
        with self._lock:
            cur = self.conn.execute("""
                SELECT modification_id,target_module,target_function,original_code,modified_code,
                       safety_verdict,safety_confidence,mathematical_soundness,reasoning,timestamp
                FROM code_modifications WHERE target_function=? ORDER BY timestamp DESC""",
                (target_function,))
            rows = cur.fetchall()
        mods: List[CodeModification] = []
        for r in rows:
            mods.append(CodeModification(
                modification_id=r[0], target_module=r[1], target_function=r[2],
                original_code=r[3], modified_code=r[4], modification_type="modify",
                safety_score=r[6] or 0.0, mathematical_soundness=r[7] or 0.0,
                reasoning=r[8] or "", timestamp=r[9] or time.time()))
        return mods

    def log_probe(self, run_id: str, problem: str, trial: Dict[str,Any],
                  forecast: Dict[str,Any], observed: Dict[str,Any],
                  pit: float, ci: float, pmf_at_y: float, bump: bool, elapsed: float):
        with self._lock:
            self.conn.execute("""
                INSERT INTO probes(run_id,problem,trial,forecast,observed,pit,ci,pmf_at_y,bump,elapsed,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, problem, _canon(trial), _canon(forecast), _canon(observed),
                 float(pit), float(ci), float(pmf_at_y), int(bump), float(elapsed), time.time()))
            self.conn.commit()

    def log_bump(self, run_id: str, cc: float, eps: float):
        with self._lock:
            self.conn.execute("INSERT INTO bumps(run_id,cc,eps,created) VALUES(?,?,?,?)",
                              (run_id, float(cc), float(eps), time.time()))
            self.conn.commit()

    def log_verdict(self, run_id: str, claim: Dict[str,Any], v: Verdict):
        with self._lock:
            self.conn.execute("""
                INSERT INTO verdicts(run_id,claim,verdict_tv,confidence,reason,provenance,witness,created)
                VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, _canon(claim), v.tv.name, float(v.confidence), _canon(v.reason),
                 _canon(v.provenance), _canon(v.witness), time.time()))
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

# -------------------------
# Safety perspectives (unchanged semantics)
# -------------------------
class CodeSafetyPerspective(PEACEPerspective):
    def __init__(self):
        super().__init__(
            name="code_safety",
            evaluate_fn=self._evaluate_safety,
            confidence_fn=self._compute_safety_confidence,
        )
    def _evaluate_safety(self, modification: CodeModification) -> TV:
        checks = [
            self._check_syntax(modification.modified_code),
            self._check_infinite_loops(modification.modified_code),
            self._check_memory_safety(modification.modified_code),
            self._check_side_effects(modification.modified_code),
            self._check_imports(modification.modified_code),
        ]
        safe = sum(1 for x in checks if x)
        tot = len(checks)
        if safe == tot: return TV.TRUE
        if safe == 0:   return TV.FALSE
        if safe >= int(0.7*tot): return TV.BOTH
        return TV.FALSE

    def _compute_safety_confidence(self, modification: CodeModification) -> float:
        try:
            tree = ast.parse(modification.modified_code)
            complexity = len(list(ast.walk(tree)))
            base = max(0.1, 1.0 - (complexity / 100.0))
            pattern_boost = sum(0.05 for p in ["if ", "return", "self.", "try:", "except:"] if p in modification.modified_code)
            return min(0.95, base + pattern_boost)
        except SyntaxError:
            return 0.1
        except Exception:
            return 0.3

    def _check_syntax(self, code: str) -> bool:
        try: ast.parse(code); return True
        except SyntaxError: return False

    def _check_infinite_loops(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        if not any(isinstance(n, ast.Break) for n in ast.walk(node)): return False
                elif isinstance(node, ast.For):
                    if isinstance(node.iter, ast.Call) and not (isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
                        if not any(isinstance(n, ast.Break) for n in ast.walk(node)): return False
            return True
        except Exception:
            return False

    def _check_memory_safety(self, code: str) -> bool:
        bad = ["exec(", "eval(", "globals()", "__import__", "setattr(", "delattr(", "vars()", "locals()"]
        return not any(p in code for p in bad)

    def _check_side_effects(self, code: str) -> bool:
        side = ["os.", "sys.", "subprocess.", "open(", "file(", "input(", "write(", "delete"]
        return not any(p in code and p != "print(" for p in side)

    def _check_imports(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bad = ["os", "sys", "subprocess", "pickle", "marshal"]
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in bad: return False
                    elif isinstance(node, ast.ImportFrom) and node.module in bad:
                        return False
            return True
        except Exception:
            return False

class MathematicalSoundnessPerspective(PEACEPerspective):
    def __init__(self):
        super().__init__(
            name="mathematical_soundness",
            evaluate_fn=self._evaluate_soundness,
            confidence_fn=self._compute_soundness_confidence,
        )
    def _evaluate_soundness(self, modification: CodeModification) -> TV:
        s = modification.modified_code.lower()
        methods = ["asymptotic","prime","sieve","theorem","conjecture","logarithm","probability"]
        rigor   = ["bound","limit","convergence","error","precision","confidence","estimate","approximation"]
        danger  = [" prove "," proof "," qed","therefore proven","definitively true","absolutely certain"]
        if any(d in s for d in danger): return TV.FALSE
        m = sum(1 for w in methods if w in s); r = sum(1 for w in rigor if w in s)
        if m >= 2 and r >= 1: return TV.TRUE
        if m >= 1 or r >= 1:  return TV.BOTH
        return TV.FALSE
    def _compute_soundness_confidence(self, modification: CodeModification) -> float:
        s = modification.modified_code.lower()
        kws = ["theorem","conjecture","prime","asymptotic","logarithm","analysis","heuristic","probability","distribution"]
        base = min(0.9, 0.3 + 0.1*sum(1 for k in kws if k in s))
        if "asymptotic" in s and "analysis" in s: base += 0.15
        if "bound" in s or "limit" in s: base += 0.10
        return min(0.95, base)

class CategoryErrorPerspective(PEACEPerspective):
    def __init__(self):
        super().__init__(
            name="category_error",
            evaluate_fn=self._evaluate_category_error,
            confidence_fn=lambda _ : 0.7,
        )
    def _evaluate_category_error(self, mod: CodeModification) -> TV:
        s = mod.modified_code.lower()
        impossible = [("infinite","verify"), ("all","numbers"), ("every","integer"), ("prove","conjecture")]
        bounds     = ["bound","limit","threshold","computational","feasible","approximation","heuristic","if n >"]
        meta       = ["confidence","probability","estimate","likely","suggests","indicates"]
        has_impossible = any(all(w in s for w in pair) for pair in impossible)
        has_bounds = any(w in s for w in bounds)
        has_meta   = any(w in s for w in meta)
        if has_impossible and not has_bounds: return TV.FALSE
        if has_bounds and has_meta: return TV.TRUE
        if has_bounds or has_meta: return TV.BOTH
        return TV.BOTH

# -------------------------
# LLM Interface (mock, unchanged)
# -------------------------
class LLMInterface:
    def __init__(self): self.conversation_history = []
    async def analyze_mathematical_problem(self, problem: MathematicalProblem) -> Dict[str, Any]:
        logger.info(f"Analyzing problem: {problem.name}")
        return {
            "structure": "generic_problem",
            "approaches": ["bounded_checks","probabilistic_forecasts","pattern_learning"],
            "meta_logical_potential": 0.8
        }
    async def suggest_solution_method(self, problem: MathematicalProblem, analysis: Dict) -> Dict[str, Any]:
        return {
            "primary_method": "oracle_guided_verification" if problem.complexity_score >= 6 else "direct_computation",
            "requires_code_modification": problem.complexity_score >= 7,
            "modification_targets": ["verification_engine","pattern_learner"] if problem.complexity_score >= 7 else [],
            "confidence": 0.85 if problem.complexity_score >= 6 else 0.95
        }
    async def analyze_current_code(self, module: str, fn: str) -> Dict[str, Any]:
        return {"current_capabilities":["basic_verification"],"limitations":["limited_heuristics"],"improvement_potential":0.9}
    def _stable_id(self, *parts: str, prefix: str="mod_", n: int=12) -> str:
        return _sid(*parts, prefix=prefix, n=n)
    async def propose_code_modifications(self, problem: MathematicalProblem, current: Dict, method: Dict) -> List[CodeModification]:
        if not method.get("requires_code_modification"): return []
        now = time.time()
        return [
            CodeModification(
                modification_id=self._stable_id(problem.name,"adapter_enhancement"),
                target_module="oracle.adapters", target_function="run_probe",
                original_code="# stub", modified_code="# improved stub with bounds", modification_type="modify",
                safety_score=0.0, mathematical_soundness=0.0, reasoning="Improve bounded probe efficiency",
                timestamp=now
            )
        ]

# -------------------------
# Discrete models & calibration (generic)
# -------------------------
class DiscreteModel:
    """Abstract: given params θ and outcome y, provide pmf(y;θ) and cdf(y;θ)."""
    def pmf(self, y: int, theta: Dict[str, float]) -> float: raise NotImplementedError
    def cdf(self, y: int, theta: Dict[str, float]) -> float: raise NotImplementedError

class BernoulliModel(DiscreteModel):
    # theta: {"p": prob of success}
    def pmf(self, y: int, theta: Dict[str, float]) -> float:
        p = max(1e-12, min(1-1e-12, theta.get("p", 0.5)))
        if y not in (0,1): return 0.0
        return p if y==1 else (1-p)
    def cdf(self, y: int, theta: Dict[str, float]) -> float:
        if y < 0: return 0.0
        if y < 1: return self.pmf(0, theta)
        return 1.0

class PoissonModel(DiscreteModel):
    # theta: {"lam": rate}, optional {"cap": int} to cap Y
    def pmf(self, y: int, theta: Dict[str, float]) -> float:
        lam = max(1e-12, theta.get("lam", 1.0))
        cap = theta.get("cap", None)
        if cap is None or y < cap:
            return math.exp(-lam + y*math.log(lam) - math.lgamma(y+1))
        # pile tail on cap
        cdf = sum(math.exp(-lam + k*math.log(lam) - math.lgamma(k+1)) for k in range(int(cap)))
        return max(0.0, 1.0 - cdf)
    def cdf(self, y: int, theta: Dict[str, float]) -> float:
        lam = max(1e-12, theta.get("lam", 1.0))
        cap = theta.get("cap", None)
        if cap is None:
            s = 0.0
            for k in range(max(0,y)+1):
                s += math.exp(-lam + k*math.log(lam) - math.lgamma(k+1))
            return min(1.0, s)
        s = 0.0
        for k in range(max(0,y)+1):
            if k < cap:
                s += math.exp(-lam + k*math.log(lam) - math.lgamma(k+1))
            else:
                return 1.0
        return min(1.0, s)

def _rand_pit_discrete(y: int, F: Callable[[int], float], rng: random.Random) -> float:
    Fy  = F(y)
    Fy1 = 0.0 if y <= 0 else F(y-1)
    v = rng.random()
    return Fy1 + v*(Fy - Fy1)

@dataclass
class Heuristic:
    name: str
    # given features -> params θ for the chosen model
    predict_theta: Callable[[Dict[str,Any]], Dict[str,float]]

class CalibrationEngine:
    """Mixture-of-experts over heuristics with discrete models; tracks PIT & CI."""
    def __init__(self, model: DiscreteModel, heuristics: List[Heuristic], rng: random.Random, pit_bins: int=10):
        assert heuristics, "need heuristics"
        self.model = model
        self.h = heuristics
        self.w = [1.0/len(heuristics)]*len(heuristics)
        self.rng = rng
        self.n_obs = 0
        self.pit_vals: List[float] = []
        self.loglik_each = [0.0]*len(heuristics)
        self.loglik_mix = 0.0
        self.bucket_scores = defaultdict(float)  # generic residue bucket on small moduli
        self.pit_bins = pit_bins
        self.conf_trend: List[float] = []

    def _mix_pmf_cdf(self, thetas: List[Dict[str,float]]) -> Tuple[Callable[[int],float], Callable[[int],float]]:
        def pmf(y: int) -> float:
            return max(1e-300, sum(self.w[i]*max(1e-300, self.model.pmf(y, thetas[i])) for i in range(len(thetas))))
        def cdf(y: int) -> float:
            return min(1.0, max(0.0, sum(self.w[i]*self.model.cdf(y, thetas[i]) for i in range(len(thetas)))))
        return pmf, cdf

    def forecast(self, features: Dict[str,Any]) -> Dict[str,Any]:
        thetas = [h.predict_theta(features) for h in self.h]
        mix_theta_preview = {}
        if isinstance(self.model, PoissonModel):
            mix_theta_preview["lam_mix"] = sum(self.w[i]*thetas[i].get("lam",1.0) for i in range(len(thetas)))
            mix_theta_preview["cap"] = thetas[0].get("cap")
        if isinstance(self.model, BernoulliModel):
            mix_theta_preview["p_mix"] = sum(self.w[i]*thetas[i].get("p",0.5) for i in range(len(thetas)))
        return {"weights": list(self.w), "thetas": thetas, "mix": mix_theta_preview}

    def update(self, features: Dict[str,Any], y: int, residue_key: Tuple[int,...]) -> Dict[str,Any]:
        self.n_obs += 1
        thetas = [h.predict_theta(features) for h in self.h]
        mix_pmf, mix_cdf = self._mix_pmf_cdf(thetas)
        # log-likelihoods
        pmfs = [max(1e-300, self.model.pmf(y, th)) for th in thetas]
        logs = [math.log(p) for p in pmfs]
        for i,l in enumerate(logs): self.loglik_each[i] += l
        mixp = mix_pmf(y); self.loglik_mix += math.log(mixp)
        # weight update (expert advice)
        new_w = [self.w[i]*pmfs[i] for i in range(len(self.h))]
        Z = sum(new_w); self.w = [nw/Z for nw in new_w] if Z>0 else [1.0/len(self.h)]*len(self.h)
        # PIT
        pit = _rand_pit_discrete(y, mix_cdf, self.rng)
        self.pit_vals.append(pit)
        # residue stability
        self.bucket_scores[residue_key] += math.log(mixp)
        # CI
        ci = self._confidence_index()
        self.conf_trend.append(ci)
        return {"pmf_at_y": mixp, "pit": pit, "ci": ci, "weights": list(self.w)}

    def _pit_score(self) -> float:
        if not self.pit_vals: return 0.0
        pits = sorted(self.pit_vals); n = len(pits); D = 0.0
        for i,u in enumerate(pits, start=1):
            D = max(D, abs(i/n - u), abs(u - (i-1)/n))
        c = 50.0
        return math.exp(-c*(D**2))

    def _stability_score(self) -> float:
        if not self.bucket_scores: return 1.0
        vals = list(self.bucket_scores.values())
        mean = sum(vals)/len(vals)
        var = sum((v-mean)**2 for v in vals)/max(1, len(vals)-1)
        return 1.0 / (1.0 + 0.05*var)

    def _advantage_score(self) -> float:
        best = max(self.loglik_each)
        base = self.loglik_each[0]  # treat h[0] as baseline
        adv = (best - base) / max(1,self.n_obs)
        k=2.0
        return 1.0 - math.exp(-k*max(0.0,adv))

    def _confidence_index(self) -> float:
        s_cal = self._pit_score()
        s_stab= self._stability_score()
        s_adv = self._advantage_score()
        eps=1e-6
        return max(0.0, min(1.0, math.exp((math.log(s_cal+eps)+math.log(s_stab+eps)+math.log(s_adv+eps))/3.0)))

    def summary(self) -> Dict[str,Any]:
        best_i = max(range(len(self.h)), key=lambda i: self.loglik_each[i])
        return {
            "observations": self.n_obs,
            "best_model_index": best_i,
            "best_model_name": self.h[best_i].name,
            "mix_avg_logscore": self.loglik_mix / max(1,self.n_obs),
            "pit_score": self._pit_score(),
            "stability_score": self._stability_score(),
            "advantage_score": self._advantage_score(),
            "confidence_index": self.conf_trend[-1] if self.conf_trend else 0.0
        }

# -------------------------
# Problem Adapters (generic)
# -------------------------
@dataclass
class Trial:
    features: Dict[str,Any]   # model-facing features
    run: Callable[[], Dict[str,Any]]  # returns {"y": int, ...}
    residue_key: Tuple[int,...]       # for stability buckets
    claim: Dict[str,Any] = field(default_factory=dict)

class ProblemAdapter:
    """Abstract adapter; produce bounded trials for a problem."""
    name: str
    obs_model: DiscreteModel
    heuristics: List[Heuristic]
    def next_trial(self, rng: random.Random) -> Trial: raise NotImplementedError

# --- Demo adapter (replace with real adapters) ---
# This is intentionally simple: chooses N from a range; observes y = f(N) with a tiny budget.
# Heuristics forecast Bernoulli(success) where success means "y increased since previous".
class DemoCountAdapter(ProblemAdapter):
    name = "demo_count_adapter"
    def __init__(self, max_N: int = 50_000):
        self.max_N = max_N
        # binary observation: did we see >=1 twin prime in [N/2, N] (toy)
        self.obs_model = BernoulliModel()
        self.heuristics = [
            Heuristic("H0_flat",      lambda feat: {"p": 0.3}),
            Heuristic("H1_growth",    lambda feat: {"p": min(0.95, 0.2 + 0.15*math.log10(max(10, feat["N"]))/2)}),
            Heuristic("H2_smooth",    lambda feat: {"p": min(0.9,  0.25 + 0.1 * (feat["N"]**0.25)/ (self.max_N**0.25))}),
        ]
        self._last_count = 0

    @staticmethod
    def _is_prime(k: int) -> bool:
        if k < 2: return False
        if k % 2 == 0: return k == 2
        i=3
        while i*i <= k:
            if k % i == 0: return False
            i += 2
        return True

    def _twin_primes_count_upto(self, n: int) -> int:
        count=0
        p=3
        while p+2 <= n:
            if self._is_prime(p) and self._is_prime(p+2):
                count += 1
            p += 2
        return count

    def next_trial(self, rng: random.Random) -> Trial:
        # pick N with mild bias toward larger ranges
        N = int(1_000 + rng.random()**2 * (self.max_N - 1_000))
        features = {"N": N}
        def run():
            t0 = time.time()
            # bounded check: count twins in [N//2, N] quickly
            c_before = self._last_count
            c_now = self._twin_primes_count_upto(N) - self._twin_primes_count_upto(N//2)
            self._last_count += max(0, c_now)
            success = 1 if c_now > 0 else 0
            return {"y": success, "window_twins": int(c_now), "elapsed": time.time()-t0, "N": N}
        residue = (N % 3, N % 5, N % 7)
        return Trial(features=features, run=run, residue_key=residue, claim={"exists":"twin_in_window","N":N})

# -------------------------
# Oracle Orchestrator (general)
# -------------------------
@contextmanager
def stopwatch():
    t0 = time.time()
    yield lambda: time.time() - t0

class SelfModifyingPEACEOracle:
    """Main orchestrator with calibration + earned confidence + self-mod scaffolding."""
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.cache = VersionedPEACECache()
        self.safety_perspectives = [CodeSafetyPerspective(), MathematicalSoundnessPerspective(), CategoryErrorPerspective()]
        self.active_modifications: Dict[str, CodeModification] = {}
        self.perspectives: Dict[str, PEACEPerspective] = {}  # problem-name keyed
        logger.info("Initialized Self-Modifying PEACE Oracle")

    # ---- Safety for self-mods (unchanged behavior) ----
    async def _evaluate_modification_safety(self, modification: CodeModification) -> Dict[str, Any]:
        verdicts: Dict[str, TV] = {}; confs: Dict[str, float] = {}
        for p in self.safety_perspectives:
            v,c = p.evaluate(modification)
            verdicts[p.name] = v; confs[p.name] = c
            self.cache.record_evaluation(modification, v, c, p.name)
        # integrate (no UNKNOWN key now)
        weights = {"TRUE":0.0,"FALSE":0.0,"BOTH":0.0}; tot=0.0
        for name,v in verdicts.items():
            w = confs.get(name,0.0); weights[v.name] += w; tot += w
        if tot>0: 
            for k in weights: weights[k] /= tot
        if weights["FALSE"] > 0.3: iv=TV.FALSE
        elif weights["TRUE"] > 0.7: iv=TV.TRUE
        else: iv=TV.BOTH
        return {"verdict": iv, "confidence": max(weights.values()), "score_breakdown": weights}

    async def _apply_modifications(self, mods: List[CodeModification]):
        for m in mods:
            self.cache.record_modification(m)
            self.active_modifications[m.modification_id] = m

    # ---- Problem solving loop (generic) ----
    async def solve(self,
                    problem: MathematicalProblem,
                    adapter: ProblemAdapter,
                    trials: int,
                    rng: random.Random,
                    agreement: Dict[str, float],
                    earned: EarnedConfidence,
                    run_id: str) -> Dict[str,Any]:

        # analysis & (optional) mod proposals
        analysis = await self.llm.analyze_mathematical_problem(problem)
        method = await self.llm.suggest_solution_method(problem, analysis)
        if method.get("requires_code_modification"):
            proposed = await self.llm.propose_code_modifications(problem, {}, method)
            safe_mods = []
            for mod in proposed:
                sr = await self._evaluate_modification_safety(mod)
                if (sr["verdict"] in (TV.TRUE, TV.BOTH)) and (sr["confidence"]>0.6):
                    safe_mods.append(mod)
            if safe_mods: await self._apply_modifications(safe_mods)

        # calibration engine for this adapter
        rng_pit = random.Random(rng.randrange(1<<30))
        engine = CalibrationEngine(adapter.obs_model, adapter.heuristics, rng_pit)

        # log run header
        self.cache.log_run(run_id, seed=rng.seed if hasattr(rng, "seed") else 0, code_hash=_sid(__doc__ or "", prefix="code_"))

        eps = agreement.get("eps", 0.002)
        pmf_thresh = agreement.get("pmf_thresh", 0.20)
        pit_alpha  = agreement.get("pit_alpha", 0.10)
        ci_thresh  = agreement.get("ci_thresh", 0.60)

        def agree(pmf_at_y: float, pit: float, ci: float)->bool:
            return (pmf_at_y >= pmf_thresh) and (pit_alpha <= pit <= (1.0-pit_alpha)) and (ci >= ci_thresh)

        t0=time.time()
        examples=[]
        for _ in range(trials):
            trial = adapter.next_trial(rng)
            forecast = engine.forecast(trial.features)
            with stopwatch() as elapsed:
                obs = trial.run()
            y = int(obs["y"])
            upd = engine.update(trial.features, y, trial.residue_key)
            bump = agree(upd["pmf_at_y"], upd["pit"], upd["ci"])
            if bump:
                earned.bump(eps)
                self.cache.log_bump(run_id, earned.cc, eps)
            self.cache.log_probe(run_id, problem.name, trial.claim, forecast, obs,
                                 pit=upd["pit"], ci=upd["ci"], pmf_at_y=upd["pmf_at_y"], bump=bump, elapsed=obs.get("elapsed", elapsed()))
            if len(examples) < 5 and y>0:
                examples.append({"trial": trial.claim, "obs": obs})

        elapsed = time.time()-t0
        return {
            "elapsed": elapsed,
            "per_trial_ms": 1000.0*elapsed/max(1,trials),
            "engine_summary": engine.summary(),
            "earned_cc": earned.cc,
            "examples": examples,
            "mods_applied": len(self.active_modifications)
        }

    def close(self):
        self.cache.close()
        logger.info("Oracle shutdown complete")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Generalized PEACE Oracle — trivalent, problem-agnostic")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--pmf-thresh", type=float, default=0.20)
    ap.add_argument("--pit-alpha", type=float, default=0.10)
    ap.add_argument("--ci-thresh", type=float, default=0.60)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    problem = MathematicalProblem(
        name="demo_twins_in_window",
        description="Toy adapter: binary observation of at-least-one twin-prime in [N/2,N]",
        complexity_score=6,
        computational_bound=50_000,
        problem_type="number_theory"
    )
    adapter = DemoCountAdapter(max_N=problem.computational_bound)
    earned = EarnedConfidence()
    llm = LLMInterface()
    oracle = SelfModifyingPEACEOracle(llm)
    try:
        run_id = _sid(problem.name, str(args.seed), prefix="run_")
        res = asyncio_run(oracle.solve(
            problem=problem,
            adapter=adapter,
            trials=args.trials,
            rng=rng,
            agreement={"eps":args.eps,"pmf_thresh":args.pmf_thresh,"pit_alpha":args.pit_alpha,"ci_thresh":args.ci_thresh},
            earned=earned,
            run_id=run_id
        ))
        print("=== Generalized PEACE Oracle ===")
        print(f"run_id: {run_id}")
        print(f"trials: {args.trials}  seed: {args.seed}")
        print(f"elapsed: {res['elapsed']:.4f}s  per_trial: {res['per_trial_ms']:.2f} ms")
        es = res["engine_summary"]
        for k in ["observations","best_model_index","best_model_name","mix_avg_logscore","pit_score","stability_score","advantage_score","confidence_index"]:
            if k in es: print(f"{k}: {es[k]}")
        print(f"earned_cc: {res['earned_cc']:.6f}  (eps={args.eps})")
        if res["examples"]:
            print("\n--- example successes (first few) ---")
            for ex in res["examples"]:
                print(ex)
        print(f"\nmods_applied: {res['mods_applied']}")
        print("\nNotes: BOTH subsumes undecidable/unknown; only bounded checks may yield TRUE/FALSE; "
              "confidence bumps occur solely on agreement (pmf≥τ, PIT in band, CI≥τ). "
              "Replace DemoCountAdapter with real adapters for your tasks.")
    finally:
        oracle.close()

# Simple helper to avoid importing asyncio for one-liners
def asyncio_run(coro):
    try:
        import asyncio
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

if __name__ == "__main__":
    main()
