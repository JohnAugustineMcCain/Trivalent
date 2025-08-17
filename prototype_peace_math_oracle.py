import ast
import sqlite3
import logging
import time
import hashlib
from enum import Enum
from typing import Callable, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC
import threading  # Tier A: simple lock for sqlite access

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
# Truth values
# -------------------------
class TV(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2      # Paraconsistent both true and false
    UNKNOWN = 3   # Insufficient information

# -------------------------
# Core dataclasses
# -------------------------
@dataclass(frozen=True)
class MathematicalProblem:
    name: str
    description: str
    complexity_score: int  # 1-10 scale
    computational_bound: int  # Maximum feasible direct computation
    problem_type: str  # "number_theory", "combinatorics", "analysis", etc.
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
    modification_type: str  # "add", "modify", "delete"
    safety_score: float
    mathematical_soundness: float
    reasoning: str
    timestamp: float

    def __hash__(self) -> int:
        return hash(self.modification_id)

# -------------------------
# PEACE Perspective
# -------------------------
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

# -------------------------
# Versioned cache (SQLite)
# -------------------------
class VersionedPEACECache:
    def __init__(self, db_path: str = "peace_cache.db"):
        self.cache: Dict[str, List[Tuple[TV, float, str]]] = defaultdict(list)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()  # Tier A: guard concurrent access
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
        statement_str = str(statement)
        statement_hash = hashlib.sha256(statement_str.encode()).hexdigest()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO evaluations 
                (statement_hash, statement, verdict, confidence, perspective, version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (statement_hash, statement_str, verdict.name, confidence, perspective, version, time.time()),
            )
            self.conn.commit()

    def record_modification(self, modification: CodeModification):
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO code_modifications
                (modification_id, target_module, target_function, original_code, modified_code,
                 safety_verdict, safety_confidence, mathematical_soundness, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    modification.modification_id,
                    modification.target_module,
                    modification.target_function,
                    modification.original_code,
                    modification.modified_code,
                    "PENDING",                 # unchanged: placeholder verdict
                    modification.safety_score, # stored as safety_confidence (original design)
                    modification.mathematical_soundness,
                    modification.reasoning,
                    modification.timestamp,
                ),
            )
            self.conn.commit()

    def get_modification_history(self, target_function: str) -> List[CodeModification]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT modification_id, target_module, target_function, original_code, modified_code,
                       safety_verdict, safety_confidence, mathematical_soundness, reasoning, timestamp
                FROM code_modifications 
                WHERE target_function = ? 
                ORDER BY timestamp DESC
                """,
                (target_function,),
            )
            rows = cursor.fetchall()

        modifications: List[CodeModification] = []
        for row in rows:
            # Map columns correctly to dataclass fields (Tier A mapping fix)
            mod = CodeModification(
                modification_id=row[0],
                target_module=row[1],
                target_function=row[2],
                original_code=row[3],
                modified_code=row[4],
                modification_type="modify",                     # table has no column; keep your default
                safety_score=row[6] if row[6] is not None else 0.0,  # safety_confidence -> safety_score
                mathematical_soundness=row[7] if row[7] is not None else 0.0,
                reasoning=row[8] or "",
                timestamp=row[9] if row[9] is not None else time.time(),
            )
            modifications.append(mod)
        return modifications

    def close(self):
        with self._lock:
            self.conn.close()

# -------------------------
# LLM Interface (mock)
# -------------------------
class LLMInterface:
    """Interface for Large Language Model interactions (mocked)."""

    def __init__(self):
        self.conversation_history = []

    async def analyze_mathematical_problem(self, problem: MathematicalProblem) -> Dict[str, Any]:
        logger.info(f"Analyzing mathematical problem: {problem.name}")
        if problem.problem_type == "number_theory":
            return {
                "structure": "number_theoretic_conjecture",
                "approaches": ["direct_verification", "asymptotic_analysis", "probabilistic_methods"],
                "computational_feasibility": problem.computational_bound,
                "algorithmic_strategies": ["sieve_based", "oracle_guided_leaps", "pattern_recognition"],
                "meta_logical_potential": 0.8,
                "complexity_analysis": {
                    "computational_complexity": f"O(n^{problem.complexity_score/5})",
                    "verification_difficulty": "super_linear" if problem.complexity_score > 6 else "polynomial",
                },
            }
        elif problem.problem_type == "analysis":
            return {
                "structure": "analytical_conjecture",
                "approaches": ["functional_analysis", "measure_theory", "complex_analysis"],
                "computational_feasibility": problem.computational_bound // 10,
                "algorithmic_strategies": ["numerical_methods", "symbolic_computation"],
                "meta_logical_potential": 0.6,
            }
        else:
            return {
                "structure": "general_mathematical_problem",
                "approaches": ["computational_search", "theoretical_analysis"],
                "computational_feasibility": problem.computational_bound,
                "algorithmic_strategies": ["exhaustive_search", "heuristic_methods"],
                "meta_logical_potential": 0.5,
            }

    async def suggest_solution_method(self, problem: MathematicalProblem, analysis: Dict) -> Dict[str, Any]:
        logger.info(f"Suggesting solution method for: {problem.name}")
        if problem.complexity_score >= 8:
            return {
                "primary_method": "oracle_guided_verification",
                "secondary_methods": ["pattern_learning", "asymptotic_extrapolation"],
                "confidence": 0.85,
                "requires_code_modification": True,
                "modification_targets": ["verification_engine", "pattern_learner", "oracle_evaluator"],
                "reasoning": "High complexity requires advanced oracle capabilities",
            }
        elif problem.complexity_score >= 5:
            return {
                "primary_method": "enhanced_verification",
                "secondary_methods": ["direct_computation", "heuristic_analysis"],
                "confidence": 0.9,
                "requires_code_modification": True,
                "modification_targets": ["verification_engine"],
                "reasoning": "Medium complexity benefits from enhanced verification",
            }
        else:
            return {
                "primary_method": "direct_computation",
                "secondary_methods": ["exhaustive_search"],
                "confidence": 0.95,
                "requires_code_modification": False,
                "reasoning": "Low complexity suitable for direct computation",
            }

    async def analyze_current_code(self, module_name: str, function_name: str) -> Dict[str, Any]:
        try:
            return {
                "current_capabilities": ["basic_verification", "simple_patterns", "direct_computation"],
                "limitations": ["no_large_scale_leaping", "limited_heuristics", "no_asymptotic_analysis"],
                "improvement_potential": 0.9,
                "safety_concerns": ["infinite_loops", "memory_overflow", "stack_overflow"],
                "lines_of_code": 150,
                "complexity_score": 6,
            }
        except Exception as e:
            logger.error(f"Failed to analyze code {module_name}.{function_name}: {e}")
            return {"error": str(e), "capabilities": "unknown"}

    # -------- Tier A: stable, deterministic IDs for modifications --------
    def _stable_id(self, *parts: str, prefix: str = "mod_", n: int = 12) -> str:
        h = hashlib.sha256("||".join(parts).encode()).hexdigest()
        return f"{prefix}{h[:n]}"

    async def propose_code_modifications(
        self, problem: MathematicalProblem, current_analysis: Dict, solution_method: Dict
    ) -> List[CodeModification]:
        modifications: List[CodeModification] = []
        if solution_method.get("requires_code_modification"):
            logger.info(f"Proposing code modifications for {problem.name}")

            if problem.problem_type == "number_theory" and problem.complexity_score >= 7:
                mod = CodeModification(
                    modification_id=self._stable_id(problem.name, "oracle_enhancement"),
                    target_module="peace_oracle",
                    target_function="evaluate_large_scale",
                    original_code=(
                        "def evaluate_large_scale(self, n):\n"
                        "    # Basic implementation\n"
                        "    if n <= self.computational_bound:\n"
                        "        return self.direct_verify(n)\n"
                        "    return None\n"
                    ),
                    modified_code=(
                        "def evaluate_large_scale(self, n):\n"
                        "    # Enhanced with Hardy-Littlewood heuristics and oracle guidance\n"
                        "    if n <= self.computational_bound:\n"
                        "        return self.direct_verify(n)\n"
                        "    elif n <= self.computational_bound * 1000:\n"
                        "        return self.oracle_guided_verification(n)\n"
                        "    else:\n"
                        "        return self.hardy_littlewood_analysis(n)\n"
                    ),
                    modification_type="modify",
                    safety_score=0.0,
                    mathematical_soundness=0.0,
                    reasoning="Add asymptotic analysis and oracle guidance for large numbers beyond computational reach",
                    timestamp=time.time(),
                )
                modifications.append(mod)

            if problem.complexity_score >= 6:
                pattern_mod = CodeModification(
                    modification_id=self._stable_id(problem.name, "pattern_learning"),
                    target_module="peace_oracle",
                    target_function="learn_patterns",
                    original_code=(
                        "def learn_patterns(self, verified_cases):\n"
                        "    # Basic pattern storage\n"
                        "    self.pattern_cache = verified_cases\n"
                    ),
                    modified_code=(
                        "def learn_patterns(self, verified_cases):\n"
                        "    # Advanced pattern recognition with statistical analysis\n"
                        "    self.pattern_cache = verified_cases\n"
                        "    # Placeholder: PatternAnalyzer/ConfidenceEstimator would be defined elsewhere\n"
                        "    # self.pattern_analyzer = PatternAnalyzer()\n"
                        "    # self.learned_heuristics = self.pattern_analyzer.extract_heuristics(verified_cases)\n"
                        "    # self.confidence_estimator = ConfidenceEstimator(self.learned_heuristics)\n"
                    ),
                    modification_type="modify",
                    safety_score=0.0,
                    mathematical_soundness=0.0,
                    reasoning="Enhance pattern learning with advanced statistical analysis",
                    timestamp=time.time(),
                )
                modifications.append(pattern_mod)

        logger.info(f"Proposed {len(modifications)} modifications")
        return modifications

# -------------------------
# Safety perspectives
# -------------------------
class CodeSafetyPerspective(PEACEPerspective):
    """Evaluates safety of code modifications"""

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
        safe_count = sum(1 for x in checks if x)
        total_checks = len(checks)

        if safe_count == total_checks:
            return TV.TRUE
        elif safe_count == 0:
            return TV.FALSE
        elif safe_count >= int(total_checks * 0.7):
            return TV.BOTH
        else:
            return TV.FALSE

    def _compute_safety_confidence(self, modification: CodeModification) -> float:
        try:
            tree = ast.parse(modification.modified_code)
            complexity = len(list(ast.walk(tree)))
            base_confidence = max(0.1, 1.0 - (complexity / 100.0))
            safe_patterns = ["if ", "return", "self.", "try:", "except:"]
            pattern_boost = sum(0.05 for pattern in safe_patterns if pattern in modification.modified_code)
            return min(0.95, base_confidence + pattern_boost)
        except SyntaxError:
            return 0.1
        except Exception:
            return 0.3

    def _check_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _check_infinite_loops(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                        if not has_break:
                            return False
                elif isinstance(node, ast.For):
                    if isinstance(node.iter, ast.Call):
                        if not (isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
                            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                            if not has_break:
                                return False
            return True
        except Exception:
            return False

    def _check_memory_safety(self, code: str) -> bool:
        dangerous_patterns = ["exec(", "eval(", "globals()", "__import__", "setattr(", "delattr(", "vars()", "locals()"]
        return not any(p in code for p in dangerous_patterns)

    def _check_side_effects(self, code: str) -> bool:
        side_effect_patterns = ["os.", "sys.", "subprocess.", "open(", "file(", "input(", "write(", "delete"]
        safe_exceptions = ["print("]  # allow prints as pseudo-logging
        for pattern in side_effect_patterns:
            if pattern in code and pattern not in safe_exceptions:
                return False
        return True

    def _check_imports(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    dangerous_modules = ["os", "sys", "subprocess", "pickle", "marshal"]
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in dangerous_modules:
                                return False
                    elif isinstance(node, ast.ImportFrom) and node.module in dangerous_modules:
                        return False
            return True
        except Exception:
            return False

class MathematicalSoundnessPerspective(PEACEPerspective):
    """Evaluates mathematical correctness of modifications (lightweight heuristics)"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        super().__init__(
            name="mathematical_soundness",
            evaluate_fn=self._evaluate_soundness,
            confidence_fn=self._compute_soundness_confidence,
        )

    def _evaluate_soundness(self, modification: CodeModification) -> TV:
        code_lower = modification.modified_code.lower()

        established_methods = [
            "hardy_littlewood", "asymptotic", "prime", "sieve",
            "theorem", "conjecture", "logarithm", "probability"
        ]
        rigor_indicators = [
            "bound", "limit", "convergence", "error", "precision",
            "confidence", "estimate", "approximation"
        ]
        dangerous_claims = [
            " prove ", " proof ", " qed", "therefore proven",
            "definitively true", "absolutely certain"
        ]

        method_score = sum(1 for m in established_methods if m in code_lower)
        rigor_score = sum(1 for r in rigor_indicators if r in code_lower)
        danger_score = sum(1 for d in dangerous_claims if d in code_lower)

        if danger_score > 0:
            return TV.FALSE
        elif method_score >= 2 and rigor_score >= 1:
            return TV.TRUE
        elif method_score >= 1 or rigor_score >= 1:
            return TV.BOTH
        else:
            return TV.FALSE

    def _compute_soundness_confidence(self, modification: CodeModification) -> float:
        code_lower = modification.modified_code.lower()
        math_keywords = [
            "theorem", "conjecture", "prime", "asymptotic",
            "logarithm", "analysis", "heuristic", "probability", "distribution"
        ]
        keyword_count = sum(1 for k in math_keywords if k in code_lower)
        base_conf = min(0.9, 0.3 + (keyword_count * 0.1))
        if "hardy_littlewood" in code_lower:
            base_conf += 0.2
        if "asymptotic" in code_lower and "analysis" in code_lower:
            base_conf += 0.15
        if "bound" in code_lower or "limit" in code_lower:
            base_conf += 0.1
        return min(0.95, base_conf)

class CategoryErrorPerspective(PEACEPerspective):
    """Detects category errors in problem formulation or solution approach"""

    def __init__(self):
        super().__init__(
            name="category_error",
            evaluate_fn=self._evaluate_category_error,
            confidence_fn=self._compute_category_confidence,
        )

    def _evaluate_category_error(self, modification: CodeModification) -> TV:
        code_lower = modification.modified_code.lower()

        impossible_patterns = [
            ("infinite", "verify"),
            ("all", "numbers"),
            ("every", "integer"),
            ("prove", "conjecture"),
        ]
        boundary_patterns = [
            "bound", "limit", "threshold", "if n >", "computational",
            "feasible", "approximation", "heuristic",
        ]
        meta_patterns = [
            "confidence", "certainty", "probability", "estimate",
            "likely", "suggests", "indicates",
        ]

        has_impossible = any(all(word in code_lower for word in pat) for pat in impossible_patterns)
        has_boundaries = any(p in code_lower for p in boundary_patterns)
        has_meta = any(p in code_lower for p in meta_patterns)

        if has_impossible and not has_boundaries:
            return TV.FALSE
        elif has_boundaries and has_meta:
            return TV.TRUE
        elif has_boundaries or has_meta:
            return TV.BOTH
        else:
            return TV.BOTH

    def _compute_category_confidence(self, modification: CodeModification) -> float:
        return 0.7

# -------------------------
# Self-modifying Oracle
# -------------------------
class SelfModifyingPEACEOracle:
    """Main self-modifying PEACE Oracle class"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.cache = VersionedPEACECache()
        self.safety_perspectives = [
            CodeSafetyPerspective(),
            MathematicalSoundnessPerspective(llm_interface),
            CategoryErrorPerspective(),
        ]
        self.mathematical_perspectives: Dict[str, PEACEPerspective] = {}
        self.modification_queue: List[CodeModification] = []
        self.active_modifications: Dict[str, CodeModification] = {}
        logger.info("Initialized Self-Modifying PEACE Oracle")

    async def solve_mathematical_problem(self, problem: MathematicalProblem) -> Dict[str, Any]:
        logger.info(f"Starting analysis of problem: {problem.name}")
        try:
            analysis = await self.llm.analyze_mathematical_problem(problem)
            solution_method = await self.llm.suggest_solution_method(problem, analysis)

            current_code_analysis: Dict[str, Any] = {}
            if solution_method.get("requires_code_modification"):
                for target in solution_method.get("modification_targets", []):
                    module_analysis = await self.llm.analyze_current_code("peace_oracle", target)
                    current_code_analysis[target] = module_analysis

            proposed_mods = await self.llm.propose_code_modifications(problem, current_code_analysis, solution_method)

            safe_mods: List[CodeModification] = []
            for mod in proposed_mods:
                safety_result = await self._evaluate_modification_safety(mod)
                if (safety_result["verdict"] in [TV.TRUE, TV.BOTH]) and (safety_result["confidence"] > 0.6):
                    safe_mods.append(mod)
                    logger.info(f"Approved modification: {mod.modification_id}")
                else:
                    logger.warning(f"Rejected unsafe modification: {mod.modification_id}")

            if safe_mods:
                await self._apply_modifications(safe_mods)
                logger.info(f"Applied {len(safe_mods)} modifications")

            solution_result = await self._attempt_problem_solution(problem, solution_method)

            if not solution_result.get("algorithmic_solution"):
                meta_result = await self._meta_logical_analysis(problem, analysis)
                solution_result.update(meta_result)

            return solution_result

        except Exception as e:
            logger.error(f"Error solving problem {problem.name}: {e}")
            return {"error": str(e), "algorithmic_solution": False, "meta_logical_analysis": False}

    async def _evaluate_modification_safety(self, modification: CodeModification) -> Dict[str, Any]:
        verdicts: Dict[str, TV] = {}
        confidences: Dict[str, float] = {}

        for perspective in self.safety_perspectives:
            verdict, confidence = perspective.evaluate(modification)
            verdicts[perspective.name] = verdict
            confidences[perspective.name] = confidence
            self.cache.record_evaluation(modification, verdict, confidence, perspective.name)

        integrated = self._integrate_safety_verdicts(verdicts, confidences)
        return {
            "verdict": integrated["verdict"],
            "confidence": integrated["confidence"],
            "individual_verdicts": verdicts,
            "individual_confidences": confidences,
            "score_breakdown": integrated.get("score_breakdown", {}),
        }

    def _integrate_safety_verdicts(self, verdicts: Dict[str, TV], confidences: Dict[str, float]) -> Dict[str, Any]:
        weighted_scores = {"TRUE": 0.0, "FALSE": 0.0, "BOTH": 0.0, "UNKNOWN": 0.0}
        total_weight = 0.0
        for perspective, verdict in verdicts.items():
            weight = confidences.get(perspective, 0.0)
            weighted_scores[verdict.name] += weight
            total_weight += weight

        if total_weight == 0:
            return {"verdict": TV.BOTH, "confidence": 0.0}

        for k in weighted_scores:
            weighted_scores[k] /= total_weight

        if weighted_scores["FALSE"] > 0.3:
            integrated_verdict = TV.FALSE
        elif weighted_scores["TRUE"] > 0.7:
            integrated_verdict = TV.TRUE
        else:
            integrated_verdict = TV.BOTH

        integrated_confidence = max(weighted_scores.values())
        return {"verdict": integrated_verdict, "confidence": integrated_confidence, "score_breakdown": weighted_scores}

    async def _apply_modifications(self, modifications: List[CodeModification]):
        for modification in modifications:
            try:
                self.cache.record_modification(modification)
                self.active_modifications[modification.modification_id] = modification
                logger.info(f"Applied modification {modification.modification_id}")
            except Exception as e:
                logger.error(f"Failed to apply modification {modification.modification_id}: {e}")

    async def _attempt_problem_solution(self, problem: MathematicalProblem, method: Dict[str, Any]) -> Dict[str, Any]:
        if problem.verification_function and method.get("primary_method") == "direct_computation":
            try:
                # Example: run verification for a small bound
                result = problem.verification_function(1000)
                return {
                    "algorithmic_solution": True,
                    "method": "direct_computation",
                    "result": result,
                    "confidence": 1.0,
                }
            except Exception as e:
                logger.error(f"Direct verification failed: {e}")

        if self.active_modifications:
            return {
                "algorithmic_solution": True,
                "method": "enhanced_verification",
                "result": "Enhanced verification capabilities active",
                "modifications_applied": len(self.active_modifications),
                "confidence": 0.8,
            }

        return {
            "algorithmic_solution": False,
            "method": "basic_analysis",
            "result": "Insufficient computational resources for direct solution",
            "confidence": 0.3,
        }

    async def _meta_logical_analysis(self, problem: MathematicalProblem, analysis: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Performing meta-logical analysis for {problem.name}")

        if problem.name not in self.mathematical_perspectives:
            self.mathematical_perspectives[problem.name] = self._create_problem_perspective(problem, analysis)
        perspective = self.mathematical_perspectives[problem.name]

        test_statements = self._generate_test_statements(problem, analysis)

        results: Dict[str, Dict[str, Any]] = {}
        for statement in test_statements:
            verdict, confidence = perspective.evaluate(statement)
            results[statement] = {"verdict": verdict, "confidence": confidence}
            self.cache.record_evaluation(statement, verdict, confidence, problem.name)

        integrated = self._integrate_meta_logical_results(results, problem, analysis)
        return {
            "meta_logical_analysis": True,
            "problem_perspective": problem.name,
            "statements_evaluated": len(test_statements),
            "integrated_verdict": integrated["verdict"],
            "confidence": integrated["confidence"],
            "reasoning": integrated["reasoning"],
            "detailed_results": results,
        }

    def _create_problem_perspective(self, problem: MathematicalProblem, analysis: Dict[str, Any]) -> PEACEPerspective:
        def evaluate_fn(statement: Any) -> TV:
            s = str(statement).lower()
            if problem.problem_type == "number_theory":
                if "prime" in s:
                    if "infinite" in s and "gap" not in s:
                        return TV.TRUE
                    elif "distribution" in s:
                        return TV.BOTH
                    else:
                        return TV.UNKNOWN
                elif "conjecture" in s:
                    return TV.BOTH
                else:
                    return TV.UNKNOWN
            elif problem.problem_type == "analysis":
                if "convergent" in s or "bounded" in s:
                    return TV.BOTH
                elif "continuous" in s:
                    return TV.TRUE
                else:
                    return TV.UNKNOWN
            else:
                if "finite" in s:
                    return TV.TRUE
                elif "infinite" in s:
                    return TV.BOTH
                else:
                    return TV.UNKNOWN

        def confidence_fn(statement: Any) -> float:
            base_conf = 0.5
            complexity_factor = (10 - problem.complexity_score) / 10.0
            base_conf *= complexity_factor
            s = str(statement).lower()
            if problem.problem_type == "number_theory":
                base_conf += 0.2
            if "infinite" in s or "all" in s:
                base_conf *= 0.7
            return min(0.9, max(0.1, base_conf))

        return PEACEPerspective(
            name=f"problem_{problem.name}",
            evaluate_fn=evaluate_fn,
            confidence_fn=confidence_fn,
        )

    def _generate_test_statements(self, problem: MathematicalProblem, analysis: Dict[str, Any]) -> List[str]:
        statements = [
            f"The {problem.name} has a finite solution",
            f"The {problem.name} can be solved computationally",
            f"The {problem.name} requires infinite computation",
        ]
        if problem.problem_type == "number_theory":
            statements.extend([
                "Prime numbers are infinite",
                "Prime gaps are bounded",
                "There exists a pattern in prime distribution",
                "Asymptotic methods apply to this problem",
                "Hardy-Littlewood heuristics are relevant",
            ])
        elif problem.problem_type == "analysis":
            statements.extend([
                "The function has bounded variation",
                "Convergence is uniform",
                "The series converges absolutely",
                "Measure theory applies to this problem",
            ])
        elif problem.problem_type == "combinatorics":
            statements.extend([
                "The counting problem has closed form",
                "Generating functions apply",
                "The asymptotic growth is polynomial",
                "Probabilistic methods are effective",
            ])

        if problem.complexity_score >= 7:
            statements.extend([
                "Direct computation is infeasible",
                "Heuristic methods are necessary",
                "The problem admits approximation algorithms",
                "Meta-logical reasoning provides insight",
            ])
        if analysis.get("meta_logical_potential", 0) > 0.7:
            statements.extend([
                "Oracle guidance improves solution quality",
                "Pattern learning enhances verification",
                "Self-modification provides computational advantages",
            ])
        return statements

    def _integrate_meta_logical_results(
        self, results: Dict[str, Dict[str, Any]], problem: MathematicalProblem, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        verdict_scores = {"TRUE": 0.0, "FALSE": 0.0, "BOTH": 0.0, "UNKNOWN": 0.0}
        total_weight = 0.0
        for _, res in results.items():
            vname = res["verdict"].name
            c = res["confidence"]
            verdict_scores[vname] += c
            total_weight += c

        if total_weight > 0:
            for k in verdict_scores:
                verdict_scores[k] /= total_weight

        max_score = max(verdict_scores.values()) if verdict_scores else 0.0
        overall_verdict = None
        for vn, sc in verdict_scores.items():
            if sc == max_score:
                overall_verdict = TV[vn]
                break

        reasoning_parts: List[str] = []
        if verdict_scores["TRUE"] > 0.4:
            reasoning_parts.append("Strong evidence supports computational tractability")
        if verdict_scores["FALSE"] > 0.3:
            reasoning_parts.append("Significant barriers to direct solution exist")
        if verdict_scores["BOTH"] > 0.4:
            reasoning_parts.append("Problem exhibits paraconsistent aspects requiring careful analysis")
        if verdict_scores["UNKNOWN"] > 0.5:
            reasoning_parts.append("Insufficient information for definitive assessment")
        if problem.complexity_score >= 8:
            reasoning_parts.append("High complexity suggests meta-logical approaches are valuable")
        if analysis.get("meta_logical_potential", 0) > 0.7:
            reasoning_parts.append("Problem structure is amenable to oracle-guided analysis")

        return {
            "verdict": overall_verdict or TV.UNKNOWN,
            "confidence": max_score,
            "reasoning": ". ".join(reasoning_parts) if reasoning_parts else "Analysis inconclusive",
            "verdict_distribution": verdict_scores,
        }

    async def get_modification_history(self, function_name: Optional[str] = None) -> List[CodeModification]:
        if function_name:
            return self.cache.get_modification_history(function_name)
        return list(self.active_modifications.values())

    async def analyze_self_improvement_potential(self) -> Dict[str, Any]:
        modification_count = len(self.active_modifications)
        capability_enhancement = modification_count * 0.1
        solved_problems = len(self.mathematical_perspectives)
        learning_factor = min(1.0, solved_problems * 0.05)
        safety_score = 1.0
        improvement_potential = min(1.0, capability_enhancement + learning_factor) * safety_score
        return {
            "current_modifications": modification_count,
            "capability_enhancement": capability_enhancement,
            "learning_factor": learning_factor,
            "safety_score": safety_score,
            "improvement_potential": improvement_potential,
            "recommended_focus": self._recommend_improvement_focus(),
        }

    def _recommend_improvement_focus(self) -> List[str]:
        recommendations: List[str] = []
        if len(self.active_modifications) < 3:
            recommendations.append("Enhance computational capabilities")
        if len(self.mathematical_perspectives) < 5:
            recommendations.append("Develop more specialized mathematical perspectives")
        if not any("pattern" in m.reasoning for m in self.active_modifications.values()):
            recommendations.append("Implement advanced pattern recognition")
        if not any("asymptotic" in m.reasoning for m in self.active_modifications.values()):
            recommendations.append("Add asymptotic analysis capabilities")
        recommendations.append("Strengthen meta-logical reasoning frameworks")
        return recommendations

    def close(self):
        self.cache.close()
        logger.info("PEACE Oracle shutdown complete")

# -------------------------
# Example verification functions (safe & fast)
# -------------------------
def count_twin_primes_upto(n: int) -> int:
    """Naive but clear twin-prime counter up to n (odd-only trial division)."""
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0:
                return False
            i += 2
        return True

    count = 0
    p = 3
    while p + 2 <= n:
        if is_prime(p) and is_prime(p + 2):
            count += 1
        p += 2
    return count

def goldbach_all_even_upto(limit_n: int) -> bool:
    """Check Goldbach for all even numbers up to limit_n (small)."""
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0:
                return False
            i += 2
        return True

    for n in range(4, min(limit_n, 10000) + 1, 2):
        found = False
        for a in range(2, n // 2 + 1):
            if is_prime(a) and is_prime(n - a):
                found = True
                break
        if not found:
            return False
    return True

# -------------------------
# Example problems
# -------------------------
TWIN_PRIME_CONJECTURE = MathematicalProblem(
    name="twin_prime_conjecture",
    description="There are infinitely many twin primes (primes p such that p+2 is also prime)",
    complexity_score=9,
    computational_bound=10**6,
    problem_type="number_theory",
    verification_function=count_twin_primes_upto
)

GOLDBACH_CONJECTURE = MathematicalProblem(
    name="goldbach_conjecture",
    description="Every even integer greater than 2 can be expressed as sum of two primes",
    complexity_score=8,
    computational_bound=10**6,
    problem_type="number_theory",
    verification_function=lambda limit: goldbach_all_even_upto(limit)
)

# -------------------------
# Main
# -------------------------
async def main():
    llm = LLMInterface()
    oracle = SelfModifyingPEACEOracle(llm)
    try:
        logger.info("=== Self-Modifying PEACE Oracle Demonstration ===")

        logger.info("\n--- Analyzing Twin Prime Conjecture ---")
        twin_prime_result = await oracle.solve_mathematical_problem(TWIN_PRIME_CONJECTURE)
        print("\nTwin Prime Analysis Result:")
        print(f"Algorithmic Solution: {twin_prime_result.get('algorithmic_solution', False)}")
        print(f"Meta-logical Analysis: {twin_prime_result.get('meta_logical_analysis', False)}")
        if twin_prime_result.get("integrated_verdict"):
            print(f"Integrated Verdict: {twin_prime_result['integrated_verdict']}")
            print(f"Confidence: {twin_prime_result.get('confidence', 0):.3f}")
            print(f"Reasoning: {twin_prime_result.get('reasoning', 'N/A')}")

        logger.info("\n--- Analyzing Goldbach Conjecture ---")
        goldbach_result = await oracle.solve_mathematical_problem(GOLDBACH_CONJECTURE)
        print("\nGoldbach Analysis Result:")
        print(f"Algorithmic Solution: {goldbach_result.get('algorithmic_solution', False)}")
        print(f"Meta-logical Analysis: {goldbach_result.get('meta_logical_analysis', False)}")
        if goldbach_result.get("integrated_verdict"):
            print(f"Integrated Verdict: {goldbach_result['integrated_verdict']}")
            print(f"Confidence: {goldbach_result.get('confidence', 0):.3f}")
            print(f"Reasoning: {goldbach_result.get('reasoning', 'N/A')}")

        logger.info("\n--- Self-Improvement Analysis ---")
        improvement = await oracle.analyze_self_improvement_potential()
        print("\nSelf-Improvement Analysis:")
        print(f"Current Modifications: {improvement['current_modifications']}")
        print(f"Improvement Potential: {improvement['improvement_potential']:.3f}")
        print(f"Recommended Focus: {', '.join(improvement['recommended_focus'])}")

        modifications = await oracle.get_modification_history()
        if modifications:
            print(f"\nActive Modifications: {len(modifications)}")
            for mod in modifications:
                print(f"  - {mod.modification_id}: {mod.reasoning}")

    finally:
        oracle.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
