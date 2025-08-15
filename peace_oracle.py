from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable

from peace import (
    TV, Formula, Var, Not, And, Or, Implies, TruePred, FalseStar, SelfRef,
    Valuation, Perspective, PerspectiveResult, Context, select_verdict, pretty
)

@dataclass
class OracleAnswer:
    value: TV                 # T/F/B
    justification: Dict[str, Dict[str, Any]]  # per-perspective reasons

class PeaceOracle:
    """
    Thin adapter that turns PEACE into an oracle usable by solvers.
    You pass in a Context and a portfolio of Perspectives.
    Query with a 'claim' (any Formula you define, e.g., 'Graph has treewidth <= k').
    """
    def __init__(self, context: Context, perspectives: List[Perspective], set_valued: bool=False):
        self.context = context
        self.perspectives = perspectives
        self.set_valued = set_valued

    def query(self, claim: Formula, valuation: Optional[Valuation]=None) -> OracleAnswer:
        v, details = select_verdict(
            claim,
            valuation or {},
            self.context,
            self.perspectives,
            set_valued=self.set_valued
        )
        just = {
            name: {
                "value": pretty(res.value),
                "admissible": res.admissible,
                "load_bearing": res.load_bearing,
                "reason": res.reason
            }
            for name, res in details.items()
        }
        return OracleAnswer(v, just)

# ----------- Example claim shapes you can reuse ------------

# “There exists a ranking function r: State -> N that strictly decreases per loop”
@dataclass(frozen=True)
class ExistsRanking(Formula):
    code_snippet: str
    loop_label: Optional[str] = None

# “Graph G has treewidth <= k”
@dataclass(frozen=True)
class HasTreewidthAtMost(Formula):
    # you provide encoding in 'G' (adj list/edge list) and target k
    G: Any
    k: int
