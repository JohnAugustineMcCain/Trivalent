# PPC — Perspectivistic Paraconsistent Contextualism

## Attribution & Citation

This work is © John A. McCain and licensed for non-commercial use under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

When referencing, adapting, or building upon this work, please cite as:

John A. McCain. *PPC+ — Perspectivistic Paraconsistent Contextualism: LP-first, Safety-first Reasoning Engine.*

> **TL;DR**: A production-ready reasoning engine that **actually implements a modified idea of Priest’s LP** (paraconsistent logic) and uses it to make LLMs more **contradiction-resilient, reliability-aware, and safely self-modifying**. Default prior is **BOTH** (dialetheic), with TRUE/FALSE emerging only via perspectival/contextual collapse.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CCBYNC-blue.svg)](#)
[![LP](https://img.shields.io/badge/logic-Priest's%20LP-purple)](#)
[![Safety](https://img.shields.io/badge/focus-AI%20Safety-orange)](#)

---

## Why This Could Be Revolutionary

### 1) Actually Implements Concepts from Priest’s LP in a New Way
**Working paraconsistent semantics** with proper truth pairs `(t,f)` and truth-functional connectives:
- `AND`: `(min t, max f)`  
- `OR`: `(max t, min f)`  
- `NOT`: `(f, t)`  
- `IMPLIES`: `¬A ∨ B`

### 2) Production-Ready Architecture
- **Adapter pattern** for LLM integration (multiple perspectives, prompts, or models)
- **Reliability tracking** via rolling Brier scores → sources self-calibrate
- **Safe parameter updates** with tests + rollback (`SafeUpdater`)
- **Comprehensive provenance** (model/meta/prompt/context hashing)

### 3) Genuine AI Safety Innovation
- **Contradiction resilience** (LP foundation, BOTH as default prior)
- **Self-modification safety** (guarded tunables, revert on failed tests)
- **Graceful degradation** (never “NEITHER”; fall back to **BOTH**)
- **Reliability learning** (bounded weights, down-weight poor sources)

### 4) Based on functional conclusions about P vs NP (see P vs NP as Epistemic Illusion)
**Bottom line:** This moves from what was once an “interesting idea” to a **deployable AI safety architecture** grounded in paraconsistent logic and practical engineering.

---

## Requirements
- Python **3.7+** (recommended **3.10+**)
- No runtime dependencies (standard library only)

### Optional (dev)
- pytest, ruff, black, mypy, hypothesis

## Core Idea (Meta-Dialetheism)
Reality presents by default as jointly true and/or false. This engine **preserves contradiction** as a first-class feature (**BOTH**), and only collapses to classical bivalence (TRUE or FALSE) with sufficient **perspectival/contextual** warrant (high consensus + high confidence).

---

## Install
```bash
pip install -e .
# or just vendor `reasoning_engine.py` into your project
