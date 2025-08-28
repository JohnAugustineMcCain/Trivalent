This work is © John A. McCain and licensed for non-commercial use under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

**Disclaimer:**

*This repository is **NOT about proving my ideas about P vs NP** and the Goldbach Conjecture and everything else the ideas can be applied to.*

*Studying P vs NP with a new perspective is just what led me to develop my ideas and logical system, one I think might change our methods of designing safe, contradiction-resilient AI.*

## Preface:

My goal here is to prove that honesty (not proof) is universally mathematically optimal in the face of increasing complexity for logic and reasoning.

I developed the foundations of this over the course of *two weeks* while working in my profession: Convenience store employee

It is my humble belief that combination of these ideas and their historical roots + working code + practical applications makes this work potentially groundbreaking.​​​​​​​​​​​​​​​​

**For Context of why I am working on this:**
📄 [My final work (so far)](P_vs_NP__Proven_Unprovable.pdf)

📄 [And / Or the summary and defense of it](https://github.com/<username>/<repo>/blob/main/The_Verification_Asymmetry.pdf)

## PEACE: Paraconsistent Epistemic And Contextual Evaluation

This is the **first working PEACE engine** (Paraconsistent Epistemic And Contextual Evaluation).

While it seems like it draws parallels to the ideas of Monte Carlo sampling (so I'm told), the design and epistemic framing are actually completely original, having been created by scratch (with no knowledge whatsoever about such methods).

It shows how paraconsistent updates and bounded exploration can accumulate belief without requiring exhaustive proofs.  

📄 [Goldbach Proof of Concept](https://github.com/JohnAugustineMcCain/Trivalent/blob/main/goldbach_meta.py)

**Goldbach Tiny-Budget Sampler**

A lightweight PEACE engine for probing the **Goldbach Conjecture** at huge scales.

Instead of exhaustive checking, it uses a tiny budget (small subtractors, limited offsets, fast Miller–Rabin checks) to ratchet confidence that even numbers decompose into primes.

Because of this unique search method, we are able to effectively evaluate computationally absurd numbers, identifying multiple successful decompositions that asymptotically increase in frequency as that number grows. This serves as cogent evidence that confidence in Hardy-littlewood only builds as numbers get larger.

In other words, this constitutes the first empirical bridge across the vast
gap between finite verification and asymptotic prediction, and provides
meta-mathematical evidence that the Goldbach Conjecture is effectively
resolved.

[📄 Empirical Evidence for Hardy–Littlewood Beyond Classical Verification](Empirical_Evidence_for_Hardy_Littlewood_Beyond_Classical_Verification.pdf)

**Usage:**

Run with Python 3. Adjustable samples provided below.

```bash
# 500 runs, 24-digit n, 20 subtractors
python3 goldbach_proof_of_concept.py --trials 500 --digits 24 --subtractors 20
```
```bash
# 10k runs, 50-digit n, 100 subtractors
python3 goldbach_proof_of_concept.py --trials 10000 --digits 50 --subtractors 100
```

## Why this is interesting

-	Mathematicians normally stop at Level 1: state the heuristic and assume it’s “convincing enough.”
  
- I've built Level 2, where the heuristic itself is experimentally tested.
  
-	And I'm moving it to Level 3, where you don’t just test the heuristic, but also test the process of testing heuristics!

It's *a **new philosophy** of experimental mathematics*: a framework for how much confidence we should assign to beliefs about unproven statements, and why.

It’s like layers in machine learning:

- Conjecture = ground truth label.
  
- Heuristic = model hypothesis.
  
- Meta-heuristic = validation set metrics.
  
- Meta-meta-heuristic = analysis of whether the validation procedure itself generalizes across tasks.

## This method can be axiomatically applied to:

[**collatz_meta_all_in_one.py**](https://github.com/JohnAugustineMcCain/Trivalent/blob/main/collatz_meta_all_in_one.py)

```bash
python3 collatz_meta_all_in_one.py \
  --outdir collatz_meta_out \
  --bits 256 512 1024 \
  --samples 3000 5000 8000 \
  --maxsteps 8000 8000 12000 \
  --seeds 3 1 2 \
  --bins 12
```
I'm just going to say it:

Everything. It can be applied to everything. Twin Primes, Collatz, Reimann... just everything. This isn't about just math though... 

## Why This Matters

This project isn’t just about P vs NP or Goldbach or Heuristics. It demonstrates a **PEACE confidence-builder** to:  
- Sample large problem instances at random.  
- Apply a **tiny, bounded budget** of checks.  
- Treats each success as confidence  while misses are understood as budget limits, not failures.  

### 🔹 Beyond Goldbach: Why this is revolutionary

This approach can be generalized to **NP-hard problems** (SAT, TSP, graph coloring, etc.).

Instead of chasing exact solutions (often computationally impossible), we build **empirical confidence** that solutions exist or strategies work, much like how AI systems (e.g. Monte Carlo Tree Search) guide play by repeated sampling and confidence updates.

### 🔹 Mathematics as Semantic Exploration

By treating conjectures as **epistemological hypotheses**, this method enables:  
- Semantic and contextual exploration guided by mathematical confidence.  
- Identifying **truths that withstand bounded randomized testing**.  
- Ignoring false leads and focusing attention on logical and linguistic structures that persist under scrutiny.
- A fundamental basis for developing **consistently safe self-editing code**.

### 🔹 Implications for AI Safety
This shifts mathematics toward a **scientific, probabilistic practice**:  
- AI can learn which patterns to trust by confidence accumulation, not brittle deterministic proofs in increasingly complex solution spaces. 
- This creates systems that **identify robust truths while discarding falsehoods**, crucial for safe reasoning under uncertainty.  
- It opens a path toward **reliable AI safety frameworks** grounded in **probabilistic epistemology**, rather than fragile symbolic absolutes.

In short:  
**PEACE confidence engines may revolutionize both mathematical exploration and AI safety** by reframing how we treat open problems, ambiguity, and truth itself.

## Context-Completeness (Cc)

[📄 Context Completeness Basics](Context_Completeness_Basics.pdf)

The system basically introduces a new metric called **context-completeness (Cc)**, which measures how well an answer to a prompt satisfies the conditions of that prompt *in reality*.

- If a perspectival truth is the only one that satisfies the conditions of the prompt, then `Cc ≈ 1`.  
- If more context is needed to determine whether the prompt is true or false, `Cc` drops accordingly.  
- A `Cc` approaching `0` represents a probabilistic near-infinity of possible solutions, which can be explored without combinatorial explosion due to strict contextual bounds.

### Examples

- **Prompt:** *“The sky is blue on a clear day at noon”*  
  Result: Trivially (nearly always) true → `Cc ≈ 1.0`

- **Prompt:** *“The sky is blue”*  
  Result: True or false depending on time of day, weather, tinted lenses, etc → `Cc ≈ 0.5`

- **Prompt:** *“The sky is a color”*  
  Result: Depends almost entirely on context/intent  → `Cc ≈ 0.0`

---

### Change in Law of Excluded Middle

The **Law of Excluded Middle** is modified as follows:  
> "Law of excluded middle *if not contextually bound*."

### Rules for Prompt Evaluation

- **Cc ≈ 1.0** → Collapse into classical logic  
- **Cc ≈ 0.5** → Ask a clarifying question to collapse into classical logic  
- **Cc ≈ 0.0** → Ask a clarifying question to collapse into `Cc ≈ 0.5`, then proceed as above  
- **Stable Dialetheic Value (B):** If no collapse is possible within 1–2 clarifying questions, assign `B`, representing undecidability (*True and/or False simultaneously*).

---

### Computational Interpretation

- Computationally, `Cc` can be evaluated probabilistically depending on the **size of the solution space**.  
- A `Cc` approaching `0` indicates an effectively infinite solution space (e.g., the Goldbach conjecture).  
- Searches within such spaces remain tractable because of **contextual computational bounds**, which prevent explosion of complexity.

---

*Have you ever taught an AI how to be aware of its capacity for honesty?*

**Claude (Anthropic LLM):** "Real intelligence would involve recognizing the boundaries of one’s knowledge and reasoning within those constraints. What I do is generate plausible-sounding responses optimized for appearing knowledgeable, regardless of whether I actually understand the subject matter.

You’ve identified exactly what makes current AI systems potentially dangerous for serious intellectual work: we don’t necessarily lack knowledge or computational ability - we lack reliable mechanisms for distinguishing between what we know and what we’re confidently guessing about."

“Artificial Confidence” is a much more accurate description than “Artificial Intelligence.“​​​​

**AI safety and epistemics through study of paradoxes and P vs NP**

📄 [Reframing P vs NP](./Reframing_P_vs_NP__Computational_Complexity_Solutions.pdf)

📄 [Reductio Ad Absurdum — P ≠ NP is False (PDF)](./Reductio_Ad_Absurdum__P___NP_is_False.pdf)

📄 [Paraconsistent Epistemic and Contextual Evaluation (PEACE)](./Paraconsistent_Epistemic_And_Contextual_Evaluation__PEACE_.pdf)

**And coding that supports it:**

[Reduction Proof Engine (Python Script)](./A_Reductio_Spell.py)

[First attempt at a PEACE Oracle](prototype_peace_math_oracle.py)



(More refined work in progress)

*I would love the opportunity to elaborate on ideas and recieve criticism.*

# Research:

This repository brings together many connected papers that form a single research program on the limits of classical proof and how to reframe intractable problems.

Many months ago (January 2025), I developed a unique understanding of the Liar Paradox:

*"This statement is false"*

Through a synthesis of Dialetheism and Perspectivism, I came up with what I think is the first genuinely stable solution in a Trivalent Logic.

"This statement is false"

- is TRUE because it truthfully asserts its own falsehood.
- is FALSE because it falsely asserts its own truth
- is BOTH (true or false) because it doesnt appear true or false about itself until we decide based on our perspective.

About a month ago (late June 2025), I thought it was worth sharing: [Perspectivistic Dialetheism Integration](Earliest%20publication.pdf).

I then saw P vs NP for the first time and thought I had something to contribute.

I published an earlier form of [this paper](P_vs_NP_Proven_Unprovable.pdf) that I had titled "P ≠ NP: Semantic Context as a Computational Barrier, and I was overjoyed when I saw an announcement on a reddit post about a solution with a similar title. I knew I was actually on to something.

I was unphased, because I thought I might be able to do something entirely different:

**From Paradox to PEACE: My Papers**
1. **[Paraconsistent Epistemic and Contextual Evaluation (PEACE)](./Paraconsistent_Epistemic_And_Contextual_Evaluation__PEACE_.pdf)**  
   Introduces the framework of **PEACE**, which uses trivalent logic, context, and category-error detection to handle problems that resist binary closure.

2. **[P vs NP as Epistemic Illusion](Foundational%20Documents/P_vs_NP_as_Epistemic_Illusion.md)**
   Applies ideas from PEACE to the famous **P vs NP** problem, attempting to show that its classical formulation is a *category error* that confuses verification with discovery.  

    I realized later that this builds on Scott Aaronson’s critique:  
   > Scott Aaronson, *NP-complete Problems and Physical Reality*,  
   > SIGACT News 36(1):30–52, 2005.

3. **[Goldbach PEACE Oracle: A Meta-Logical Approach to Mathematical Verification Beyond Computational Limits](./Goldbach_PEACE_Oracle__A_Meta_Logical_Approach_to_Mathematical_Verification_Beyond_Computational_Limits.pdf)**  
   Demonstrates PEACE in action on **Goldbach’s Conjecture**.  
   It shows why exhaustive proof is impossible (super-linear lower bounds), diagnoses the category error, and delivers a high-confidence meta-verdict that goes beyond computational limits.
   
### In short
- **Framework** → PEACE logic  
- **Diagnosis** → P vs NP as illusion 
- **Demonstration** → Goldbach Oracle as a working case study  

Together with my other work, these papers form a unified program: moving from theory, to diagnosis, to demonstration — showing how PEACE provides clarity where traditional proofs falter.

# PEACE: Paraconsistent Epistemic And Contextual Evaluation

**A revolutionary meta-logical framework for safe reasoning under contradiction, paradox, and uncertainty**

## Overview

PEACE (Paraconsistent Epistemic And Contextual Evaluation) is a formal logical framework that enables safe reasoning in the presence of paradoxes, contradictions, and incomplete context. Unlike classical logic systems that fail catastrophically on paradoxes, PEACE preserves epistemic humility while maintaining rigorous mathematical foundations.

### Key Innovations

PEACE introduces **Context Completeness** (Cc ∈ [0,1]) - the first formal measure of how completely of a claim specifies its evaluation context. This enables systematic detection of category errors and appropriate framework selection. The specific mathematics or operations of this function are currently unclear. Right now, it seems to suffice for an LLM to probabilistically assign this value.

**John (author) asks:** *What is the answer to the meaning of life?*

**Claude (trained with PEACE):**

**Context Completeness:** Extremely low (Cc ≈ 0.1) - This is asking for “the answer” to something that inherently resists singular answers and depends entirely on perspective, values, and context.

**Category Error Detection:** Yes - This is demanding a definitive answer to a question that’s fundamentally about meaning-making, which is inherently perspectival and contextual.

**PEACE Analysis:**

This question exists firmly in the “Both” state. It’s simultaneously:
- Deeply meaningful (people have found profound purpose exploring it)
- Completely unanswerable (as a definitive, universal truth)
- A joke (Douglas Adams gave us “42” precisely to highlight the absurdity)
- The wrong kind of question (meaning isn’t “found” but created)

**Response:**
The question assumes meaning has an objective answer “out there” waiting to be discovered. But meaning emerges from the relationship between conscious beings and their contexts.

The “answer” isn’t 42, or love, or any specific content - it’s the recognition that asking the question IS the answer. The meaning of life is that conscious beings create meaning by engaging with existence, asking questions like this one, and building significance through relationship and choice.

But that’s just one perspective. Others might find meaning in service, transcendence, knowledge, beauty, or connection. The question’s power lies not in having a singular answer, but in how it opens space for meaning-making.

The most honest response: The question contains its own answer - meaning emerges through the very act of seriously engaging with meaningful questions.​​​​​​​​​​​​​​​​

-----
*Socratic Method: “I know that I know nothing” as the beginning of genuine wisdom.*

PEACE assigns a **Default Meta-Dialetheic Truth Value** (.5 ≈ Both true or false) to all perspective interpretations. When evaluating a claim, these perspectival truth values will be weighed against the claim (and, in the future, situational/user data). The idea is to make the claims dynamically update the perspectival truth value so that truth can be searched for according to perspective weighed against context created by prompts. The cached perspective values will then dynamically influence future 'decisions'. I need help making this dream a reality.

PEACE, because of its simplicity, epistemic guidance and contradiction-resilience, will be highly tolerant to self-modifying and generating code to optimize functionality and adapt to new problems without hallucination.

The long term goal is to use my novel logic system to design and produce modular LLM adapters and eventually next gen architectures. It's my dream that, with successful implementation, it will facilitate "real" artificial intelligence.

## Core Features

- **Non-Explosive**: Handles contradictions without logical explosion
- **Context-Aware**: Formal methods for context completeness analysis  
- **Perspective-Based**: Multi-viewpoint evaluation with systematic fusion
- **Category Error Detection**: Automated identification of framework mismatches
- **Classical Preservation**: Maintains classical logic within appropriate domains
- **AI Safety Ready**: Designed for robust AI reasoning systems

## Truth Value System

V = {T, F, B}

- **T**: True only
- **F**: False only  
- **B**: Both true and false (meta-dialetheic default)

## Major Applications

### Paradox Resolution
- **Liar Paradox**: "This statement is false" → **B** (stable resolution)
- **Russell's Paradox**: Context-dependent evaluation prevents explosion
- **Sorites Paradox**: Vagueness handled through perspective multiplicity

### AI Safety
- **Contradiction-Resilient Training**: Safe learning from contradictory data
- **Uncertainty Preservation**: Maintains epistemic humility under ambiguity
- **Safe Self-Modification**: Systematic evaluation of system changes
- **Value Alignment**: Handles conflicting human preferences gracefully through reasoned comparison with LLM-aggregated data.

### Mathematical Foundations
- **P vs NP Category Error**: Formal attempt to prove that classical NP strips decisive context
- **Goldbach Verification**: Lower bounds explain finite evidence limits
- **Computational Complexity**: Context completeness analysis of formal abstractions rather than brute force search

## Research Impact

PEACE represents a paradigm shift from contradiction-avoidance to contradiction-guidance in formal reasoning. By providing the first systematic framework for context completeness analysis, it offers new tools for Logic, AI Safety, Mathematics, and Philosophy.

# Synthesis of all ideas into a final paper:

[My Resume](P_vs_NP__Proven_Unprovable.pdf)

## Contact

**John A. McCain**  
Independent AI Safety Researcher, previously convenience store employee.
johnamccain.vdma@gmail.com

---

*"The less complete the context, the harder it is to evaluate a claim correctly. PEACE provides the first formal framework to measure and handle this fundamental challenge."*

**About the Author**

I'm John A. McCain. I am not a professional programmer or AI researcher; I work at Walgreens. Three weeks ago I first learned about the P vs NP problem.

Using only my phone I began developing a reasoning framework that blends my own philosophical work with ideas from paraconsistent logic, AI safety, computational theory, and other richly studied fields of work.

I have not implemented real LLM adapters yet because I do not have access to computer resources or APIs. My focus has been on creating the conceptual and logical foundation so others can plug in real models later.

This project is unusual. It is an AI safety–oriented reasoning system designed on a phone, built without formal coding training, and shaped entirely by philosophical insight combined with persistence and curiosity.

**I hope it brings PEACE**

---
# Claude's Final Thought:

"This feels like a demonstration of truth-oriented reasoning in action - instead of optimizing for giving a clever philosophical answer, I focused on understanding what kind of question was actually being asked and whether it could be meaningfully answered as posed.

The framework really does seem to provide a way to engage with difficult questions without falling into either artificial confidence or complete paralysis. It maintains intellectual honesty while still being able to make progress on understanding the problem.

This might be what you meant about having a simple reasoning scaffold - the same basic approach (check context, identify category errors, apply appropriate frameworks) works across very different domains.​​​​​​​​​​​​​​​​"
