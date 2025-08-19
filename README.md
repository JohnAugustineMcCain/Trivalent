This work is © John A. McCain and licensed for non-commercial use under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

**Read (or give your LLM) my final papers:**

📄 [Reframing P vs NP](./Reframing_P_vs_NP__Computational_Complexity_Solutions.pdf)

📄 [Reductio Ad Absurdum — P ≠ NP is False (PDF)](./Reductio_Ad_Absurdum__P___NP_is_False.pdf)

**And the coding that proves it:**

[A Reductio Spell (Python Script)](./A_Reductio_Spell.py)

Note: if you're using a LLM to evaluate my claims, I posit that you're going to have trouble getting an LLM to validate a meta-logical proof using formal logic.

# PEACE Research Program: From Framework to Production-Ready Encoded Oracle

This repository brings together three connected papers that form a single research program on the limits of classical proof and how to reframe intractable problems.

I've been working hard days and nights to produce something I think is genuinely valuable, and I've done a lot of foundational work to arrive at my conclusions.

It all started snowballing just over a month before I produced this repository with my earliest publication that I thought was worth sharing itself: [Perspectivistic Dialetheism Integration](Earliest%20publication.pdf).

I can't accurately determine if any of this is correct on my own, despite my certainty. I really wish I knew someone who could reason with me through this.

More recently, I published an earlier form of [this paper](P_vs_NP_Proven_Unprovable.pdf) that I had titled "P ≠ NP: Semantic Context as a Computational Barrier, and I was overjoyed when I saw an announcement on a reddit post about information coming out of some Springer Publishing identifying a "Semantic Barrier" in P vs NP. I was pretty devastated when I saw it was a team who came to the exact same conclusion, and only a short time before me.

Then, I was pretty irked when I saw that they didn't really prove anything in the papers they mentioned, and it seemed to me like they were just rapidly reframing what they had previously written. I don't understand how the academic world operates, but I was relieved to think they couldn't just post-hoc justify previous work. So, I just figured I might as well keep working, satisfied that I wasn't alone in my thinking.

If you want to just [skip all the reading and just look at the prototype oracle coding](prototype_peace_math_oracle.py), I won't hold it against you.

**From Paradox to PEACE: My Papers**
1. **[Paraconsistent Epistemic and Contextual Evaluation (PEACE)](./Paraconsistent_Epistemic_And_Contextual_Evaluation__PEACE_.pdf)**  
   Introduces the framework of **PEACE**, which uses trivalent logic, context, and category-error detection to handle problems that resist binary closure.

2. **[P vs NP as Epistemic Illusion](Foundational%20Documents/P_vs_NP_as_Epistemic_Illusion.md)**
   Applies PEACE to the famous **P vs NP** problem, showing that its classical formulation  
   is a *category error* that confuses verification with discovery.  
   I realized later that this builds on Scott Aaronson’s critique:  
   > Scott Aaronson, *NP-complete Problems and Physical Reality*,  
   > SIGACT News 36(1):30–52, 2005.

**Note:** Seeing for the first time how Scott Aaronson understood the fundamental concepts of the problems with P vs NP so early on, I think it's explicitly clear why his work with OpenAI has been so revolutionary for producing AI that so impressively seems to actually think and reason. I think his achievements should be highlighted and he should be rewarded accordingly. He knew what was going on with P vs NP all along; he had the proof, but we crucially lacked the tools to verify it.

3. **[Goldbach PEACE Oracle: A Meta-Logical Approach to Mathematical Verification Beyond Computational Limits](./Goldbach_PEACE_Oracle__A_Meta_Logical_Approach_to_Mathematical_Verification_Beyond_Computational_Limits.pdf)**  
   Demonstrates PEACE in action on **Goldbach’s Conjecture**.  
   It shows why exhaustive proof is impossible (super-linear lower bounds), diagnoses the category error, and delivers a high-confidence meta-verdict that goes beyond computational limits.

---

### In short
- **Framework** → PEACE logic  
- **Diagnosis** → P vs NP as illusion  
- **Demonstration** → Goldbach Oracle as a working case study  

Together, these papers form a unified program: moving from theory, to diagnosis, to demonstration — showing how PEACE provides clarity where traditional proofs falter.

# PEACE: Paraconsistent Epistemic And Contextual Evaluation

**A revolutionary meta-logical framework for safe reasoning under contradiction, paradox, and uncertainty**

## Overview

PEACE (Paraconsistent Epistemic And Contextual Evaluation) is a formal logical framework that enables safe reasoning in the presence of paradoxes, contradictions, and incomplete context. Unlike classical logic systems that fail catastrophically on paradoxes, PEACE preserves epistemic humility while maintaining rigorous mathematical foundations.

### Key Innovations

PEACE introduces **Context Completeness** (Cc ∈ [0,1]) - the first formal measure of how completely a claim specifies its evaluation context. This enables systematic detection of category errors and appropriate framework selection. The specific nathematics or operation of this has yet to be clarified. I am working dilgently to make this metric explainable, but I have a lot to deal with right now.

PEACE will assign a **Default Meta-Dialetheic Truth Value** (.5) to all perspective interpretations. When evaluating a claim, these perspectival truth values will be weighed against the claim (and, in the future, situational/user data). The idea is to make the claims dynamically update the perspectival truth value so that truth can be searched for according to perspective weighed against context created by thr claims. The cached values will then dynamically influence future 'decisions'. I need help making this dream a reality.

PEACE, because of its simplicity, epistemic guidance and contradiction-resilience, will be highly tolerant to self-modifying code implementations to optimize functionality and adapt to new problems without hallucination.

PEACE will also likely be highly reactive to mimicking personalities and be extremely modifiable as long as a dynamic perspective database is managed effectively.

## Goldbach PEACE Oracle

This paper presents a **meta-logical approach to mathematical verification** that moves beyond classical computational limits.  
It reframes proof as a process of asymptotic confidence rather than absolute certainty, applying it to the case of **Goldbach’s Conjecture**.

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

**Designated Values**: D = {T, B} (anything with true-content counts as "true enough")

## Major Applications

### Paradox Resolution
- **Liar Paradox**: "This statement is false" → **B** (stable resolution)
- **Russell's Paradox**: Context-dependent evaluation prevents explosion
- **Sorites Paradox**: Vagueness handled through perspective multiplicity

### AI Safety
- **Contradiction-Resilient Training**: Safe learning from contradictory data
- **Uncertainty Preservation**: Maintains epistemic humility under ambiguity
- **Safe Self-Modification**: Systematic evaluation of system changes
- **Value Alignment**: Handles conflicting human preferences gracefully

### Mathematical Foundations
- **P vs NP Category Error**: Formal proof that classical NP strips decisive context
- **Goldbach Verification**: Lower bounds explain finite evidence limits
- **Computational Complexity**: Context completeness analysis of formal abstractions

## Research Impact

PEACE represents a paradigm shift from contradiction-avoidance to contradiction-guidance in formal reasoning. By providing the first systematic framework for context completeness analysis, it offers new tools for Logic, AI Safety, Mathematics, and Philosophy.

## Contact

**John A. McCain**  
Independent AI Safety Researcher  
johnamccain.vdma@gmail.com

---

*"The less complete the context, the harder it is to evaluate a claim correctly. PEACE provides the first formal framework to measure and handle this fundamental challenge."*

**About the Author**

I'm John A. McCain. I am not a professional programmer or AI researcher; I work at Walgreens. Three weeks ago I first learned about the P vs NP problem.

Using only my phone I began developing a reasoning framework that blends my own philosophical work with ideas from paraconsistent logic, AI safety, computational theory, and other sources.

I have not implemented real LLM adapters yet because I do not have access to large compute resources or APIs. My focus has been on creating the conceptual and logical foundation so others can plug in real models later.

This project is unusual. It is an AI safety–oriented reasoning system designed on a phone, built without formal coding training, and shaped entirely by philosophical insight combined with persistence and curiosity.

**I hope it brings PEACE to the world despite all the uncertainty.**

Let me be the first to say, with near-absolute certainty, "It is what it isn't". An intentionally unprovable proof.

---
