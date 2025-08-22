This work is © John A. McCain and licensed for non-commercial use under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

**Skip the readme?**
📄 [My final work (so far)](P_vs_NP__Proven_Unprovable.pdf)

📄 [And / Or the summary and defense of it](https://github.com/<username>/<repo>/blob/main/The_Verification_Asymmetry.pdf)

**Goldbach Proof of Concept (Cc + Budgeted Collapse)**

📄 [Goldbach Proof of Concept](https://github.com/JohnAugustineMcCain/Trivalent/blob/main/goldbach_proof_of_concept.py)

This prototype illustrates *Context-Completeness (Cc)* belief updates when probing bounded instances of Goldbach’s conjecture.

Belief increments slightly on bounded successes, is unaffected by bounded failures, and collapses only under:

1. A *validated counterexample* (computationally impossible), or  
2. A *budgeted collapse* (after 499 steps or if sampled n > 4·10^90).  

Without these budgets, verification or counter-example proof would take longer than the age of the universe.

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
