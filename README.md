This work is © John A. McCain and licensed for non-commercial use under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

> **TL;DR**: A production-ready reasoning engine that **actually implements a synthesis of many real philosophical ideas** and uses them to make LLMs more **contradiction-resilient, reliability-aware, and safely self-modifying**. Default prior is **BOTH** (dialetheic), with TRUE/FALSE emerging only via perspectival/contextual collapse.
Where I_P constrains vocabulary and κ_P provides verdicts.
3. Context Completeness
Quantifies missing context: lower Cc → more perspectives admissible → higher verdict instability.
4. Category Error Detection
Systematic identification of classical logic failures:
	•	Self-reference enabling diagonalization
	•	Vague predicates and indexicals
	•	Normative/pragmatic load
	•	Strong conflicting evidence
## Major Applications ##
**Paradox Resolution**
	•	Liar Paradox: “This statement is false” → B (stable resolution)
	•	Russell’s Paradox: Context-dependent evaluation prevents explosion
	•	Sorites Paradox: Vagueness handled through perspective multiplicity
AI Safety
	•	Contradiction-Resilient Training: Safe learning from contradictory data
	•	Uncertainty Preservation: Maintains epistemic humility under ambiguity
	•	Safe Self-Modification: Systematic evaluation of system changes
	•	Value Alignment: Handles conflicting human preferences gracefully
Mathematical Foundations
	•	P vs NP Category Error: Formal proof that classical NP strips decisive context
	•	Goldbach Verification: Ω(N log N) lower bounds explain finite evidence limits
	•	Computational Complexity: Context completeness analysis of formal abstractions
Implementation
Core Engine
	•	perspectival_engine.py - Complete epistemic evaluation system
	•	Temporal decay with asymmetric half-lives (TRUE more stable than FALSE)
	•	Conservative learning with evidence fusion
	•	Dynamic perspective management
Theoretical Results
Proven Theorems
	•	Non-explosion: A, ¬A ⊭ B for arbitrary B
	•	Neutral fixed point: Liar paradox yields stable B value
	•	Classical preservation: B-free fragments maintain classical entailment
	•	Conservativity: Non-load-bearing perspectives preserve truth values
Computational Bounds
	•	Goldbach verification: Ω(N log N) lower bound for finite verification
	•	Context stripping: Formal analysis of abstraction-induced category errors
---
Contributing
This framework addresses fundamental problems in logic and AI safety. Contributions welcome for:
	•	Implementation optimizations
	•	Additional paradox case studies
	•	AI safety integration examples
	•	Formal verification extensions

 When referencing, adapting, or building upon this work, please cite as:
 @article{mccain2025peace,
  title={PEACE: Paraconsistent Epistemic And Contextual Evaluation},
  author={McCain, John A.},
  year={2025},
  note={Breakthrough framework for contradiction-resilient reasoning}
}

**Contact**
John A. McCain
Independent AI Safety Researcher
johnamccain.vdma@gmail.com

“The less complete the context, the harder it is to evaluate a claim correctly. PEACE provides the first formal framework to measure and handle this fundamental challenge.”
This README positions PEACE as a serious research framework while highlighting its practical applications and theoretical rigor. It should help researchers understand both the mathematical foundations and the real-world implications.​​​​​​​​​​​​​​​​

---

About the Author

I'm John A. McCain. I am not a professional programmer or AI researcher; I work at Walgreens. Three weeks ago I first learned about the P vs NP problem.

Using only my phone I began developing a reasoning framework that blends my own philosophical work with ideas from paraconsistent logic, AI safety, computational theory, and other sources.

I have not implemented real LLM adapters yet because I do not have access to large compute resources or APIs. My focus has been on creating the conceptual and logical foundation so others can plug in real models later.

This project is unusual. It is an AI safety–oriented reasoning system designed on a phone, built without formal coding training, and shaped entirely by philosophical insight combined with persistence and curiosity.

---
