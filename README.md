# Trivalent Reasoning Engine
**READ THE FULL PAPER ABOUT P VS NP:** [P vs NP as Epistemic Illusion](P_vs_NP_as_Epistemic_Illusion.md)
A prototype trivalent self-editing reasoning engine for integration with LLMs intended to be a foundation for artificial "life".

Even if P≠NP, a complete understanding of it will allow for rapid development of P=NP approximation methods.

Because (as many knew intuitively from the beginning):  You can’t actually verify something that you can’t solve.

- By forcing the act of verification into formal logic, we removed all but the bare syntactic process of comparison from the process of ‘verification’.
- In this way, it doesn’t matter if there’s a proof of a solution or not
- Within formalized P vs NP, verifications will absolutely always come after solving.

- Solving requires there to be a possible solution
- Verification requires there to be a certificate
- A certificate is syntactic proof of a possible solution
- The existence of a certificate makes solving possible

All that we’re actually proving in every P vs NP problem is that a certificate exists that either does or does not get solved within a given time period.

- Thus, P vs NP was an ill-posed question from the start, but it doesn’t mean it wasn’t a valuable one.

- Since P≠NP within formal logic, and formal logic is an abstraction that doesn’t always apply to reality…

…P versus NP in reality.

> ⚠ **Disclaimer:** This code is **experimental and untested**. It has not been run in a clean environment by the author. Use at your own risk.

# Tri-Valued Reasoning Engine
<!-- Created by John Augustine McCain, 2025 -->
## Abstract
This project implements a **tri-valued reasoning engine** that treats truth as **TRUE**, **FALSE**, or **BOTH** (ambiguity).  
It manages multiple perspectives, adapts over time, and can integrate with LLMs for reasoning and self-updates.  
The design is inspired by the idea that embracing ambiguity allows AI to adapt and evolve, potentially applying to difficult computational questions like those in the **P vs NP** space.
<!-- Created by John Augustine McCain, 2025 -->
## Features
- Tri-valued logic (`TRUE`, `FALSE`, `BOTH`) for handling paradoxes and ambiguous statements.
- Multiple perspectives with dynamic reliability weighting.
- Self-updating architecture that evaluates, tests, and deploys changes.
- Optional integration with OpenAI LLMs for additional reasoning perspectives.
- Example demo included.
<!-- Created by John Augustine McCain, 2025 -->
## Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/JohnAugustineMcCain/Trivalent.git
   cd Trivalent
   ```

2. Create and activate a virtual environment (recommended):  
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
See `requirements.txt`:
```
Python 3.9 or higher
openai>=1.0.0,<2.0.0
```

## Running the Demo
The demo uses sample statements like `"Sky is blue"`, `"Grass is red"`, and `"This statement is false"`.

Run:
```bash
python reasoning_engine.py
```

If you want LLM integration, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your_api_key_here"      # macOS/Linux
setx OPENAI_API_KEY "your_api_key_here"        # Windows
```
<!-- Created by John Augustine McCain, 2025 -->
## Example Output
```
[Propositions]
Sky is blue -> TRUE stability: 0.99
Grass is red -> FALSE stability: 0.98
Liar -> BOTH stability: 0.45
```

## License
This project is licensed under the **Creative Commons Attribution–NonCommercial 4.0 International License** (CC BY-NC 4.0).  
You may share and adapt it for non-commercial purposes with attribution.

---

© 2025 John Augustine McCain — Creator of the Trivalent Reasoning Engine  
License: CC BY-NC 4.0  
Repository: https://github.com/JohnAugustineMcCain/Trivalent - Trevalence@myyahoo.com

---

## Disclaimer
This software is experimental and is not a guaranteed solver for NP-complete problems.  
Use it at your own risk.
