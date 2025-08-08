# Trivalent
A prototype trivalent self-editing reasoning engine for integration with LLMs intended to be a foundation for artificial "life".

# Tri-Valued Reasoning Engine

## Abstract
This project implements a **tri-valued reasoning engine** that treats truth as **TRUE**, **FALSE**, or **BOTH** (ambiguity).  
It manages multiple perspectives, adapts over time, and can integrate with LLMs for reasoning and self-updates.  
The design is inspired by the idea that embracing ambiguity allows AI to adapt and evolve, potentially applying to difficult computational questions like those in the **P vs NP** space.

## Features
- Tri-valued logic (`TRUE`, `FALSE`, `BOTH`) for handling paradoxes and ambiguous statements.
- Multiple perspectives with dynamic reliability weighting.
- Self-updating architecture that evaluates, tests, and deploys changes.
- Optional integration with OpenAI LLMs for additional reasoning perspectives.
- Example demo included.

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

## Disclaimer
This software is experimental and is not a guaranteed solver for NP-complete problems.  
Use it at your own risk.
