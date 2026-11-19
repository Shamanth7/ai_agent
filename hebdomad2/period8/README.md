# Prompt Chaining (Multi-Step LLM Pipelines)


In real-world enterprise engineering, tasks are rarely simple, single-turn prompts (e.g., *"Raju had ₹100 and spent ₹50"*). Real production workflows—such as candidate evaluation, document synthesis, or multi-stage customer support triage—are inherently complex.

This module explores **Prompt Chaining**: the architectural practice of decomposing a monolithic, high-risk prompt into discrete, manageable sub-prompts where the output of one step serves as the contextual input for the next.

---

## 1. Project Overview & Key Objectives

* **Monolithic Prompts vs. Prompt Chaining**:
  * Why dumping a massive, multi-page instruction set into a single LLM call leads to failure.
  * The House Construction Analogy: general contractor vs. specialized stage-by-stage trades.
* **Architectural Contrast: ReAct vs. Prompt Chaining**:
  * **ReAct**: The model holds autonomy to reason and select which tool to call in an open loop.
  * **Prompt Chaining**: The developer defines and hardwires the deterministic pipeline sequence.
* **The Core Advantages of Prompt Chaining**:
  * **Granular Debugging & Observability**: Pinpointing exact points of failure without inspecting massive token dumps.
  * **System Modularity**: Modifying or testing single components without introducing regression risks across the application.
  * **Cost & Latency Optimization (Heterogeneous Model Routing)**: Assigning expensive frontier models to complex reasoning steps, and fast/inexpensive models to mechanical tasks.
  * **Granular Retries**: Re-running only the failing sub-step rather than re-executing the entire pipeline.
* **Hands-On Implementation**:
  * Building a modular, 3-step candidate evaluation pipeline using Groq and Python.
  * Common Python pitfalls: avoiding missing f-string syntax (`f"..."`) and constraining intermediate outputs.

---

## 2. Core Concepts & Architecture

### The Monolithic Prompt Trap (The House Construction Analogy)

Imagine you want to build a house:

* **The Monolithic Approach**: You hire a single worker and give them one massive instruction list: *"Buy land at coordinates X, draw the architectural blueprints, get municipality permits, excavate the foundation, build the brick walls, install plumbing, and paint the rooms."* If the resulting house is unlivable, diagnosing who or what went wrong is nearly impossible. Did they buy the wrong plot? Were the blueprints flawed? Did the concrete fail?
* **The Chained Pipeline Approach**: You break the construction into explicit sub-tasks. Task 1 (Buy Land) produces a boundary deed. That deed feeds into Task 2 (Architectural Blueprint). The blueprint feeds into Task 3 (Excavation & Build). Each milestone is validated before proceeding to the next.

In AI engineering:

```text
+-------------------------------------------------------------------------+
| Monolithic Prompt (Brittle & Opaque)                                    |
| Raw Inputs (Resume + JD) ----> [ Massive Monolithic Prompt ] ----> Score|
+-------------------------------------------------------------------------+

+-------------------------------------------------------------------------+
| Prompt Chaining Pipeline (Modular & Observable)                         |
|                                                                         |
| [Candidate Resume] ----> (Step 1: Parse Candidate Skills)               |
|                                    |                                    |
|                                    v [Candidate Skill List]             |
|                                                                         |
| [Job Description]  ----> (Step 2: Parse Required JD Skills)             |
|                                    |                                    |
|                                    v [Required Skill List]              |
|                                                                         |
| [Candidate Skills] + [JD Skills]                                        |
|           |                                                             |
|           v                                                             |
| (Step 3: Comparative Matching Engine) ----> [Match Breakdown & Score]   |
|           |                                                             |
|           v                                                             |
| (Optional Step 4: Conditional Logic)  ----> Score >= 60? Interview HR   |
|                                             Score <  60? Rejection Email|
+-------------------------------------------------------------------------+
```

---

### ReAct vs. Prompt Chaining: Who Decides the Workflow?

A common question among engineers is how Prompt Chaining differs from the ReAct pattern covered in Episode 08:

| Dimension | ReAct (Reasoning + Action) | Prompt Chaining |
| :--- | :--- | :--- |
| **Control Flow** | **Model-Driven**: The LLM autonomously decides *what* to do, *which* tool to call, and *when* to conclude. | **Developer-Driven**: The software engineer explicitly defines the execution graph, function order, and data transitions. |
| **Tool Execution** | Dynamic tool invocation triggered by string pattern matching in runtime loops. | Sequential function calls where intermediate outputs are explicitly injected as arguments into subsequent prompts. |
| **Determinism** | Flexible, but higher variance and prone to loop derailment if the model hallucinates tool signatures. | Deterministic sequence of execution, ensuring consistent application behavior. |
| **Ideal Use Cases** | Open-ended agent assistants, research bots, interactive customer support copilots. | Deterministic pipelines, document evaluation, content publishing workflows, ETL data transformations. |

---

### Why Production Systems Demand Prompt Chaining

1. **Root-Cause Debuggability**:
   * If a candidate unexpectedly receives a score of `30/100` instead of `90/100`, a monolithic prompt gives you no visibility into whether the resume parser failed, the JD parser dropped key requirements, or the scoring formula miscalculated.
   * With prompt chaining, you inspect each step's discrete return value in isolation.
2. **Cost & Latency Optimization (Heterogeneous Models)**:
   * Not every step requires a high-cost frontier model.
   * You can route Step 1 (complex semantic extraction from poorly formatted text) to an advanced model, while routing Step 2/Step 3 (comparing two small extracted lists or formatting an email) to a lightweight, cheaper, and faster model.
3. **Targeted Prompt Iteration**:
   * If the comparison step fails because the candidate skills include unnecessary conversational filler, you adjust the output constraints of Step 1 without modifying or risking regressions in Step 2 or Step 3.

---

## 3. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, isolated virtual environment management.

### Step 1: Initialize Workspace

```bash
# Navigate to workspace
cd hebdomad2

# Initialize hebdomad2 
uv init hebdomad2
cd hebdomad2

# Create Python virtual environment
uv venv 

# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
uv add groq python-dotenv
```

---

## 3. Python Implementation: 3-Step Candidate Evaluation Pipeline

Below is the complete Python implementation demonstrating how to chain three discrete LLM calls using the Groq SDK.

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

JD = """
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""

RESUME = """
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git

Projects:
Built a food delivery backend using
FastAPI and MySQL.

Deployed applications using Docker.
"""

def ask_llm(system_prompt, user_prompt):
    sys_msg = {
        "role": "system",
        "content": system_prompt
    }
    user_msg = {
        "role": "user",
        "content": user_prompt
    }
    messages = [sys_msg, user_msg]
    resposne = client.chat.completions.create(model=model, messages=messages)
    answer = resposne.choices[0].message.content
    return answer


def step1_res_extract(RESUME):
    print("STEP 1")
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the candidates resume provided.
    Only return the skills no other information. Do not invent any skillsby yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt = f"""
    Extract the skills from this resume
    {RESUME}
    """
    return ask_llm(system_prompt, user_prompt)


def step2_JD_extract(JD):
    print("step2")
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the Job description  provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt = f"""
    Extract the skills from this JD
    {JD}
    """
    return ask_llm(system_prompt, user_prompt)


def step3_match(candidate, jd):
    print("step3")
    system_prompt = """
    You are a professional HR assistant. compare the skills of candidate and the skills required in the JD and produce a final score between
    1 and 100. also produce a short verdict whther the candidate is a good fit for the role.
    """
    user_prompt = f"""
    Compare and matc h the skills
    JD:
    {jd}
    Candidate:
    {candidate}
    """
    return ask_llm(system_prompt, user_prompt)


candidate = step1_res_extract(RESUME)
print(candidate)
sleep(2)

jd = step2_JD_extract(JD)
print(jd)
sleep(2)

score = step3_match(candidate, jd)
print(score)
```

---

## 4. Code Walkthrough & Component Breakdown

### 1. Client Initialization & Environment Guardrails
* **`load_dotenv()` & `os.getenv("GROQ_API_KEY")`**: Loads the secret API key from `.env` so credentials are never hardcoded into source control.
* **Fail-Fast Validation**: If `GROQ_API_KEY` is missing, `raise ValueError("where is your api key")` stops execution immediately before making failing network requests.
* **Model Selection**: Uses `openai/gpt-oss-120b` via the `Groq` client for fast inference across all three stages.

### 2. Reusable LLM Wrapper (`ask_llm`)
* Instead of repeating `client.chat.completions.create(...)` inside every pipeline stage, `ask_llm(system_prompt, user_prompt)` encapsulates the API call.
* It separates **behavioral rules** (`role: "system"`) from **raw input data** (`role: "user"`), ensuring clean instruction-following at every step.

### 3. Stage 1: Resume Skill Extraction (`step1_res_extract`)
* **Single Responsibility**: Reads unstructured candidate resume text (`RESUME`) and extracts only technical skills.
* **Anti-Hallucination & Format Constraints**: Explicitly instructs the model *"Do not invent any skills by yourself"* and forces a clean comma-separated list (`Python, FastAPI, MySQL, Docker, REST APIs, Git`) with zero conversational filler so downstream functions receive clean data.

### 4. Stage 2: Job Description Skill Extraction (`step2_JD_extract`)
* **Single Responsibility**: Parses the hiring requirements (`JD`) into a matching comma-separated format (`Python, FastAPI, Django, PostgreSQL, Docker, AWS, REST APIs`).
* **Symmetrical Formatting**: By enforcing the exact same output structure in Stage 1 and Stage 2, Stage 3 receives two clean, standardized lists to compare.

### 5. Stage 3: Comparative Matching & Scoring (`step3_match`)
* **Context Injection**: Takes the outputs of Stage 1 (`candidate`) and Stage 2 (`jd`) and injects them into a new `user_prompt` using Python f-strings (`f"""..."""`).
* **Synthesis & Verdict**: Evaluates overlaps (`Python`, `FastAPI`, `Docker`, `REST APIs`) against missing requirements (`PostgreSQL` vs. `MySQL`, and missing `AWS`), returning a numerical score (`1–100`) and a hiring recommendation.

### 6. Pipeline Orchestration & Rate-Limit Protection
* **Sequential Data Flow**:
  `RESUME` $\to$ `step1_res_extract()` $\to$ `candidate`
  `JD` $\to$ `step2_JD_extract()` $\to$ `jd`
  `(candidate, jd)` $\to$ `step3_match()` $\to$ `score`
* **Throttling (`sleep(2)`)**: Pauses execution for 2 seconds between consecutive API calls to avoid triggering Groq burst rate limits (`HTTP 429`).

---

## 5. Common Pitfalls & Live Debugging Insights

During multi-step pipeline construction, subtle engineering bugs often emerge:

1. **The Missing Python f-string Trap (`"..."` vs `f"..."`)**:
   * If you write:
     ```python
     user_prompt = "Extract skills from: {resume_text}"  # Missing 'f'
     ```
     Python passes the literal string `"{resume_text}"` to the LLM instead of the candidate's data. The model will respond: *"I cannot extract skills because no resume was provided."*
   * Always ensure template prompts use valid f-string syntax (`f"..."`).

2. **Intermediate Output Formatting Leaks**:
   * If Step 1 returns conversational fluff (*"Sure! Here are the candidate skills: ..."*), Step 3's matching engine may get confused or incorporate the conversational text into its comparison.
   * Enforce clean output constraints in intermediate steps (e.g., *"Return only a comma-separated list with no preamble"*).

3. **API Rate Limiting (`HTTP 429`)**:
   * Calling multiple LLM prompts in rapid succession can trigger burst rate limits.
   * Always include throttling safeguards (`time.sleep(2)`) between pipeline stages.

---

## 6. Key Takeaways & Best Practices

1. **One Prompt, One Single Responsibility**: Adhere to the Single Responsibility Principle. Design each prompt to perform exactly one task exceptionally well.
2. **Chain, Don't Cram**: Decompose complex tasks into verifiable, sequential stages.
3. **Isolate Failures**: When a pipeline produces an error, inspect intermediate step outputs to identify whether the issue lies in extraction, synthesis, or formatting.
4. **Mix & Match Models**: Optimize operational budgets by reserving expensive reasoning models for high-complexity steps while delegating mechanical extraction to lightweight models.