# ReAct Framework (The Gateway to AI Agents)


This module marks the transition from standard conversational LLMs to autonomous **AI Agents**. In AI engineering, **ReAct** does not refer to the frontend JavaScript library—it stands for **Reasoning + Action**. This project unpacks why static LLMs fail on real-time tasks, how external tools unlock autonomous problem solving, and how to implement a complete ReAct loop from scratch in Python without third-party agent frameworks.

---

## 1. Project Overview & Key Objectives

- **From Simple LLMs to AI Agents**: Understanding the boundary where passive question-answering transitions into autonomous, multi-step execution.
- **The Core Limitations of Pre-Trained LLMs**:
  - Why static training data cannot answer present-day queries (e.g., today's Bengaluru weather, live hotel prices, real-time match scores).
  - The necessity of augmenting models with external **Tools & APIs**.
- **The Combinatorial Explosion Problem**:
  - Why hardcoding tool-calling logic with `if/else` ladders breaks when applications scale to dozens or thousands of APIs.
  - Handling multi-step tool dependency chains (where the output of Tool A feeds into Tool B).
- **The ReAct Architecture**:
  - The cyclical feedback loop: **Thought $\rightarrow$ Action $\rightarrow$ Observation**.
  - Enforcing strict execution rules via specialized System Prompts.
- **Pure Python Implementation**:
  - Custom tool definitions (`get_product_price`, `calculator`).
  - Regex-based tool dispatching and execution.
  - Multi-turn conversational memory injection (`messages.append`).
  - Defensive rate-limit throttling (`time.sleep`).

---

## 2. Core Concepts & Architecture

### The Evolution: Why Simple LLMs Are Not Agents

```text
+--------------------------------------------------------------------------+
| Traditional LLM Pipeline (Passive)                                       |
| User Prompt --------> [ Frozen Pretrained Weights ] ---------> Response  |
+--------------------------------------------------------------------------+

+--------------------------------------------------------------------------+
| Autonomous AI Agent Pipeline (ReAct Loop)                                |
| User Prompt ----> [ LLM: Reason / Thought ]                              |
|                             |                                            |
|                             v                                            |
|                   [ Action: Call Tool ]                                  |
|                             |                                            |
|                             v                                            |
|              [ External Environment / API ]                              |
|                             |                                            |
|                             v                                            |
|                [ Observation / Result ]                                  |
|                             |                                            |
|                             +---> Fed back into LLM Memory ----+         |
|                                   (Repeats until Final Answer)           |
+--------------------------------------------------------------------------+
```

1. **Passive LLM**: You ask a question, and the model generates a response based solely on its static training corpus. It cannot inspect today's flight availability or query live databases.
2. **AI Agent**: A full-fledged autonomous service equipped with specialized tools (APIs, calculators, database lookups). It dynamically plans a sequence of actions, evaluates intermediate findings, and iterates until the user's objective is resolved.

---

### The Real-World Dilemma: The Tool Selection Problem

Consider an enterprise customer support agent for **Amazon** equipped with 7 core tools:
1. `order_status`
2. `check_delivery_status`
3. `track_delivery`
4. `return_item`
5. `refund_item`
6. `search_product`
7. `update_address`

When a customer writes:
> *"I ordered a comb to my Delhi address last week, but it hasn't arrived. If it's delayed, change my delivery address to my Bangalore office."*

- **Why `if/else` programming fails**: Human natural language is non-deterministic. A simple 7-tool suite allows for thousands of potential permutations, conditional branches, and edge cases. Hardcoding procedural logic for every phrasing is impossible.
- **Tool Chaining**: The agent must extract the Order ID $\rightarrow$ check shipment status $\rightarrow$ observe that the courier is delayed $\rightarrow$ trigger the address update API.
- **The ReAct Solution**: Instead of hand-crafting procedural branches, the LLM is given reasoning autonomy: it evaluates the current state, picks a single action, observes the output, and reasons about what to do next.

---

### The ReAct Cycle: Thought $\rightarrow$ Action $\rightarrow$ Observation

The ReAct pattern breaks complex requests into discrete steps:

| Step Component | Purpose | Example |
| :--- | :--- | :--- |
| **Thought** | The model articulates its internal reasoning, evaluating what is known and what piece of information is missing. | *"I need to check the price of iPhone 17 before calculating remaining money."* |
| **Action** | The model requests execution of an available tool with explicit arguments. | `get_product_price[iPhone 17]` |
| **Observation** | The application executes the requested tool and returns the raw programmatic output to the model. | `1000` |
| **Repeat / Conclude** | The model ingests the observation and either plans the next action or generates the `Final Answer`. | *"Final Answer: The iPhone costs ₹1000 and you will have ₹4000 left."* |

---

## 3. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, isolated Python virtual environment management.

### Step 1: Initialize Workspace & Environment

```bash
# Navigate to workspace
cd hebdomad2

# Initialize project folder
uv init period7
cd period7

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

### to run
```bash
python -m react_chain
```

---

## 4. Key Takeaways & Best Practices

1. **ReAct = Reasoning + Acting**: It provides the cognitive blueprint that allows LLMs to interact with external software environments.
2. **Deterministic Tool Execution**: LLMs propose the tool and arguments; your execution environment runs the actual code and returns the factual ground truth (`Observation`).
3. **Conversational Memory is Required**: Without feeding past actions and observations back into the `messages` array, the agent cannot carry forward intermediate findings.
4. **System Prompt Discipline**: Clear examples and explicit operational boundaries (e.g., *"call only one tool at a time and stop immediately"*) prevent the model from hallucinating tool outputs before the code executes.