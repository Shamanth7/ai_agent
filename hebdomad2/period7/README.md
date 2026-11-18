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

## 4. Code Implementation

```python
import os
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


def get_product_price(product):
    if product == 'iPhone 17':
        return 1000
    elif product == "iPhone 15":
        return 500
    else:
        return 0


def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"


tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

system_prompt = """
You are a shopping assistant.

You have these tools:

get_product_price(product)
calculator(expression)
IMPORTANT:
Call tools exactly like these examples:

Action: get_product_price("iPhone 17")
Action: calculator("5000 - 1000")

Never write:
get_product_price(product="iPhone 17")

Never write:
calculator(expression="5000 - 1000")
Follow these rules:

1. Decide what you need to do next.
2. Call ONLY ONE tool at a time.
3. After writing an Action, STOP immediately.
4. Never guess or invent a tool result.
5. Wait until you receive an Observation.
6. Then decide your next action.
7. When the task is complete, give the Final Answer.

Format:

Thought: what you need to do
Action: tool_name(argument)

When finished:

Final Answer: your answer
"""


def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):
        print("\n------------------")
        print("STEP", step + 1)
        print("------------------")

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content
        print(answer)

        # Agent has finished
        if "Final Answer:" in answer:
            break

        # Find the Action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            # Run the tool
            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)
            else:
                observation = "Tool not found"

            print("Observation:", observation)

            # Add LLM response to memory
            messages.append({
                "role": "assistant",
                "content": answer
            })

            # Give tool result back to LLM
            messages.append({
                "role": "user",
                "content": "Observation: " + str(observation)
            })
            sleep(5)


prompt = """
I have 5000 rupees. What is the price of an iphone 17?
and how much money will I have left?
"""
run_agent(prompt)
```

---

## 5. Step-by-Step Code Breakdown

### 1. Client & Environment Setup
* `load_dotenv()` reads `GROQ_API_KEY` from your `.env` file so sensitive credentials are never hardcoded.
* `Groq(api_key=my_api_key)` initializes the inference client targeting `openai/gpt-oss-120b`.

### 2. Custom Tools & Tool Registry (`tools`)
* **`get_product_price(product)`**: Simulates a live product catalog or database lookup. It returns `1000` for `"iPhone 17"`, `500` for `"iPhone 15"`, and `0` otherwise.
* **`calculator(expression)`**: Evaluates mathematical expressions (like `"5000 - 1000"`) so the LLM relies on deterministic Python arithmetic rather than guessing math. *(Note: In production environments, replace `eval()` with a safe math parser like `ast.literal_eval` or `numexpr` to prevent arbitrary code execution).*
* **`tools` dictionary**: Maps string tool names (e.g., `"get_product_price"`) directly to executable Python function references, enabling dynamic dispatch without `if/else` chains.

### 3. The ReAct `system_prompt`
* **Few-Shot Formatting**: Explicitly shows `Action: get_product_price("iPhone 17")` and forbids keyword arguments (`product=...`) so our regex parser can extract arguments reliably.
* **Execution Guardrails**: Rules 2–5 prevent the LLM from hallucinating an `Observation:` on its own or chaining multiple unexecuted actions in a single turn.

### 4. The Autonomous Execution Loop (`run_agent`)
1. **Bounded Loop (`for step in range(5)`)**: Caps execution at 5 iterations to prevent infinite loops and runaway API costs if the model gets stuck.
2. **Deterministic Inference (`temperature=0`)**: Minimizes randomness so the model strictly adheres to the `Thought` / `Action` / `Final Answer` syntax.
3. **Termination Check**: As soon as `"Final Answer:"` appears in the model's output, the loop breaks and finishes.
4. **Regex Action Parser (`re.search(r"Action:\s*(\w+)\((.*?)\)", answer)`)**:
   * `(\w+)` captures the function name into `tool_name` (Group 1).
   * `(.*?)` captures the raw argument inside parentheses into `tool_input` (Group 2), which is then stripped of whitespace and double quotes (`"`). *(Tip: Adding `.strip("'")` alongside `.strip('"')` makes it resilient to single-quoted strings as well).*
5. **Tool Execution & Memory Injection**:
   * Looks up `tool_name` in `tools`, executes it with `tool_input`, and captures the return value as `observation`.
   * Appends both the assistant's `Thought`/`Action` message and the new `"Observation: ..."` user message to `messages`, giving the LLM full conversational memory for the next step.
6. **Rate-Limit Protection (`sleep(5)`)**: Pauses 5 seconds between tool iterations to stay comfortably within free-tier API requests-per-minute (RPM) limits.

---

## 6. Execution Trace (Dry Run)

When `run_agent(prompt)` runs with *"I have 5000 rupees. What is the price of an iphone 17? and how much money will I have left?"*, the loop executes across three steps:

* **STEP 1**:
  * **LLM Output**:
    ```text
    Thought: I need to find the price of an iPhone 17 first.
    Action: get_product_price("iPhone 17")
    ```
  * **Python Tool Execution**: `Observation: 1000`
* **STEP 2**:
  * **LLM Output**:
    ```text
    Thought: The iPhone 17 costs 1000 rupees. Now I need to subtract 1000 from 5000 to find out how much money will be left.
    Action: calculator("5000 - 1000")
    ```
  * **Python Tool Execution**: `Observation: 4000`
* **STEP 3**:
  * **LLM Output**:
    ```text
    Thought: I now have both the price of the iPhone 17 and the remaining balance.
    Final Answer: The price of an iPhone 17 is 1000 rupees, and you will have 4000 rupees left.
    ```
  * **Loop Result**: `"Final Answer:"` is detected, and the loop exits cleanly.

---

## 7. Key Takeaways & Best Practices

1. **ReAct = Reasoning + Acting**: It provides the cognitive blueprint that allows LLMs to interact with external software environments.
2. **Deterministic Tool Execution**: LLMs propose the tool and arguments; your execution environment runs the actual code and returns the factual ground truth (`Observation`).
3. **Conversational Memory is Required**: Without feeding past actions and observations back into the `messages` array, the agent cannot carry forward intermediate findings.
4. **System Prompt Discipline**: Clear examples and explicit operational boundaries (e.g., *"call only one tool at a time and stop immediately"*) prevent the model from hallucinating tool outputs before the code executes.