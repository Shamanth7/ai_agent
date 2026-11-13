# Tokens Explained | LLM Tokenomics & API Limits

This module demystifies the fundamental unit of computation in Large Language Models: **Tokens**. Rather than viewing tokens as abstract units, this project unpacks the algorithmic and economic necessity of tokenization, explores how text translates into numbers, examines token-based billing and limits, and demonstrates how to inspect and control token consumption using Python and the Groq SDK.

---

## 1. Project Overview & Key Objectives

- **Why Tokens Exist**: Understand the computational necessity of translating human language into numbers.
- **The Fast-Food Cart Analogy**: Intuitive mental model for pre-computed, reusable vocabulary versus ad-hoc assembly.
- **Evolution of Text Representation**:
  - Why **character-level / letter-by-letter (ASCII)** encoding fails.
  - Why **whole-word dictionary lookup (e.g., Oxford Dictionary)** fails.
  - How **sub-word tokenization** solves the out-of-vocabulary problem.
- **Token Economics & Billing**:
  - `prompt_tokens` (input tokens).
  - `completion_tokens` (output / generated tokens).
  - `total_tokens` (total billable transaction cost).
- **Execution Governance**:
  - Restricting output cost with `max_tokens`.
  - Detecting termination states with `finish_reason` (`"stop"` vs. `"length"`).
- **Critical Python Pitfall**: Avoiding circular imports caused by naming files `token.py`.

---

## 2. Core Concepts & Architecture

### The Fast-Food Cart Analogy

Imagine running a street-food fast-food cart:
- Every morning, you pre-cook high-demand, common items: plates of Maggi, boiled eggs, and cups of chai. When a customer orders Maggi, it is served immediately.
- If a customer asks for a rare, custom combination like **"Chocolate Egg Maggi"**, you do not have it pre-made. You have to break the order into components (cook Maggi + boil an egg + melt chocolate on top) and assemble it piece by piece.

**Tokens follow this exact principle:**
- Highly common, reusable words and syllables are stored as predefined building blocks in the model's vocabulary.
- Rare, compound, or non-dictionary words are broken into smaller sub-tokens and processed piece by piece.

---

### The Evolution: Why Sub-Word Tokens Won

LLMs are computer programs running on digital hardware. Computers do not understand English, Hindi, or emojis; they only process numbers:

```text
Human Text ("Hello") 
   ---> Tokenization / Number Conversion 
   ---> LLM Computation 
   ---> LLM Output (Numbers) 
   ---> Detokenization / Text Mapping 
   ---> Human Response ("Hi")
```

#### Approach 1: Character / Letter-by-Letter Encoding (ASCII)
- **Concept**: Convert each letter to its ASCII code (`'A' = 65`, `'a' = 97`, etc.).
- **Why it failed**:
  - **Extreme input bloat**: Encoding `"Hello"` requires 5 numbers; a standard 100-word paragraph explodes into hundreds of numbers.
  - **Inefficient context windows**: A 5,000-line codebase would exceed memory capacity, forcing harsh restrictions (e.g., limiting users to 4 words per prompt).

#### Approach 2: Whole-Word Dictionary Lookups (~600k Oxford Words)
- **Concept**: Assign a unique ID to every word in a dictionary (`"Hello" = 1`, `"the" = 2`, `"apple" = 3`).
- **Why it failed**:
  - **Open-ended language**: Real-world communication includes slang, personal names (*"Pratyush"*), brand names (*"Zomato"*), typos (*"aple"* instead of *"apple"*), and invented expressions (*"pratyushification"*).
  - **Out-Of-Vocabulary (OOV) crashes**: A static dictionary cannot cover infinite neologisms or compound words.

#### Approach 3: Sub-Word Tokenization (The Child Language Model)
- **How it works**: Just like a child learns root words (`play`, `walk`, `eat`) and learns that appending the suffix `-ing` modifies the action (`playing`, `walking`, `eating`), LLM tokenizers identify common sub-word stems and affixes across massive text corpora.
- **Example**:
  - **Common words remain single tokens**: `"Bengaluru"` (9 letters) or `"Aeroplane"` may exist as **1 single token**.
  - **Rare words split into sub-tokens**: `"Pratyush"` (8 letters) might split into 3 sub-tokens: `["Pr", "ty", "ush"]`.
- **Cost Consequence**: Because APIs charge per token, processing an 8-letter rare word can cost $3\times$ more than processing a 9-letter common word.

---

### Token Accounting & Limits

Every API roundtrip returns a `usage` payload tracking your transaction cost:

$$\text{Total Tokens} = \text{Prompt Tokens (Input)} + \text{Completion Tokens (Output)}$$

When controlling response lengths and costs in production:
- `max_tokens`: Enforces a hard ceiling on output generation to prevent runaway token expenditure.
- `finish_reason`:
  - `"stop"`: The model naturally completed its thought and reached its end-of-sequence token.
  - `"length"`: The generation was abruptly cut off because it hit the configured `max_tokens` ceiling.

---

## 3. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast Python package and environment management.

### Step 1: Initialize Workspace & Virtual Environment

```bash
# Navigate to your workspace directory
cd hebdomad

# Initialize project folder
uv init period3
cd period3

# Create virtual environment
uv venv 

# Activate virtual environment

source .venv/bin/activate
```

### Step 2: Install Dependencies & Configure Environment Variables

Install the required Groq SDK and `python-dotenv` packages:

```bash
uv add groq python-dotenv
```


> **CRITICAL PYTHON PITFALL**: **Never** name your Python script `token.py`! Python has a built-in standard library module named `token` that `python-dotenv` relies on internally. Naming your file `token.py` will trigger an `ImportError` / circular import crash. Always use a safe name like `tokens.py` or `token_limits.py`.

---


---

## 5. Running the Project & Expected Behavior

Run the script inside your activated virtual environment:

```bash
python -m tokens
```

### Observing `finish_reason`
1. **Short Prompt (`"Hi"`)**: Naturally finishes well under the `100` token ceiling $\rightarrow$ `finish_reason` is **`"stop"`**.
2. **Long Prompt (`"Write a 1000-word essay..."`)**: Hits the `max_tokens=100` limit mid-generation and gets truncated $\rightarrow$ `finish_reason` is **`"length"`**.
3. **Scaling `max_tokens`**: Increasing `max_tokens` to `5000` allows all three prompts to complete naturally with `finish_reason` returning **`"stop"`**.


# Token Consumption & API Limits: Code Walkthrough (`tokens.py`)

This guide provides a detailed breakdown of the Python implementation used to track token consumption, enforce output limits with `max_tokens`, and inspect termination states using `finish_reason` via the Groq API.

---

## 1. Complete Code (`tokens.py`)

> **Important Note on File Naming:** Do **not** name this script `token.py`. Python has a built-in standard library module named `token` that `python-dotenv` relies on internally. Naming your script `token.py` will cause a circular import crash (`ImportError`). Always name it `tokens.py`.

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"

# 3 prompts
prompt1 = "Hi!"
prompt2 = "Explain time travel in Detail but under 100 words"
prompt3 = "Write a 1000 word essay on Machine learning"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
        "role": role,
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=5000
    )
    usage = response.usage
    print(
        f"Prompt: {prompt} -->"
        f"your tokens: {usage.prompt_tokens} "
        f"completion_tokens: {usage.completion_tokens} "
        f"total tokens: {usage.total_tokens}  "
        f"Finish Reason: {response.choices[0].finish_reason}"
    )
```

---

## 2. Step-by-Step Explanation

### Step 1: Importing Libraries & Loading Environment Variables

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")
```

* **`load_dotenv()`**: Reads the `.env` file in your project folder and loads your secret credentials into environment variables so you don't hardcode sensitive keys in your source code.
* **`os.getenv("GROQ_API_KEY")`**: Retrieves the value stored under `GROQ_API_KEY`.
* **Guard Check (`if not my_api_key`)**: Immediately stops the program with a custom error message (`"API key kaha hai bhai"`) if the API key is missing or `.env` was not found.

---

### Step 2: Initializing the Groq Client & Model Settings

```python
client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"
```

* **`client = Groq(...)`**: Registers your script as an authenticated client with the Groq server using your API key.
* **`model`**: Specifies which LLM to run (`llama-3.3-70b-versatile`, a 70-billion-parameter model).
* **`role = "user"`**: Sets the sender role for the chat messages. In chat completion APIs, every message needs a role (`"system"`, `"user"`, or `"assistant"`).

---

### Step 3: Defining Prompts of Varying Output Lengths

```python
prompt1 = "Hi!"
prompt2 = "Explain time travel in Detail but under 100 words"
prompt3 = "Write a 1000 word essay on Machine learning"

prompts = [prompt1, prompt2, prompt3]
```

* Three prompts are defined to observe how output length directly affects token consumption:
  1. **`prompt1`**: A single greeting expecting a short reply.
  2. **`prompt2`**: A medium-length explanation constrained to under 100 words.
  3. **`prompt3`**: A long-form 1,000-word essay requiring many output tokens.
* All three strings are grouped into a Python list (`prompts`) so they can be processed sequentially in a loop.

---

### Step 4: Sending Requests with `max_tokens`

```python
for prompt in prompts:
    message = {
        "role": role,
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=5000
    )
```

* **Formatting `messages`**: The Groq chat completions endpoint expects a list of dictionaries (`messages`), where each dictionary contains `"role"` and `"content"`.
* **`max_tokens=5000`**: Sets a hard upper limit on how many tokens the model is allowed to generate in its response.
  * With `5000`, even the 1,000-word essay ($\approx 1,300\text{--}1,500$ tokens) has plenty of room to finish completely.
  * If you lower this to `max_tokens=50`, the model will abruptly cut off `prompt2` and `prompt3` once it hits 50 output tokens.

---

### Step 5: Inspecting Token Usage & `finish_reason`

```python
    usage = response.usage
    print(
        f"Prompt: {prompt} -->"
        f"your tokens: {usage.prompt_tokens} "
        f"completion_tokens: {usage.completion_tokens} "
        f"total tokens: {usage.total_tokens}  "
        f"Finish Reason: {response.choices[0].finish_reason}"
    )
```

Every API response returns a `usage` object and a `choices` list:

* **`usage.prompt_tokens`**: How many tokens your input prompt (plus hidden chat template tokens) consumed.
* **`usage.completion_tokens`**: How many tokens the LLM generated in its response.
* **`usage.total_tokens`**: The sum of input and output tokens:
  $$\text{total\_tokens} = \text{prompt\_tokens} + \text{completion\_tokens}$$
* **`response.choices[0].finish_reason`**: Tells you **why** the model stopped generating text:
  * **`"stop"`**: The model finished its answer naturally before reaching `max_tokens`.
  * **`"length"`**: The response was forcefully cut off mid-generation because it hit the `max_tokens` ceiling.

---

## 3. Experimenting with Token Limits

To see `finish_reason` change in action, modify `max_tokens` in `client.chat.completions.create(...)` and re-run `python tokens.py`:

| `max_tokens` Setting | `prompt1` (`"Hi!"`) | `prompt2` (`< 100 words`) | `prompt3` (`1000-word essay`) |
| :--- | :--- | :--- | :--- |
| **`max_tokens=50`** | `stop` | `length` (cut off) | `length` (cut off) |
| **`max_tokens=500`** | `stop` | `stop` | `length` (cut off) |
| **`max_tokens=5000`** | `stop` | `stop` | `stop` (completed) |