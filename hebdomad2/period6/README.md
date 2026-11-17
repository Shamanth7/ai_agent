# EPrompt Engineering  (The 6-Pillar Framework)


This module demystifies **Prompt Engineering**. Rather than treating prompt engineering as an overly complex specialty requiring expensive degrees or standalone courses, this guide breaks it down into practical engineering logic: transforming ambiguous, non-deterministic prompts into robust, stable, and production-ready constraints.

---

## 1. Why Prompt Engineering Matters in Production

### The Core Problems with Raw LLMs

1. **Non-Deterministic Behavior**
   - Unlike a calculator where `2 + 2 = 4` is a deterministic constant, asking an LLM the same question across multiple runs produces varying responses in structure, punctuation, and phrasing.
   - In production pipelines, LLM outputs are rarely read by humans; they are consumed by downstream code functions and microservices. If the receiving service expects a specific category label and receives an essay or conversational preamble, the application crashes.

2. **Unbounded Model Scope & Edge Cases**
   - A customer support chatbot deployed for an e-commerce or food delivery app (e.g., Zomato, Domino's, or an electronics store) must only handle domain-specific requests (e.g., order tracking, cold food, delivery delays, defective hardware).
   - Without explicit boundaries, users can submit out-of-domain requests—such as relationship breakup advice, requests for staff phone numbers, or free Python code generation—and an unconstrained model will attempt to answer them instead of refusing or falling back safely.

---

## 2. The 6-Pillar Prompt Engineering Framework

To turn a vague ("bad") prompt into an engineered, secure, and predictable prompt, incorporate these six foundational components:

```text
+--------------------------------------------------------------------------+
| 6 PILLARS OF PROMPT DESIGN                                               |
+--------------------------------------------------------------------------+
| 1. ROLE          -> Defines identity, domain, and operational scope      |
| 2. TASK          -> Explicit action verb (e.g., classify, extract)       |
| 3. CONSTRAINTS   -> Fixed boundaries (e.g., allowed buckets only)        |
| 4. OUTPUT FORMAT -> Deterministic structure (e.g., single word, JSON)    |
| 5. EXAMPLES      -> Demonstrations (Zero-shot, One-shot, Few-shot)       |
| 6. FALLBACK      -> Graceful handling of out-of-scope inputs (Murphy's)  |
+--------------------------------------------------------------------------+
```

### 1. Role (`Who are you?`)
- Specifies domain expertise and functional responsibilities (e.g., *"You are a support assistant at a mobile and laptop company"*).
- **Anti-pattern**: Avoid writing inflated labels like *"You are a genius engineer"* expecting the model's fundamental IQ to increase. Ground roles in concrete operational contexts.

### 2. Task (`What must you do?`)
- Assign a clear, unambiguous action (e.g., *"Classify the user complaint into a category"*).
- Avoid vague verbs like `"Handle this"`.

### 3. Constraints (`What are the boundaries?`)
- Hard-code boundaries so the model does not invent custom categories.
- **Example**: Restrict classification exclusively to:
  - `Billing`
  - `Technical`
  - `Return`

### 4. Output Format (`How should the result look?`)
- Strip conversational padding, markdown asterisks, and preamble.
- **Example**: *"Your answer should be in one word only, matching one of the categories given in Constraints."*

### 5. Examples / Shots (`Show, Don't Just Tell`)
- Just as humans understand code algorithms better with a dry run, LLMs mirror the exact pattern and format of concrete examples.
- **Zero-shot**: Providing instructions without examples.
- **One-shot**: Providing a single concrete demonstration (e.g., `If user wants refund -> Return`).
- **Few-shot**: Providing multiple examples covering tricky or borderline cases.

### 6. Fallback (`Handling Murphy's Law`)
- In software, *Murphy's Law* dictates that anything that can go wrong will go wrong. You cannot assume inputs will always follow the happy path.
- If a customer sends an unrelated or personal issue (e.g., *"My girlfriend left me"* or *"My marriage is broken"*), a model forced into standard constraints might misclassify it as `Billing` or `Technical`.
- An explicit fallback instructs the model: *"If the issue is unrelated to any of the categories mentioned in Constraints, output 'Other'."*

---

## 3. Evolution: Bad Prompt vs. Engineered Prompt

### The Bad Prompt

```text
This is a user complaint:
My laptop is not working.
Handle this.
```

**Failure Modes:**
- The model writes multi-paragraph explanations and diagnostic steps rather than an automated classification.
- Run repeatedly, the wording and formatting shift unpredictably.
- Submitting out-of-domain inputs produces unsolicited advice or boundary violations.

---

### The Fully Engineered Prompt

```text
ROLE:
You are a support assistant at a mobile and laptop company.

TASK:
You have to classify the customer issue into a category.

CONSTRAINTS:
You have to classify the issue into one of the three categories:
- Billing
- Technical
- Return

OUTPUT FORMAT:
Your answer must be in ONE WORD ONLY.
The one word must be one of the categories provided in Constraints.

EXAMPLE:
For instance: If customer says "I want a refund", category is "Return".

FALLBACK:
If the issue is completely unrelated to any of the categories mentioned in Constraints, your answer must be "Other".
```

### Code Example

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


def llm_ans(prompt):
    message = {
        "role": "user",
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages)
    ans = response.choices[0].message.content
    return ans


bad_prompt = """
#ROLE:
You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of three categories namely billing, technical, return.
#OUTPUT FORMAT
Your answer should be in one word only. The one word shoud be one of the categories given in constraints
#Example
For instance if a user compalin says he wants a refund then the category is Return
#FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER
This is a user complaint:
My marriage is broke
"""

print(llm_ans(bad_prompt))
```
### once see
```text
if in the :- This is a user complaint:
My marriage is broke
if i add :- my laptop is not working (in this paragraph only)
the output will be --> techincal
```
### How the Program Works

1. **Environment & Security Setup (`load_dotenv` & `os.getenv`)**
   * Loads environment variables from a local `.env` file so sensitive credentials like `GROQ_API_KEY` are never hardcoded in source control.
   * Performs a fail-fast validation check (`if not my_api_key: raise ValueError(...)`) before initializing the API client.

2. **Groq Client & Model Configuration**
   * Initializes the `Groq` client with the authenticated API key and targets the `openai/gpt-oss-120b` model for inference.

3. **The Inference Wrapper (`llm_ans`)**
   * Wraps the input string into the standard Chat Completions message structure (`[{"role": "user", "content": prompt}]`).
   * Calls `client.chat.completions.create(...)` and extracts the generated text via `response.choices[0].message.content`.

4. **Testing the 6-Pillar Prompt & Edge Case (`bad_prompt`)**
   * Despite the variable name `bad_prompt`, the string actually implements all **6 Pillars** (`#ROLE`, `#TASK`, `#CONSTRAINT`, `#OUTPUT FORMAT`, `#Example`, and `#FALLBACK`).
   * **The Edge Case Test:** The user input is `"My marriage is broke"`—an out-of-domain personal problem unrelated to laptops or mobile devices.
   * **Expected Output:** `OTHER`
   * **Why It Succeeds:** Because of the `#FALLBACK` and `#OUTPUT FORMAT` pillars, the model resists forcing the input into `billing`, `technical`, or `return`, avoids giving unsolicited relationship advice, and outputs a clean, single-word `OTHER` that downstream code can safely handle.