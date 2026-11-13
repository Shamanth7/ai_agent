# Structured Outputs with Pydantic & JSON

In production systems, Large Language Models rarely interact solely with human end-users. In over 90% of real-world enterprise architectures, the output of an AI agent is consumed directly by downstream software services (written in Python, Go, Java, C++, etc.) that execute millions of times a day.

This module transitions from conversational text generation to **Structured Outputs**, showing how to enforce strict schemas using **Pydantic `BaseModel`**, request a `json_object` from the LLM, and parse responses reliably into strongly typed Python objects.

This guide explains how to enforce **Structured Outputs** from Large Language Models using **Pydantic** and **JSON**. Instead of receiving unpredictable conversational paragraphs that break downstream code, you learn how to force the LLM to return clean, machine-readable JSON strictly matching a defined schema.

---

## Why Structured Outputs Matter

When humans interact with an LLM, plain text paragraphs are easy to read. However, in real-world software systems (where an AI agent processes millions of requests and passes data to other Python, Java, C++, or JavaScript services), **unstructured strings are a nightmare to parse**:

* **Unstructured Output (Hard for Code):**
  > *"Sure! Here is the customer information: **Name:** Shamanth, **Email:** abc@gmail.com..."*
  Downstream code must write fragile string-matching logic to strip conversational filler and Markdown formatting.
* **Structured JSON Output (Easy for Code):**
  ```json
  {
    "name": "shamanth",
    "email": "abc@gmail.com",
    "issue": "iphone not working at all"
  }
  ```
  Downstream services can immediately parse this into native objects/variables (`ticket.name`, `ticket.email`, `ticket.issue`) with zero guesswork.

---
---

## 1. Project Overview & Key Objectives

- **The Problem with Raw Strings**: Why conversational preambles (e.g., *"Sure, I'd be happy to help!"*), markdown formatting (`**Name:**`), and free-form paragraphs break downstream code.
- **The Watchman Analogy**: Understanding unstructured narrative logging versus structured key-value recording.
- **Data Exchange via JSON**: Why JavaScript Object Notation is the universal lingua franca for inter-service communication.
- **Data Contracts with Pydantic**:
  - Defining schemas using `BaseModel`.
  - Enforcing data types (`str`, `int`, etc.).
  - Generating schemas programmatically via `.model_json_schema()`.
- **API Response Formatting**:
  - Enforcing `response_format={"type": "json_object"}`.
  - Resolving the mandatory API rule: including the word `"json"` in prompt messages.
- **End-to-End Deserialization**:
  - Parsing raw strings with `json.loads()`.
  - Reconstructing validated Pydantic instances using dictionary unpacking (`**data`).
- **Mini-Project Homework**: Designing an HR Resume Matcher & Parser.

---

## 2. Core Concepts & Architecture

### The String Parsing Bottleneck

When you prompt an LLM to extract customer support data without structural constraints, it often responds conversationally:

```text
Sure! Here is the extracted customer information:
- Name: shamanth
- Email: abc@gmail.com
- Phone: 82155...
- Issue: Electronics (iPhone stopped working)
```

For human eyes, this is readable. For automated software, string parsing is notoriously brittle:
1. Regex and string-splitting rules fail when the model changes formatting or adds bullet points.
2. Downstream services require exact variables without prose wrappers.
3. Irrelevant input noise (e.g., customer complaints containing personal stories or unrelated remarks) must be filtered out deterministically.

---

### The Watchman Analogy: Unstructured vs. Structured

Imagine a security guard stationed at an apartment gate tasked with recording visiting vehicles:

- **Unstructured Entry (Free text)**:
  > *"Pratyush ji arrived at 2:00 PM driving a silver car with license plate DL-XE-7338."*  
  *(Requires complex NLP parsing; difficult to query or store in a database).*

- **Structured Entry (JSON)**:
  ```json
  {
    "name": "shamanth",
    "plate_number": "DL-XE-7338"
  }
  ```
  *(Directly indexable, machine-readable, and unambiguous).*

---

### Pydantic `BaseModel` as a Strict Schema Contract

Pydantic allows Python developers to declare strict data schemas. When fed to an LLM via its system prompt:
1. **Field Whitelisting**: Only fields declared in the model (e.g., `name`, `email`, `issue`) are extracted.
2. **Noise Discarding**: Extraneous data in the prompt (e.g., `"My address is banglore"`, personal stories) is discarded because it is not defined in the schema.
3. **Type Safety**: Fields are typed (`str`, `int`, etc.), eliminating type-casting bugs downstream.

```text
+-------------------------------------------------------------+
| Unstructured Input Email                                    |
| ("Hello, my name is Pratyush. Bought iPhone, stopped working|
|  Address is Delhi, Broke up with Sheetal, email abc@...")   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| LLM with Pydantic Schema Injection                          |
| System Prompt: Strictly follow JSON schema for Ticket        |
| response_format: {"type": "json_object"}                    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Validated JSON Output                                       |
| {"name": "Pratyush", "email": "abc@...", "issue": "iPhone"} |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Python Typed Object (ticket.name, ticket.email)             |
+-------------------------------------------------------------+
```

---

## 3. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast dependency and virtual environment management.

### Step 1: Initialize Project Directory

```bash
# Navigate to your workspace directory
cd hebdomad

# Initialize project folder
uv init period4
cd period4

# Create virtual environment
uv venv 

# Activate virtual environment
source .venv/bin/activate
```

### Step 2: Install Dependencies & Configure `.env`

```bash
uv add groq python-dotenv pydantic
```


---


---

## 5. Key Implementation Rules & Common Pitfalls

1. **Inherit from `BaseModel`**: Your schema class (`class Ticket(BaseModel):`) must inherit from Pydantic's `BaseModel`; otherwise, calling `Ticket.model_json_schema()` will raise an `AttributeError`.
2. **Mandatory `"json"` Keyword in Prompt**: Whenever you pass `response_format={"type": "json_object"}` to the API, your `messages` (either the `system` or `user` prompt) **must explicitly include the word `"json"`**. Without it, the API rejects the request with a validation error.
3. **Deserialization Pipeline**:
   - `response.choices[0].message.content` returns a **JSON-formatted string**, not a Python dictionary.
   - Use `json.loads(raw_json)` to convert the string into a Python `dict`.
   - Use dictionary unpacking `Ticket(**data_file)` to instantiate a strongly typed Pydantic object.

---


---

##  Complete Implementation (`json_pydantic.py`)

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


# Structure it using Pydantic
from pydantic import BaseModel

class Ticket(BaseModel):
    name: str
    email: str
    issue: str

schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""

message_system = {
    "role": "system",
    "content": system_prompt
}

text = "Hello My name is shamanth. Yesterday I broke up with my girlfriend sheetal I have an iphone which is not working at all. My address is delhi. My email is abc@gmail.com. My contact number is 82134"
prompt = f"""
This is a customer ticket. Please extract the personal information from this.
{text}
"""

# Message with role and content
message = {
    "role": role,
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

answer = response.choices[0].message.content
print(answer)


# How to read and parse the JSON response into Python objects
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)


# Pass structured fields forward to downstream logic!
print(ticket.name)
print(ticket.email)
print(ticket.issue)
```

---

## 4. Step-by-Step Code Walkthrough

### Step 1: API Client Initialization
```python
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"
```
* Loads `GROQ_API_KEY` from `.env`, validates that it exists, and initializes the Groq client targeting the `openai/gpt-oss-120b` model.

### Step 2: Defining the Data Schema with Pydantic
```python
from pydantic import BaseModel

class Ticket(BaseModel):
    name: str
    email: str
    issue: str

schema = Ticket.model_json_schema()
```
* **`BaseModel`**: By inheriting from Pydantic's `BaseModel`, `Ticket` defines the exact blueprint (fields and data types) you want extracted.
* **`Ticket.model_json_schema()`**: Automatically converts your Python class into a formal JSON Schema dictionary that can be injected directly into the LLM prompt so the model knows the exact keys and types expected.
* **Noise Filtering**: Notice that the customer message contains extra irrelevant details (girlfriend's name *"sheetal"*, address *"delhi"*, contact number *"82134"*). Because those fields are **not** in `Ticket`, the LLM ignores them.

### Step 3: Enforcing JSON Output Mode
```python
response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""
```
* **`response_format={"type": "json_object"}`**: Instructs the Groq API at the decoding level to guarantee valid JSON syntax.
* **Mandatory `"json"` Keyword Rule**: When using `response_format={"type": "json_object"}`, your prompt (`system_prompt` or `prompt`) **must explicitly contain the word `"json"`**; otherwise, the API will throw a validation error.

### Step 4: Constructing Messages & Calling the LLM
```python
message_system = {"role": "system", "content": system_prompt}
message = {"role": role, "content": prompt}
messages = [message_system, message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)
answer = response.choices[0].message.content
```
* Sends both the `system` instructions (containing the schema) and the `user` message (containing the noisy customer complaint) to the model, capturing the raw JSON string in `answer`.

### Step 5: Deserializing JSON into a Validated Pydantic Object
```python
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)
```
* **`json.loads(raw_json)`**: Converts the raw JSON string returned by the API into a standard Python dictionary (`data_file`).
* **`Ticket(**data_file)`**: Unpacks the dictionary keys/values into the `Ticket` Pydantic model, validating types and giving you clean dot-notation access (`ticket.name`, `ticket.email`, `ticket.issue`) ready to pass to databases, APIs, or business logic.

---

## 5. Expected Output

When you run `python json_pydantic.py`, you will see the raw JSON string followed by the cleanly extracted attributes:

```text
{"name": "shamanth", "email": "abc@gmail.com", "issue": "iphone is not working at all"}
shamanth
abc@gmail.com
iphone is not working at all
```