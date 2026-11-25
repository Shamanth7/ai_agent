# What is RAG? (Retrieval-Augmented Generation)

This module introduces one of the most critical and heavily interviewed concepts in modern AI engineering: **RAG (Retrieval-Augmented Generation)**. Rather than jumping straight into complex third-party libraries, this project builds RAG from its foundational first principles—exploring why LLMs fail on private data, deconstructing the early attempts to solve context delivery, and implementing the first iteration of a RAG pipeline from scratch in Python to uncover why Vector Databases and Embeddings became necessary.

---

## 1. Project Overview & The Core Dilemma

Large Language Models (LLMs) such as ChatGPT, Claude, and LLaMA are trained on massive corpora of public internet data. However, in enterprise environments, standard LLMs encounter two major failure modes:

1. **The Private Data Blindspot**:
   - Asking an LLM for public metrics (e.g., *Padho with Pratyush's* YouTube subscriber count) produces immediate, accurate answers.
   - Asking for private, internal, or non-public information (e.g., personal details, internal company architecture, unreleased codebases) fails because the model was never trained on that data.
2. **Confident Hallucination**:
   - LLMs are probabilistic token predictors that rarely say *"I don't know"*. Instead, they hallucinate plausible-sounding guesses (e.g., guessing someone's favorite brand with false confidence).
   - In 2022, when companies attempted to plug ChatGPT directly into their private systems (e.g., Akamai or Amazon internal debugging), the models hallucinated incorrect fixes because the internal service dependencies exist only inside private enterprise networks, not on the public internet.

**RAG is the architectural solution to this problem.**

---

## 2. Core Concepts: What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**:

```text
+-----------------------------------------------------------------------------------+
| USER QUERY: "What is Shamanth's age?"                                             |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 1. RETRIEVAL (R)                                                                  |
| Specialized software searches an internal Knowledge Base and extracts ONLY the    |
| relevant chunk of data (e.g., "The age of Shamanth is 22 years").                 |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. AUGMENTED (A)                                                                  |
| The prompt sent to the LLM is augmented with this retrieved chunk as Context:     |
| System: "Answer only based on this context: [The age of Shamanth is 22 years]"    |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. GENERATION (G)                                                                 |
| The LLM reads the provided context and generates a grounded, factual answer:      |
| "Shamanth is 22 years old."                                                       |
+-----------------------------------------------------------------------------------+
```

- **Open-Book Exam Mental Model**:
  - A base LLM is taking a **closed-book exam** relying solely on memory.
  - RAG turns the task into an **open-book exam**: you retrieve the exact page or paragraph needed and place it in front of the model to synthesize the final answer.

---

## 3. The Evolution: Why Alternative Approaches Failed

Before settling on RAG pipelines, developers attempted two intuitive workarounds that proved impractical at scale:

### Attempt 1: "Put Everything in the System Prompt"
- **Idea**: Feed private knowledge directly into the system prompt.
- **Why it failed**: You can describe a user's name or GPA in a system prompt, but you cannot explain Google's, Amazon's, or Akamai's entire multi-service distributed architecture in 10 lines of text.

### Attempt 2: "Upload Massive PDFs / Full Knowledge Bases Every Call"
- **Idea**: Convert all company documentation into files and feed them to the LLM with every request.
- **Why it failed**:
  - **Token Limits & Context Ceiling**: Real enterprise documentation spans 20+ years across software design docs, API specs, deployment playbooks, and testing logs (millions of lines). Uploading entire archives exceeds context windows (triggering *"Prompt Too Long"* errors).
  - **Astronomical API Costs**: Passing hundreds of thousands of prompt tokens on every query burns budgets rapidly (e.g., multi-crore monthly API bills).
  - **High Latency**: Ingesting massive contexts on every call slows down response times significantly.

---

## 4. The First Iteration of RAG & Its Breaking Points

The original, naive iteration of RAG used **keyword/string matching** to fetch relevant snippets from a local store before querying the model. While conceptually functional, it breaks under real-world usage:

| User Query | RAG Iteration 1 Outcome | Root Failure Cause |
| :--- | :--- | :--- |
| `"What is Shamanth's age?"` | ✅ `"Shamanth is 22 years old."` | Exact keyword `"age"` matches key in knowledge base. |
| `"How old is Shamanth?"` | ❌ *"There is no information provided..."* | **Synonym Blindness**: The word `"age"` is absent from the query. Rigid string matching fails to capture semantic equivalence. |
| `"What is Shamanth's net worth?"` | ✅ `"Shamanth's net worth is 5000."` | Exact keyword `"net worth"` matches key in knowledge base. |
| `"How rich is Shamanth?"` | ❌ *"Impossible to determine wealth..."* | The concept of *wealth/rich* is semantically identical to *net worth*, but string matching cannot interpret semantics. |
| `"What is age?"` | ❌ Dumps Shamanth's personal age data | **False Positive Trigger**: Any query containing the keyword `"age"` blindly triggers the retrieval regardless of relevance. |
| `"What is Shamanth's  networth?"` *(typo/missing space)* | ❌ *"No information provided..."* | **Typo Fragility**: A minor spelling error or whitespace shift causes the keyword lookup to miss completely. |

### Why This Sparked Modern RAG:
You cannot write infinite `if/else` ladders or maintain exhaustive regex synonym dictionaries for every concept in the English language. This brittle foundation directly led to the adoption of:
- **Embeddings** (converting text into semantic vectors).
- **Vector Databases** (e.g., Qdrant, ChromaDB, Pinecone) for high-dimensional similarity search.
- **Chunking strategies** for granular document segmentation.

---

## 5. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, isolated Python dependency and virtual environment management.

### Step 1: Initialize Workspace & Virtual Environment

```bash
# Navigate to workspace
cd hebdomad3

# Initialize day11-rag project folder
uv init rag
cd rag

# Create Python 3.11 virtual environment
uv venv 

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
uv add groq python-dotenv
```


---

## 3. Implementation Code (`rag_v1.py`)

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


# Step 1: Private Knowledge Base
knowledge_base = {
    "age": "The age of shamanth is 22 years",
    "net worth": "The net worth of shamanth is 5000",
}


# Step 2: Retrieval (Keyword Matching)
def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]
    else:
        return None


# Step 3: Augmentation & Generation
def ask_llm(question):
    context = retrieve_info(question)

    sys_prompt = (
        f"answer in one line only. Answer only based on this context. "
        f"do not hallucinate. Context: {context}"
    )
    system_message = {
        "role": "system",
        "content": sys_prompt,
    }
    message = {
        "role": "user",
        "content": question,
    }
    messages = [system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer


question = "what is shamanth age?"
print(ask_llm(question))
```

### How the Code Works

1. **`knowledge_base` (Data Store)**: A Python dictionary simulating an internal database holding private facts that the pretrained LLM (`openai/gpt-oss-120b`) has never seen.
2. **`retrieve_info(question)` (Retrieval Step)**: Normalizes the user query with `.lower()` and checks for exact substrings (`"age"` or `"net worth"`). If nothing matches, it returns `None`.
3. **`ask_llm(question)` (Augmentation + Generation)**:
   - Injects the retrieved `context` directly into `sys_prompt` alongside strict grounding rules (`"Answer only based on this context. do not hallucinate."`).
   - When `question = "what is shamanth age?"` runs, `context` resolves to `"The age of shamanth is 22 years"`, and the LLM returns a grounded one-line answer.
   - When an out-of-scope or synonym query like `"how old is shamanth?"` runs, `context` becomes `None`, preventing the model from guessing.