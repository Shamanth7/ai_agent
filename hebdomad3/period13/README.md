# Build Your First End-to-End RAG System!


we built a naive RAG system that broke down due to the fragility of rigid keyword matching. we discovered how **Sentence Embeddings** convert human sentences into numerical vectors that capture semantic meaning.

This module unifies both worlds to construct a **complete, end-to-end Vector-powered RAG pipeline (`full_rag.py`)** from scratch in Python. We demonstrate how semantic retrieval overcomes vocabulary mismatches (e.g., matching *"vacation"* to *"paid leave"*), how retrieved context is injected into a Groq-hosted LLM, and why local in-memory linear search inevitably demands dedicated **Vector Databases (like Qdrant)** at scale.

---

## 1. System Architecture & Workflow

A production RAG pipeline consists of two distinct stages: **Offline Knowledge Ingestion** and **Online Retrieval & Augmented Generation**.

```text
========================================================================================
1. KNOWLEDGE BASE INGESTION (Offline / Pre-computation)
========================================================================================
Company Policy Documents (5 Entries)
        │
        ▼
[SentenceTransformer: all-MiniLM-L6-v2] ──► Generates 384-dimensional dense vectors
        │
        ▼
Stored as `document_embeddings` in memory (NumPy ndarray of shape (5, 384))

========================================================================================
2. RUNTIME RETRIEVAL & AUGMENTED GENERATION (Online Query)
========================================================================================
User Query: "How much vacation do I get?"
        │
        ▼
[SentenceTransformer] ──────────────────► Query Vector (`qembedding`, shape (384,))
        │
        ▼
Compute Cosine Similarity against all `document_embeddings`
        │
        ▼
Sort Scores Descending & Select Top-1 Context Match:
"Employees receive 24 days of paid leave per year."
        │
        ▼
System Prompt Augmentation:
"answer in one line only. Answer only based on this context. do not hallucinate.
 Context: Employees receive 24 days of paid leave per year."
        │
        ▼
[Groq LLM: openai/gpt-oss-120b]
        │
        ▼
Final Grounded Answer:
"You get 24 days of paid vacation per year."
```

---

## 2. Core Concepts & Breakthroughs

### 1. Overcoming the Keyword Bottleneck
- In Episode 12, querying *"How much vacation do I get?"* failed because the knowledge base used the phrase *"paid leave"* instead of *"vacation"*.
- In this vector-based system, the embedding model recognizes that **"vacation"** and **"paid leave"** inhabit the same semantic neighborhood in high-dimensional space.
- The retrieval function correctly assigns the highest cosine similarity score to the paid leave document, completely bridging the vocabulary gap.

### 2. Separation of Models (Two-Model Architecture)
A common point of confusion is conflating the two models in a RAG pipeline:
- **Embedding Model (`model`)**: `all-MiniLM-L6-v2` running locally on CPU. Its sole responsibility is converting text strings into mathematical arrays (vectors). It never generates conversational answers.
- **Generative LLM (`groqmodel`)**: `openai/gpt-oss-120b` running via the Groq cloud API. Its job is reading the retrieved context and synthesizing a natural-language response.

---

## 3. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast virtual environment and package management.

### Step 1: Initialize Workspace & Virtual Environment

```bash
# Navigate to your workspace directory
cd hebdomad3

# Initialize project folder (no spaces in folder names)
uv init period13
cd period13

# Create Python virtual environment
uv venv 

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Required Dependencies

```bash
uv add groq python-dotenv sentence-transformers numpy
```

- `groq`: Client SDK to access fast cloud LLMs.
- `sentence-transformers`: Local inference engine for generating dense vector embeddings.
- `numpy`: Fast mathematical array operations and cosine similarity dot products.
- `python-dotenv`: Environment variable management for loading `GROQ_API_KEY`.

---

## 4. Implementation Code (`full_rag.py`)

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
groqmodel = "openai/gpt-oss-120b"

documents = [
    "Employees receive 24 days of paid leave per year.",
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
    "Employees receive Rs 3000 per month for gym reimbursement.",
    "Employees can claim Rs 2000 per month for home internet.",
    "Employees have a 90 day notice period.",
]

document_embeddings = model.encode(documents)

print(sys.getsizeof(document_embeddings))


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(qembedding):
    scores = []  # 0.4
    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(qembedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]  # line#0.9


def ask_llm(question, context):
    sys_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
    system_message = {"role": "system", "content": sys_prompt}
    message = {"role": "user", "content": question}
    messages = [system_message, message]
    response = client.chat.completions.create(model=groqmodel, messages=messages)
    answer = response.choices[0].message.content
    return answer


query = "How much vacation do I get?"
qembedding = model.encode(query)
score, context = retrieve(qembedding)
answer = ask_llm(query, context)
print(answer)
```

### Terminal Output

```text
(period13) shamanth@yoga6:~/Documents/Default Project/AI_agents/hebdomad3/period13$ python -m full_rag
Loading weights: 100%|████████████████████████████████████████| 103/103 [00:00<00:00, 3988.12it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
128
You get 24 days of paid vacation per year.
```

---

## 5. Deep-Dive Code Breakdown

Let's dissect every section of `full_rag.py` at an architectural, mathematical, and Python runtime level.

### Block 1: Imports & Model Initialization

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384
```

- **Local Transformer Loading**: `SentenceTransformer("all-MiniLM-L6-v2")` downloads (on first run) and loads a lightweight 22M-parameter transformer model into local CPU/RAM.
- **Why `# 384` matters**: This model maps any input sentence into a fixed-length dense vector of **384 floating-point numbers** ($\mathbb{R}^{384}$). Every document and every query must be embedded by this **exact same model** so that their coordinates share the same geometric space.
- **HF Hub Warning**: The terminal message `"Warning: You are sending unauthenticated requests to the HF Hub..."` appears when `SentenceTransformer` checks Hugging Face Hub for model weights without an optional `HF_TOKEN`. It is harmless and does not block local execution.

---

### Block 2: Environment Security & Groq Client Setup

```python
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
groqmodel = "openai/gpt-oss-120b"
```

- **Fail-Fast Validation**: `load_dotenv()` reads key-value pairs from `.env` into `os.environ`. Checking `if not my_api_key:` immediately halts execution before wasting local CPU cycles encoding documents if the API key is missing.
- **Inference Separation**: `client` establishes an authenticated HTTP client pointing to Groq's high-speed LPU inference API, configured to use `openai/gpt-oss-120b` for final reasoning and text synthesis.

---

### Block 3: Knowledge Base & Vector Ingestion (`document_embeddings`)

```python
documents = [
    "Employees receive 24 days of paid leave per year.",
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
    "Employees receive Rs 3000 per month for gym reimbursement.",
    "Employees can claim Rs 2000 per month for home internet.",
    "Employees have a 90 day notice period.",
]

document_embeddings = model.encode(documents)

print(sys.getsizeof(document_embeddings))
```

- **Implicit String Concatenation**: Notice how the second entry splits across two lines without a comma between them. Python automatically concatenates adjacent string literals inside a list into a single string (`"...Thursday. Monday..."`), keeping the list length at **5 documents**.
- **Batch Encoding**: Calling `model.encode(documents)` passes all 5 sentences through the neural network in a single vectorized batch, returning a 2D NumPy `ndarray` of shape `(5, 384)` and dtype `float32`.
- **Why `sys.getsizeof(document_embeddings)` prints `128`**:
  - In Python, `sys.getsizeof()` inspects only the shallow C-struct header of the `numpy.ndarray` object (`128` bytes), **not** the external C memory buffer holding the actual numbers.
  - The real memory footprint of the raw float32 matrix is calculated via `document_embeddings.nbytes`:
    $$\text{Memory} = 5 \text{ docs} \times 384 \text{ dimensions} \times 4 \text{ bytes (float32)} = 7,680 \text{ bytes} \approx 7.5\text{ KB}$$
  - At enterprise scale (e.g., 10 million chunks), that raw buffer balloons to $\approx 15.36\text{ GB}$ of RAM—one of the primary reasons production apps offload vectors to disk-backed Vector Databases.

---

### Block 4: Mathematical Similarity Engine (`cosine_similarity`)

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

This function computes the cosine of the angle $\theta$ between two 384-dimensional vectors $\mathbf{a}$ and $\mathbf{b}$:

$$\text{Cosine Similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \, \|\mathbf{b}\|_2} = \frac{\sum_{i=1}^{384} a_i b_i}{\sqrt{\sum_{i=1}^{384} a_i^2} \cdot \sqrt{\sum_{i=1}^{384} b_i^2}}$$

- **`np.dot(a, b)`**: Computes the scalar dot product (sum of element-wise multiplications). Larger positive values indicate dimensions moving in the same direction.
- **`np.linalg.norm(a) * np.linalg.norm(b)`**: Computes the Euclidean ($L_2$) magnitudes of both vectors and divides by their product. This **normalizes** the result into the bounded range $[-1, 1]$, ensuring sentence length does not skew similarity.

---

### Block 5: Brute-Force Top-1 Retriever (`retrieve`)

```python
def retrieve(qembedding):
    scores = []  # 0.4
    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(qembedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]  # line#0.9
```

1. **Iterative Scoring**: `enumerate(document_embeddings)` yields both the index `i` and the 384-D vector `document` for each policy.
2. **Tuple Packing (`(score, documents[i])`)**: Placing `score` as the **first element** of each tuple is intentional. When Python executes `scores.sort(reverse=True)`, it compares tuples lexicographically starting at index `0` (`score`), ordering documents from highest similarity to lowest.
3. **Top-1 Selection (`return scores[0]`)**: Returns the single highest-scoring `(score, text)` tuple.
   - *Subtle edge-case note*: If two documents ever had an identical floating-point `score`, Python would break ties by comparing `documents[i]` alphabetically.

---

### Block 6: Grounded Generation (`ask_llm`)

```python
def ask_llm(question, context):
    sys_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
    system_message = {"role": "system", "content": sys_prompt}
    message = {"role": "user", "content": question}
    messages = [system_message, message]
    response = client.chat.completions.create(model=groqmodel, messages=messages)
    answer = response.choices[0].message.content
    return answer
```

- **Prompt Engineering Guardrails**:
  - `answer in one line only.` enforces concise output formatting.
  - `Answer only based on this context. do not hallucinate.` overrides the LLM's default tendency to rely on pre-trained general knowledge, forcing strict **extractive/grounded synthesis**.
- **Role Separation**: Placing the retrieved `context` inside the `system` role establishes authoritative operational boundaries, while the `user` role carries the raw `question`.

---

### Block 7: End-to-End Execution Pipeline

```python
query = "How much vacation do I get?"
qembedding = model.encode(query)
score, context = retrieve(qembedding)
answer = ask_llm(query, context)
print(answer)
```

Tracing the exact runtime data flow:
1. `query` (`"How much vacation do I get?"`) is encoded into `qembedding` (`shape: (384,)`).
2. `retrieve(qembedding)` scores all 5 policy vectors. Even though `"vacation"` never appears in `documents`, its semantic vector aligns closely with `"paid leave"`, ranking `"Employees receive 24 days of paid leave per year."` at `scores[0]`.
3. Tuple unpacking (`score, context = retrieve(...)`) separates the float similarity score from the raw policy string.
4. `ask_llm(query, context)` sends both to Groq, synthesizing the human-friendly, grounded response:
   > `"You get 24 days of paid vacation per year."`

---

## 6. Scaling Bottlenecks & Production Improvements

### 1. The Linear Search Latency Bottleneck ($O(N)$)
- In `full_rag.py`, `retrieve()` performs a brute-force linear scan across every stored vector:
  ```python
  for i, document in enumerate(document_embeddings):
      score = cosine_similarity(qembedding, document)
  ```
- Searching **5 vectors** takes $<1\text{ ms}$.
- Searching **60 million vectors** linearly on every query requires 60 million dot products per user request, causing massive latency spikes and exhausting RAM.
- **The Solution (Vector Databases like Qdrant, Pinecone, ChromaDB)**: Persist vectors on disk and index them using **HNSW (Hierarchical Navigable Small World)** graphs for sub-millisecond Approximate Nearest Neighbor (ANN) search.

### 2. Missing Similarity Thresholding (Out-of-Domain Queries)
- Because `retrieve()` unconditionally returns `scores[0]`, asking an unrelated question like `"What is the recipe for biryani?"` will still return whichever company policy happens to have the highest score (e.g., `0.08` for gym reimbursement) and pass it to the LLM.
- **Production Fix**: Enforce a minimum confidence threshold (e.g., `if score < 0.25: return None`) so the system safely responds *"I don't have information about that"* when no relevant context exists.

### 3. Top-1 vs. Top-$K$ Retrieval
- Returning only `scores[0]` works when an answer lives inside a single sentence. When answers span multiple policy clauses (e.g., combining leave policy + work-from-home policy), production systems retrieve **Top-$K$ chunks** (`scores[:3]`) and join them into the context block.

---

## 7. Key Takeaways & What's Next

1. **RAG = Retrieval + Augmented Generation**:
   - Retrieval isolates the needles in the haystack via dense vector similarity.
   - The LLM reads only the retrieved needles to formulate a grounded answer.
2. **Semantic Understanding**: Vectors bridge vocabulary gaps (`"vacation"` $\leftrightarrow$ `"paid leave"`), ensuring synonyms and paraphrased questions match the correct source documents.
3. **Next Step**: **Episode 15** replaces manual in-memory NumPy arrays and $O(N)$ linear loops with the **Qdrant Vector Database**.