# Qdrant Vector Database (Production RAG Architecture)

we built an end-to-end RAG system using in-memory Python arrays and linear cosine similarity loops. While that demonstrated the mechanics of vector retrieval, it hit severe architectural walls in terms of **Time ($O(N)$ linear scan)**, **Persistence (RAM volatility)**, and **Memory (RAM limits)**.

This module replaces fragile in-memory dictionaries with **Qdrant**, an open-source, enterprise-grade **Vector Database**. We explore how vector databases translate relational database paradigms (Tables vs. Collections, Rows vs. Points), implement vector indexing with Cosine distance, and build a persistent, scalable RAG pipeline connecting Qdrant Cloud with Groq LLMs.

---

## 1. Project Overview & Key Objectives

* **Why Vector Databases? (The 3 Fatal Flaws of In-Memory RAG)**:
  * **Latency / Time**: Why linear search ($O(N)$) fails on 10,000+ to 1,000,000+ vectors (waiting 30+ seconds per query).
  * **Persistence**: Why power outages or process restarts destroy ephemeral RAM vectors, forcing expensive re-embedding.
  * **RAM Exhaustion**: Why scaling to 1M+ embeddings crashes consumer hardware (8–16 GB RAM).
- **Relational SQL vs. Vector Database Mapping**:
  * `Database Table` $\longleftrightarrow$ `Collection`
  * `Table Row` $\longleftrightarrow$ `Point`
  * The 3 elements of a Point: `id`, `vector`, and `payload`.
* **The Critical Role of Payloads**: Why storing the raw source text alongside the embedding vector is mandatory for LLM synthesis.
* **Hands-On Pipeline**:
  * Setting up Qdrant Cloud cluster credentials.
  * Creating collections with `VectorParams` and `Distance.COSINE`.
  * Batching and upserting points via `client.upsert`.
  - Querying nearest neighbors via `client.query_points`.
  * Passing retrieved payloads as grounded context into a Groq LLM (`openai/gpt-oss-120b`).

---

## 2. Core Concepts: Why Vector DB?

### The 3 Core Problems of In-Memory Python Search

```text
+-----------------------------------------------------------------------------------+
| 1. SEARCH LATENCY (O(N) Linear Search)                                            |
|    - 6 lines: ~0.001 ms                                                           |
|    - 10,000 lines: ~1 second                                                      |
|    - 1,000,000 lines: ~30+ seconds per user query (unacceptable for production UX)|
+-----------------------------------------------------------------------------------+
| 2. VOLATILITY & PERSISTENCE                                                       |
|    - Vectors live only in Python script RAM.                                      |
|    - Power failure, crash, or reboot destroys all vectors.                        |
|    - Re-embedding 1 million lines can take hours of compute time.                 |
+-----------------------------------------------------------------------------------+
| 3. RAM / MEMORY EXHAUSTION                                                        |
|    - 6 vectors = ~7.6 KB                                                          |
|    - 1,000,000 vectors = ~7+ GB RAM                                               |
|    - A standard 8 GB/16 GB developer machine exhausts its memory and crashes.     |
+-----------------------------------------------------------------------------------+
```

### Relational Database (SQL) vs. Vector Database (Qdrant)

If you already know traditional relational databases (PostgreSQL, MySQL), understanding vector databases is a direct mental translation:

| Relational Database (SQL) | Vector Database (Qdrant) | Description / Role |
| :--- | :--- | :--- |
| **Database Table** | **Collection** | The overarching container holding schema configuration and records (e.g., `"knowledge"`). |
| **Row / Record** | **Point (`PointStruct`)** | A single record stored in the collection. |
| **Primary Key (`id`)** | **Point ID (`id`)** | An integer or UUID identifying the unique record (e.g., `1`, `1001`). |
| **Columns of Data** | **Vector (`vector`)** | The dense numerical array representing semantic meaning (e.g., 384 floats). |
| **Row Attributes** | **Payload (`payload`)** | Metadata dictionary storing the original human-readable text and source tags. |

```text
                   ANATOMY OF A QDRANT POINT
+-----------------------------------------------------------------------+
| PointStruct                                                           |
| ├── ID      : 1                                                       |
| ├── Vector  : [-0.0825, 0.0382, -0.0123, ... 384 dimensions]          |
| └── Payload : {                                                       |
|                 "text": "Employees receive 24 days of paid leave..."  |
|               }                                                       |
+-----------------------------------------------------------------------+
```

### Why Payload is Essential

When Qdrant runs a vector search, it calculates similarity across dense float arrays. However, **LLMs are text-based models**—they cannot read raw floats to answer a user question. By saving the original sentence inside `payload`, Qdrant returns both the vector similarity score and the human-readable text line, allowing your application to inject that text directly into the LLM system prompt.

---

## 3. End-to-End System Architecture

```text
=========================================================================================
1. OFFLINE INGESTION PIPELINE
=========================================================================================
knowledge.txt (Raw Lines)
      │
      ▼
[SentenceTransformer: all-MiniLM-L6-v2] ──► Generates 384-d Vectors
      │
      ▼
Assemble PointStruct(id, vector, payload={"text": line})
      │
      ▼
[Qdrant Cloud Collection: 'knowledge'] ──► client.upsert(points) (Stored on Disk/Cloud)

=========================================================================================
2. RUNTIME RETRIEVAL & GENERATION PIPELINE
=========================================================================================
User Query: "How many vacation days do I get?"
      │
      ▼
[SentenceTransformer] ──► Query Vector
      │
      ▼
Qdrant: client.query_points(query=q_vec, limit=3, with_payload=True)
      │
      ├── Match 1 (Score: 0.480): "Employees receive 24 days of paid leave per year"
      ├── Match 2 (Score: 0.418): "Monday and Friday are optional work-from-home days."
      └── Match 3 (Score: 0.374): "Employees have a 90 day notice period."
      │
      ▼
Combine Top-3 Payloads into Grounded Context
      │
      ▼
Augment Prompt: "Answer the question using only the information provided below..."
      │
      ▼
[Groq LLM: openai/gpt-oss-120b]
      │
      ▼
Final Answer: "You get 24 vacation days per year."
```

---

## 4. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for isolated Python virtual environment management.

### Step 1: Initialize Workspace & Virtual Environment

```bash
# Navigate to workspace
cd hebdomad3

# Initialize episode 15 project
uv init period14
cd period14

# Create virtual environment
uv venv 

# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Core Dependencies

```bash
uv add qdrant-client sentence-transformers groq python-dotenv
```

### Step 3: Configure Environment Variables

1. Go to your **Qdrant Cloud Console** and obtain your Cluster URL (Endpoint) and API Key.
2. Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
QDRANT_URL=https://your-cluster-id.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_actual_qdrant_api_key_here
```

---

## 5. Knowledge Base Setup (`knowledge.txt`)

Create a plain text file named `knowledge.txt` in your project root:

```text
Employees receive 24 days of paid leave per year.
Employees can work from home on Mondays and Fridays.
The office provides a monthly internet stipend of ₹3000.
Health insurance covers employees and immediate family members.
Core working hours are between 10:00 AM and 4:00 PM.
Annual performance appraisals take place in the month of December.
Employees have a 90-day notice period.
```

---

## 6. Complete Implementation Code (`rag_qdrant_pipeline.py`)

```python
# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq


# Load variables from .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud!")


# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384


# Delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)


# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")


# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]
# ["line 1 ", "line2", "line3"....]
print(f"Loaded {len(documents)} documents")


# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2") #384

print("Embedding model ready!")


embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")


# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

points = []

for i, embedding in enumerate(embeddings):

    point = PointStruct(
        id=i + 1, #id=1

        vector=embedding.tolist(),

        payload={
            "text": documents[i]
        }
    )

    points.append(point)


# ============================================================
# PART 7 — UPLOAD TO QDRANT
# ============================================================

client.upsert( #upload+insert
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")


# ============================================================
# PART 8 — SEARCH QDRANT
# ============================================================

def search(query, top_k=3):

    # Convert the question into an embedding
    query_vector = model.encode(query).tolist()

    # Search Qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results


# ============================================================
# PART 9 — TEST SEARCH
# ============================================================

query = "How many vacation days do I get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()


# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PART 11 — ASK THE LLM
# ============================================================

def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

question = "How many vacation days do I get?"

results = search(question, top_k=3)


# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)
```

---

## 7. Terminal Execution & Output

```bash
(period14) shamanth@yoga6:~/Documents/Default Project/AI_agents/hebdomad3/period14$ python -m rag_qdrant_pipeline
Connected to Qdrant Cloud!
Deleting existing collection: knowledge
Created collection: knowledge
Vector size: 384
Distance: COSINE
Loaded 6 documents
Loading embedding model...
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3405.78it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Embedding model ready!
Generated 6 embeddings
Embedding size: 384
Uploaded 6 documents to Qdrant!

Search results:
Score: 0.480
Employees receive 24 days of paid leave per year

Score: 0.418
Monday and Friday are optional work-from-home days.

Score: 0.374
Employees have a 90 day notice period.


Final Answer:
You get 24 vacation days per year.
```

### What the Output Proves
- **Semantic Bridging (`0.480`)**: Even though the query asked for `"vacation days"` and `knowledge.txt` only contained `"paid leave"`, Qdrant ranked that exact policy #1 (`0.480`), well ahead of secondary HR policies (`0.418` and `0.374`).
- **Context Filtering by the LLM**: All 3 retrieved chunks were passed to `openai/gpt-oss-120b`, which ignored the irrelevant WFH and notice period policies and extracted the exact grounded answer.

---

## 8. Step-by-Step Code Walkthrough

### Part 1 & Part 2: Environment Setup & Qdrant Cloud Connection

* `load_dotenv()` reads `QDRANT_URL`, `QDRANT_API_KEY`, and `GROQ_API_KEY` from your `.env` file so sensitive credentials are never hardcoded.
* `QdrantClient(url=..., api_key=...)` establishes an authenticated connection to your remote Qdrant Cloud cluster.

### Part 3: Collection Initialization (`create_collection`)

* **`EMBEDDING_SIZE = 384`**: Must match the exact output dimension of `all-MiniLM-L6-v2` (384 floats per sentence). A dimension mismatch causes Qdrant to reject uploads.
* **`if client.collection_exists(COLLECTION_NAME):`**: Ensures a clean slate during development by deleting any older version of `"knowledge"` before recreating it.
* **`Distance.COSINE`**: Configures Qdrant's HNSW index to compare vectors by angular similarity rather than raw Euclidean length.

### Part 4 & Part 5: Document Ingestion & Vector Encoding

* The list comprehension `[line.strip() for line in f if line.strip()]` loads `knowledge.txt` while automatically skipping blank lines and trailing newline characters (`\n`).
* `model.encode(documents)` converts all text lines into a 2D NumPy array of shape `(N, 384)` locally on CPU.

### Part 6 & Part 7: Assembling `PointStruct` & Upserting

* **1-Based Point IDs (`id=i + 1`)**: Qdrant requires point IDs to be positive integers (`>= 1`) or UUID strings. Using `i + 1` avoids invalid `0` IDs.
* **`.tolist()` Conversion**: Qdrant's serializer expects standard Python lists of floats rather than NumPy `ndarray` objects.
* **`payload={"text": documents[i]}`**: Attaches the original human-readable sentence alongside its vector so it can be retrieved later for the LLM.
* **`client.upsert(...)`**: Uploads all points in a single batch call. *Upsert* (`update` + `insert`) inserts new IDs and overwrites existing IDs cleanly.

### Part 8 & Part 9: Semantic Vector Search (`query_points`)

* `search(query, top_k=3)` encodes the incoming question with the **exact same embedding model** (`all-MiniLM-L6-v2`) used during ingestion—mixing embedding models would place the query and documents in incompatible vector spaces.
* `client.query_points(..., limit=top_k, with_payload=True).points` performs sub-linear Approximate Nearest Neighbor (ANN) search and returns the top 3 closest points along with their `.score` and `.payload` dictionary.

### Part 10, Part 11 & Part 12: Context Augmentation & Grounded Generation

* **Multi-Chunk Context Assembly**: `"\n".join(result.payload["text"] for result in results)` merges the top 3 retrieved sentences into a single multi-line context block. Passing `top_k=3` instead of just `top_k=1` ensures answers spanning multiple policies are captured.
* **Strict Grounding Guardrail**: `ask_llm` instructs `openai/gpt-oss-120b` to answer *only* from `Context` and explicitly fall back to `"I don't know based on the provided information."` if the facts are missing—eliminating hallucinations.

---

## 9. Key Engineering Takeaways

1. **Upsert Semantics**: `upsert` merges `insert` and `update`. If a point with the same ID already exists, Qdrant overwrites it; if not, it inserts a new record.
2. **Persistent Storage**: Once uploaded, your vectors live safely on Qdrant Cloud. In production, you separate ingestion (Parts 3–7) from querying (Parts 8–12) so you never re-embed your entire dataset on every user request.
3. **Sub-Linear Search Time**: Qdrant uses HNSW indexing graphs, enabling queries across millions of documents in single-digit milliseconds instead of $O(N)$ linear scans.
4. **Separation of Roles**:
   * **SentenceTransformer**: Encodes sentences into 384-dimensional dense vectors.
   * **Qdrant**: Indexes, persists, and retrieves points based on Cosine similarity.
   * **Groq LLM**: Reads retrieved payload strings to synthesize grounded human responses.