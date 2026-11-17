# Episode 06: AI Resume Screener & Job Matcher (End-of-Week 1 Mini Project)

Part of the **Free 8-Week AI Engineer Course** by Pratyush Narayan (*Padho with Pratyush*).

This project culminates Week 1 by combining every foundational concept covered so far—**Groq API client registration, system roles, temperature calibration, token limits, and Pydantic-enforced JSON structured outputs**—into an end-to-end, production-style AI tool: an automated **HR Resume Evaluator & Candidate Matcher**.

---

## 1. Project Overview & Workflow

In high-volume recruitment, HR departments receive thousands of resumes across diverse document formats (`.pdf`, `.docx`) for a single open position. Manually parsing unstructured resumes and comparing them against complex Job Descriptions (JDs) is slow, subjective, and prone to human error.

### System Architecture Pipeline

```text
+------------------------------------+      +-----------------------------------------+
| Unstructured Job Description       |      | Candidate Resumes Directory (Folder)    |
| (Raw plain text, e.g., Amazon SDE) |      | (Mixed .pdf, .docx, tables, paragraphs) |
+-----------------+------------------+      +--------------------+--------------------+
                  |                                              |
                  v                                              v
+------------------------------------+      +-----------------------------------------+
| Step 1: LLM JD Parser              |      | Step 2: Multi-Format Document Reader    |
| (Pydantic: JobD Schema)            |      | (pypdf + python-docx table/text extract)|
+-----------------+------------------+      +--------------------+--------------------+
                  |                                              |
                  |                                              v
                  |                         +-----------------------------------------+
                  |                         | Step 3: LLM Resume Parser               |
                  |                         | (Pydantic: Experience & Resume Schemas) |
                  |                         +--------------------+--------------------+
                  |                                              |
                  +-----------------------+----------------------+
                                          |
                                          v
                            +-----------------------------+
                            | Step 4: LLM Scoring Engine  |
                            | (Pydantic: MatchResult)     |
                            +--------------+--------------+
                                           |
                                           v
                            +-----------------------------+
                            | Step 5: Ranking & Decision  |
                            | (Top 2 vs Bottom 2 Verdict) |
                            +-----------------------------+
```

---

## 2. Core Concepts & Engineering Highlights

### 1. Structured Data vs. Raw Text Comparison
Never compare raw resume text directly against raw JD text. String-on-string comparison loses context and hallucinates missing qualifications. By extracting both the Job Description and Resumes into strict **Pydantic schemas** first, the evaluation engine performs a clean, apples-to-apples comparison.

### 2. Hierarchical Pydantic Modeling
Resumes are structurally varied and often messy:
- Candidates might have multiple prior roles, single roles, or zero experience.
- An `Experience` model (`company`, `role`, `duration`, `description`, `skills_used`) is nested inside the parent `Resume` model as `list[Experience]`.
- Optional fields (`| None = None`) prevent schema validation errors when candidates omit sections like certifications or phone numbers.

### 3. Comprehensive File Parsing (Handling Tables)
- `.pdf` files are parsed page-by-page via `pypdf`.
- `.docx` files require parsing both regular **paragraphs** and **tables** (where candidates frequently format academic qualifications and marks).
- Unsupported extensions (e.g., `.png`, `.jpg` screenshots) are filtered out immediately.

### 4. Rate Limiting & DoS (Denial of Service) Architecture
When running batch processing across multiple candidates:
- Each candidate requires two sequential LLM calls: one to parse the resume, and one to score it against the JD.
- Public LLM APIs enforce strict rate limits to prevent **Denial of Service (DoS)** attacks (preventing one client from overwhelming the server).
- **The City Road Analogy**: If a single person floods a city's main thoroughfare with 10 million cars, legitimate commuters are blocked. Rate limits ensure equitable throughput.
- **Defensive Delay**: Adding explicit delays (e.g., `time.sleep(5)`) between iterations prevents burst-limit (`HTTP 429 Too Many Requests`) rejections.

---


---

## 4. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, isolated package management.

### Step 1: Initialize Workspace & Environment

```bash
# Navigate to workspace
cd hebdomad

# Initialize project
uv init period5
cd period5

# Create virtual environment with Python 3.11
uv venv 

# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Required Dependencies

```bash
uv add groq python-dotenv pydantic pypdf python-docx
```

---

## 5. Complete Code Implementation 

# AI Resume Screener & Job Matcher (Complete Code Walkthrough)

This document provides a detailed, step-by-step explanation of your Python implementation for the **AI HR Resume Screener & Candidate Matcher** project.

---

## 1. End-to-End Execution Flow

Your script processes a Job Description (JD) and a folder of resumes in **5 distinct stages**:

```text
[Raw Job Description] ──> LLM Call (JobD Schema) ──> Validated `job` Object
                                                              │
[resumes/ Folder]                                             ▼
  ├── candidate1.pdf ──> read_pdf() ──> parse_resume() ──> final_score() ──> all_results
  └── candidate2.docx ─> read_docx() ─> parse_resume() ──> final_score() ──> all_results
                                                                                  │
                                                       Sort Descending ◄──────────┘
                                                              │
                                           ┌──────────────────┴──────────────────┐
                                           ▼                                     ▼
                                   TOP 2 CANDIDATES                     LOWEST 2 CANDIDATES
```

---

## 2. Section-by-Section Code Breakdown

### Section A: Environment Setup & Groq Client Registration

```python
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"
```

* **`load_dotenv()` & `os.getenv("GROQ_API_KEY")`**: Loads your secret API key from the `.env` file without hardcoding credentials in your script.
* **Guard Check**: Raises a `ValueError` immediately if the key is missing so the script fails fast rather than throwing cryptic authentication errors later.
* **`client = Groq(...)`**: Authenticates your script with the Groq API server.
* **`model = "openai/gpt-oss-120b"`**: Uses a 120-billion-parameter model capable of following complex JSON schemas and performing nuanced HR evaluations.

---

### Section B: Structuring the Job Description (`JobD` Schema)

```python
class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]

jobd_schema = JobD.model_json_schema()
```

* **Why convert the JD into JSON first?** Raw job descriptions (like the Amazon SDE-I posting in your code) contain company culture paragraphs and marketing text. Converting it into a `JobD` Pydantic model isolates only the actionable hiring criteria.
* **`minimum_experience: float | None`**: Entry-level roles (like SDE-I) often do not specify minimum years of experience. Allowing `None` (`null` in JSON) prevents validation errors when no number is stated.
* **Prompt Guardrails in `system_prompt`**:
  * `Do NOT return the schema itself. Do NOT return fields like "properties", "title" or "type".` — Prevents the LLM from echoing back the raw JSON schema blueprint instead of the extracted data.
  * `Do not invent information.` — Prevents hallucinations.
* **Deserialization**:
  ```python
  job_data = json.loads(raw_json)
  job = JobD(**job_data)
  ```
  Converts the raw JSON string returned by the LLM into a Python dictionary (`json.loads`), then unpacks it (`**job_data`) into a strongly typed `JobD` instance.

---

### Section C: Hierarchical Resume Schemas (`Experience` & `Resume`)

```python
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []
```

* **Nested Modeling (`experiences: list[Experience]`)**: Because candidates may have multiple internships or jobs, each role is captured as its own structured `Experience` object inside a list.
* **Optional Defaults (`= None` and `= []`)**: Resumes vary widely—some freshers have no work experience, no certifications, or omit a phone number. Defaulting missing fields to `None` or `[]` ensures Pydantic never crashes on incomplete resumes.
* **`parse_resume(resume_text)`**:
  * Instructs the LLM to extract based on **meaning**, not exact section titles (e.g., recognizing *Work History*, *Employment*, and *Internships* under `experiences`).
  * Extracts skills mentioned anywhere in the resume (projects, internships, or dedicated skills sections).

---

### Section D: Multi-Format Document Readers (`read_pdf`, `read_docx`, `read_resume`)

```python
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file_path):
    document = Document(file_path)
    text = ""
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"
    
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text += cell.text + "\n"
    return text

def read_resume(file_path):
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    else:
        return None
```

* **`read_pdf()`**: Uses `pypdf.PdfReader` to loop through every page of a PDF and concatenate its extracted text.
* **`read_docx()`**: Uses `python-docx` to extract text from both:
  1. **Standard paragraphs** (`document.paragraphs`).
  2. **Tables** (`document.tables` $\rightarrow$ `rows` $\rightarrow$ `cells`)—critical because many candidates format their 10th/12th/college grades or skills inside Word tables, which `document.paragraphs` alone would miss.
* **`read_resume()`**: Checks the file extension (`file_path.suffix.lower()`) and routes the file to the appropriate parser.

---

### Section E: Candidate Scoring Engine (`MatchResult` & `final_score`)

```python
class MatchResult(BaseModel):
    score: float
    details: dict
```

* **`final_score(job, resume)`**:
  * Serializes both validated objects using `job.model_dump_json(indent=2)` and `resume.model_dump_json(indent=2)`.
  * Asks the LLM (acting as an HR recruiter) to return a JSON object matching `MatchResult` containing:
    * `score`: A float between `0` and `100`.
    * `details`: A dictionary containing candidate name, matching skills, missing important skills, experience requirement status, match percentage, and a concise final verdict.

---

### Section F: Batch Folder Loop, Rate Limiting & Ranking

```python
resume_folder = Path("resumes")
all_results = []

for file_path in resume_folder.iterdir():
    if file_path.suffix.lower() not in [".pdf", ".docx"]:
        continue
    print("\nProcessing:", file_path.name)
    resume_text = read_resume(file_path)
    parsed_resume = parse_resume(resume_text)  # LLM Call 1
    time.sleep(5)
    result = final_score(job, parsed_resume)   # LLM Call 2
    time.sleep(5)
    print("Score:", result.score)
    all_results.append({
        "name": parsed_resume.name,
        "score": result.score,
        "details": result.details
    })
```

* **File Filtering**: `if file_path.suffix.lower() not in [".pdf", ".docx"]: continue` safely skips unsupported files (such as `.png`/`.jpg` screenshots or hidden system files).
* **Two LLM Calls per Candidate**:
  1. `parse_resume(resume_text)` converts raw text into structured `Resume` JSON.
  2. `final_score(job, parsed_resume)` compares `JobD` JSON against `Resume` JSON.
* **Why `time.sleep(5)` appears twice**:
  * API servers enforce **Rate Limits** (requests/tokens per minute) to prevent Denial-of-Service (DoS) overload.
  * Placing `time.sleep(5)` after **both** LLM calls ensures a 5-second cooldown between `parse_resume()` and `final_score()`, as well as between `final_score()` and the next candidate's `parse_resume()`, preventing `HTTP 429 Too Many Requests` errors.
* **Sorting & Shortlisting**:
  ```python
  all_results.sort(key=lambda candidate: candidate["score"], reverse=True)
  top_2 = all_results[:2]
  worst_2 = all_results[-2:]
  ```
  Sorts all evaluated candidates from highest score to lowest (`reverse=True`), slices the first two (`[:2]`) as the **Top 2 Candidates** to interview, and slices the last two (`[-2:]`) as the **Lowest 2 Candidates**.

---

## to Run
```bash
python -m resume_parser
```

---

## Recap & Engineering Takeaways

1. **Client Isolation**: Virtual environments (`uv venv`) guarantee that dependency conflicts never corrupt disparate modules.
2. **Behavioral Conditioning**: System roles govern agent personas, operational constraints, and tone.
3. **Entropy Control**: Lower temperatures (`temperature=0.0`–`0.1`) ensure deterministic schema compliance and factual grounding, while higher temperatures induce creative variance.
4. **Token Economics**: Monitoring `prompt_tokens` and `completion_tokens`—and enforcing ceilings with `max_tokens`—directly controls cloud billing and latency.
5. **Robust Structured Outputs**: Combining Pydantic `BaseModel` schemas with `response_format={"type": "json_object"}` transforms probabilistic generative models into deterministic, production-ready software components.