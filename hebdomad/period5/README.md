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

## 5. Complete Code Implementation (`main.py`)

```python
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document

# 1. Load environment & initialize client
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

# 2. Define Pydantic Schemas
class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]

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

class MatchResult(BaseModel):
    score: float
    details: dict

# 3. Document Reader Helpers
def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file_path: Path) -> str:
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

def read_resume(file_path: Path) -> str | None:
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    return None

# 4. LLM Parsing & Scoring Functions
def parse_job_description(jd_text: str) -> JobD:
    jobd_schema = JobD.model_json_schema()
    system_prompt = f"""
    You are an expert HR assistant. Analyze job descriptions and extract structured information.
    Return ONLY valid JSON matching this schema:
    {jobd_schema}

    IMPORTANT:
    - Do NOT return the schema itself (no "properties", "title", or "type").
    - If minimum experience is not mentioned, return null.
    - If information for a list is missing, return an empty list.
    - Do not invent information.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze the following job description:\n{jd_text}"}
        ],
        response_format={"type": "json_object"}
    )
    return JobD(**json.loads(response.choices[0].message.content))

def parse_resume(resume_text: str) -> Resume:
    resume_schema = Resume.model_json_schema()
    system_prompt = f"""
    You are an expert resume parser. Extract information based on meaning across sections
    (Experience, Work History, Employment, Internships, Projects, Skills).
    Return ONLY valid JSON matching this schema:
    {resume_schema}

    Rules:
    1. Do not invent information.
    2. Return null for missing scalar values and [] for missing lists.
    3. Include internships inside experiences.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse the following resume:\n{resume_text}"}
        ],
        response_format={"type": "json_object"}
    )
    return Resume(**json.loads(response.choices[0].message.content))

def final_score(job: JobD, resume: Resume) -> MatchResult:
    match_schema = MatchResult.model_json_schema()
    prompt = f"""
    You are an HR recruiter. Compare the candidate's resume with the job description.

    JOB DESCRIPTION:
    {job.model_dump_json(indent=2)}

    CANDIDATE RESUME:
    {resume.model_dump_json(indent=2)}

    Return JSON matching this schema:
    {match_schema}

    Include inside details:
    1. Candidate name
    2. Matching skills
    3. Missing important skills
    4. Whether experience requirement is met
    5. Overall match percentage (0 to 100)
    6. A short final verdict
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return MatchResult(**json.loads(response.choices[0].message.content))

# 5. Execution Pipeline
if __name__ == "__main__":
    job_description = """Paste your target Job Description here..."""
    job = parse_job_description(job_description)

    resume_folder = Path("resumes")
    all_results = []

    for file_path in resume_folder.iterdir():
        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        print(f"\nProcessing: {file_path.name}")
        resume_text = read_resume(file_path)

        parsed_resume = parse_resume(resume_text)
        time.sleep(5)  # Rate-limit protection

        result = final_score(job, parsed_resume)
        time.sleep(5)  # Rate-limit protection

        print("Score:", result.score)
        all_results.append({
            "name": parsed_resume.name,
            "score": result.score,
            "details": result.details
        })

    all_results.sort(key=lambda c: c["score"], reverse=True)

    print("\n--- TOP 2 CANDIDATES ---")
    for candidate in all_results[:2]:
        print(f"{candidate['name']} - {candidate['score']}%")
        print(candidate["details"])

    print("\n--- LOWEST 2 CANDIDATES ---")
    for candidate in all_results[-2:]:
        print(f"{candidate['name']} - {candidate['score']}%")
        print(candidate["details"])
```

---

## Recap & Engineering Takeaways

1. **Client Isolation**: Virtual environments (`uv venv`) guarantee that dependency conflicts never corrupt disparate modules.
2. **Behavioral Conditioning**: System roles govern agent personas, operational constraints, and tone.
3. **Entropy Control**: Lower temperatures (`temperature=0.0`–`0.1`) ensure deterministic schema compliance and factual grounding, while higher temperatures induce creative variance.
4. **Token Economics**: Monitoring `prompt_tokens` and `completion_tokens`—and enforcing ceilings with `max_tokens`—directly controls cloud billing and latency.
5. **Robust Structured Outputs**: Combining Pydantic `BaseModel` schemas with `response_format={"type": "json_object"}` transforms probabilistic generative models into deterministic, production-ready software components.