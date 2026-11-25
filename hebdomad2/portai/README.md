# PortAI — Ask My Résumé

An interactive AI portfolio assistant that lets recruiters and visitors chat directly with your résumé. Built with a **FastAPI + Groq** backend and a single-file **HTML/CSS/JS** neo-brutalist frontend, deployed as two separate projects on **Vercel**.

---

## Project Structure

```text
portai/
├── backend/
│   ├── main.py            # FastAPI server, PDF reader, Groq LLM & Pydantic parsing
│   └── my_resume.pdf      # Source résumé file
├── frontend/
│   └── index.html         # Single-file chat UI (HTML, CSS, JS)
├── pyproject.toml         # Dependencies & Vercel FastAPI entrypoint
├── uv.lock                # Locked dependency versions
└── README.md
```

---

## How the Code Works

### 1. Backend (`backend/main.py`)
The backend is a **FastAPI** application powered by the **Groq SDK** (`openai/gpt-oss-120b`).

* **CORS Middleware:** Configured with `CORSMiddleware` so the browser-based frontend can send cross-origin requests to the API.
* **Structured Parsing with Pydantic:** Defines `Experience` and `Resume` models. On the first request, `read_pdf()` extracts raw text from `my_resume.pdf` using `pypdf`, and `parse_resume()` sends it to Groq with `response_format={"type": "json_object"}` and `Resume.model_json_schema()` to guarantee valid, structured JSON.
* **Serverless Lazy Caching (`get_resume()`):** Parses the résumé once on the first `/chat` request and stores it in a global `resume_data` variable so subsequent requests reuse the parsed data without re-reading the PDF.
* **Grounded Q&A (`ask_candidate()`):** Injects the parsed JSON résumé into the system prompt with strict rules—answer professionally as if HR is interviewing the candidate, never hallucinate, and reply `"I don't have enough information to answer that."` when details are missing.
* **API Endpoints:**
  * `GET /` — Health check returning `{"message": "Ye home page hai"}` (used by the frontend's Online/Offline indicator).
  * `POST /chat` — Accepts `{"question": "..."}` via `ChatRequest` and returns `{"answer": "..."}`.

### 2. Frontend (`frontend/index.html`)
A zero-dependency, single-file web app with a neo-brutalist design.

* **Environment Auto-Switching:** Automatically uses `http://127.0.0.1:8000` when opened locally (`localhost` / `127.0.0.1`) and switches to `DEPLOYED_API` (`https://ai-agent-sand-five.vercel.app`) in production.
* **Live Status Check:** Calls `GET /` on page load (`checkBackend()`) to toggle the top-right status badge between **Online** (green) and **Offline** (red).
* **Interactive UI:** Features a pointer-tracking animated face (which enters a "thinking" state while awaiting a reply) and clickable starter-question stickers.
* **Safe Response Formatting:** Escapes raw HTML (`esc()`) before converting basic Markdown (`**bold**`, bullet lists, headings, and tables) into clean HTML inside chat bubbles, with a built-in **Ask again** retry button on network errors.

---

## Local Development

1. **Install dependencies** (from `hebdomad2/portai/`):
   ```bash
   uv sync
   ```
2. **Set your API key** in a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. **Run the backend:**
   ```bash
   uv run uvicorn backend.main:app --reload
   ```
4. **Open the frontend:** Open `frontend/index.html` in your browser (or use Live Server). The status dot should show **Online**.

---

## Deployment on Vercel

Because this is a monorepo with a Python backend and a static HTML frontend, deploy them as **two separate Vercel projects** from the same GitHub repository.

### Step 1: Configure `pyproject.toml`
Ensure `[tool.vercel]` is set in `hebdomad2/portai/pyproject.toml` so Vercel automatically detects the FastAPI app:
```toml
[tool.vercel]
entrypoint = "backend.main:app"
```

### Step 2: Deploy the Backend
1. In Vercel, click **Add New → Project** and import your GitHub repository.
2. Click **Edit** next to **Root Directory** and select `hebdomad2/portai` (Vercel will auto-detect **FastAPI**).
3. Under **Environment Variables**, add:
   * **Key:** `GROQ_API_KEY`
   * **Value:** *your Groq API key*
4. Click **Deploy**.
5. Once live, open your production backend URL (e.g., `https://ai-agent-sand-five.vercel.app`) to verify it returns `{"message": "Ye home page hai"}`.
   *(Note: Ensure **Settings → Deployment Protection → Vercel Authentication** is turned **Off** so the frontend can reach the API.)*

### Step 3: Connect & Deploy the Frontend
1. Open `hebdomad2/portai/frontend/index.html` and set `DEPLOYED_API` to your production backend URL (no trailing slash):
   ```js
   const DEPLOYED_API = "https://ai-agent-sand-five.vercel.app";
   ```
2. Commit and push `index.html` to GitHub.
3. In Vercel, click **Add New → Project** and import the same repository again.
4. Set **Root Directory** to `hebdomad2/portai/frontend` and leave **Framework Preset** as **Other**.
5. Click **Deploy** and open the live frontend URL—the status badge should turn **Online**.

### Optional: Lock Down CORS
Once your frontend URL is finalized, update `allow_origins` in `backend/main.py` from `["*"]` to your exact frontend domain so only your site can call the API:
```python
allow_origins=["https://your-frontend-domain.vercel.app"]
```

# Personal Portfolio AI (Backend Architecture & FastAPI)

This module begins Week 2's major milestone project: **portai**—an interactive, ChatGPT-style portfolio application that enables technical recruiters and HR hiring managers to interview an AI twin grounded strictly in your verified resume and project data.

---

## 1. Project Overview & System Vision

Traditional portfolio websites are static documents: recruiters must manually scan dense PDFs to find relevant skills. **portai** transforms a static resume into an autonomous conversational endpoint capable of:

1. Answering recruiter inquiries about skills, experience, and projects in real time.
2. Maintaining strict factual grounding—answering queries about listed technologies while honestly declining unlisted competencies (preventing hallucinated qualifications).
3. Providing an API contract designed to plug seamlessly into modern web frontends (Streamlit, React, or Next.js).

### End-to-End System Pipeline

```text
+-----------------------------------------------------------------------------------+
| BROWSER / RECRUITER CLIENT                                                        |
| (Sends HTTP POST with JSON: {"question": "Does Pratyush know networking?"})        |
+----------------------------------------+------------------------------------------+
                                         | HTTP / HTTPS Protocol (JSON Request)
                                         v
+-----------------------------------------------------------------------------------+
| FASTAPI BACKEND (Uvicorn ASGI Server on localhost:8000)                           |
|                                                                                   |
|   [ Router: @app.post("/chat") ]                                                  |
|                 |                                                                 |
|                 v                                                                 |
|   [ Request Validation via Pydantic: ChatRequest ]                                |
|                 |                                                                 |
|                 v                                                                 |
|   [ PDF Ingestion & Extraction: pypdf (my_resume.pdf) ]                           |
|                 |                                                                 |
|                 v                                                                 |
|   [ Resume Structuring: Groq LLM (llama-3.3-70b-versatile) -> Resume JSON ]       |
|                 |                                                                 |
|                 v                                                                 |
|   [ Grounded QA Engine: ask_candidate(question, resume_json) ]                    |
+----------------------------------------+------------------------------------------+
                                         | LLM Generation & Synthesis
                                         v
+-----------------------------------------------------------------------------------+
| FACTUAL RESPONSE GENERATION                                                       |
| - Found in Resume: "Yes, Pratyush has extensive experience with networking..."    |
| - Absent from Resume: "I don't have enough information to answer that."           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Concepts & Engineering Architecture

### Why Separate Backend from Frontend?

In production engineering, user interfaces and core execution engines have different operational boundaries:
- **Frontend**: Runs inside the client's web browser (Chrome, Safari). Browsers do not run native Python scripts; they parse HTML, CSS, JavaScript, and understand the **HTTP/HTTPS** protocol.
- **Backend**: Runs on private servers. It executes Python code, connects to model inference providers (Groq), manages files, and secures private API credentials.
- **Separation of Concerns**: Keeping backend logic decoupled from frontend templates ensures that your LLM pipeline can power multiple interfaces—web apps, mobile clients, Slack bots, or automated CLI tools—without rewriting core logic.

---

### What is an API? The Restaurant & Waiter Analogy

```text
+------------------+           +---------------------+           +-------------------+
|     CUSTOMER     | <-------> |       WAITER        | <-------> |       CHEF        |
| (Browser Client) |   HTTP    |        (API)        |  Python   |   (LLM Engine)    |
| Speaks Plain Eng |  Request  | Translates Formats  |   Code    | Executes Culinary |
|                  |  & Resp   | Routes Dish Orders  |  & Logic  | / Inference Logic |
+------------------+           +---------------------+           +-------------------+
```

Imagine dining at an upscale steakhouse:
- **The Customer (Client/Browser)**: Wants a meal but does not understand professional culinary terminology. The customer says: *"Cook it well, slightly charred on the outside."*
- **The Chef (Backend/LLM)**: Works inside the kitchen. The chef operates strictly on culinary specifications (*"Medium Rare"*, *"Well Done"*, internal core temperatures).
- **The Waiter (API - Application Programming Interface)**: Acts as the essential translator and intermediary. The waiter takes the customer's request, translates it into kitchen tickets the chef understands, waits for the dish to finish, and delivers the prepared plate back to the dining table.

Without the API, web browsers cannot communicate with raw Python scripts. Frameworks like **FastAPI** serve as this API layer.

---

### HTTP Routing & Decorators: `@app.get` vs. `@app.post`

When users navigate websites:
1. **`@app.get("/")` (Read-only)**:
   - Used to fetch data without sending an input payload.
   - When you type `localhost:8000/` into a browser address bar, the browser automatically dispatches an HTTP **GET** request.
2. **`@app.post("/chat")` (Write/Payload)**:
   - Used when the client submits a structured data payload (e.g., user question string inside JSON) for processing.
   - **Crucial Debugging Concept**: If you navigate directly to `localhost:8000/chat` in your browser URL bar, FastAPI returns `{"detail": "Method Not Allowed"}` (`HTTP 405`). This is because the browser dispatched a `GET` request to an endpoint configured exclusively for `POST`.

---

### ASGI Servers & Hot Reloading (`Uvicorn`)

- Python scripts containing normal functions are executed via `python main.py`.
- Once converted into a FastAPI instance (`app = FastAPI()`), the script becomes a persistent web service requiring an ASGI (Asynchronous Server Gateway Interface) server: **Uvicorn**.
- The flag `--reload` activates file watching: saving modifications inside `main.py` immediately recompiles the active local server without manual restarts.

---

## 3. Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, isolated Python virtual environment management.

### Step 1: Initialize Workspace & Environment

```bash
# Navigate to workspace
cd hebdomad2

# Initialize project folder
uv init portai
cd portai

# Create Python virtual environment
uv venv 

# Activate virtual environment

# macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Core Dependencies

```bash
uv add fastapi uvicorn groq python-dotenv pydantic pypdf
```

### Step 3: Run the Development Server

```bash
uvicorn main:app --reload
```

Once running, open your browser to `http://localhost:8000/docs` to test your endpoints using FastAPI's built-in Swagger UI.