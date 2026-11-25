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