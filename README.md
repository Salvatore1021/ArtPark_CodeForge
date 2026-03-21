# SkillDeck

An AI-driven skill analysis tool. Upload a resume and an optional job description — SkillDeck extracts your skills, scores you against a job benchmark, identifies gaps, and builds a prioritised learning roadmap.

---

## How It Works

1. **Upload** a resume (PDF or DOCX) and optionally a job description PDF.
2. **Skills are extracted** from both documents using an LLM (OpenAI or local Ollama). If neither is configured, it falls back to regex matching against a built-in skill taxonomy — no API key required.
3. **Gaps are identified** by comparing your skills against the job benchmark, each scored by how critical the missing skill is.
4. **A learning roadmap** is generated from a curated course catalog, ordered by prerequisites into time-bounded phases.
5. **Results are displayed** in the browser — a skill radar chart, gap table, role recommendations, and a phase-grouped course pathway.

---

## Running the Project

### Option A — Docker (Recommended)

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env (optional)

docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Option B — Local

**1. Install dependencies**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

**2. Start the backend**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Serve the frontend**

```bash
cd frontend
python3 -m http.server 3000
```

Open `http://localhost:3000`.

---

## Environment Variables

Copy `.env.example` to `.env`. The only variable that matters:

```env
OPENAI_API_KEY=sk-your-key-here   # optional — works without it
OPENAI_MODEL=gpt-4o-mini
```

To use a local LLM instead, install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and start `ollama serve`. SkillDeck will detect it automatically.
