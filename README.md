# SkillDeck

An AI-driven skill analysis tool. Upload a resume and an optional job description — SkillDeck extracts your skills, scores you against a job benchmark, identifies gaps, and builds a prioritised learning roadmap.

---

## How It Works

1. **Upload** a resume (PDF or DOCX) and optionally a job description PDF.
2. **Skills are extracted** from both documents using an LLM (OpenAI or local Ollama). If neither is configured, it falls back to regex matching against a built-in skill taxonomy — no API key required.
3. **Gaps are scored** — see [Skill-Gap Logic](#skill-gap-analysis-logic) below.
4. **A learning roadmap** is generated from a curated course catalog, ordered by prerequisites into time-bounded phases.
5. **Results are displayed** in the browser — a skill radar chart, gap table, role recommendations, and a phase-grouped course pathway.

---

## Skill-Gap Analysis Logic

Once skills are extracted from both documents, each job-required skill is resolved to a canonical name using alias mappings from the skill taxonomy. It is then looked up in the candidate's profile and assigned one of four statuses: `missing`, `below_required`, `met`, or `exceeds`.

Each gap is then assigned a priority score using:

```
priority = mandatory_weight × gap_magnitude × required_level
```

- **mandatory_weight** — how critical the skill is to the role (from the job description or taxonomy benchmark)
- **gap_magnitude** — difference between the required proficiency level and the candidate's current level
- **required_level** — the absolute proficiency level the role demands (1–5 scale)

Gaps are sorted by this score to determine the order in which courses appear in the learning roadmap.

---

## Dependencies

```
fastapi>=0.115.0
uvicorn>=0.30.0
python-multipart>=0.0.9
pymupdf>=1.22.0
python-docx>=1.1.0
beautifulsoup4>=4.12.0
scikit-learn>=1.2.0
numpy>=1.23.0
scipy>=1.9.0
pandas>=1.5.0
networkx>=3.0
matplotlib>=3.6.0
```

All listed in `backend/requirements.txt`.

---

## Setup

### Prerequisites

- Python 3.12
- Docker + Docker Compose v2 (for Option A)
- An OpenAI API key (optional — app works without it)

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-key-here   # optional
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
PORT=8000
```

To use a local LLM instead, install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and start `ollama serve`. SkillDeck detects it automatically with no `.env` changes needed.

---

## Running the Project

### Option A — Docker (Recommended)

```bash
docker compose up --build
```

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost           |
| Backend  | http://localhost:8000      |
| API Docs | http://localhost:8000/docs |

### Option B — Local

**1. Create a virtual environment and install dependencies**

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

Open `http://localhost:3000` in your browser.
