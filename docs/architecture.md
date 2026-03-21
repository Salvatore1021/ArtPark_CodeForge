# SkillDeck — Architecture & Workflow Documentation

## System Components

### 1. Document Parser (`backend/parsers.py`)
- **Primary**: `pdfplumber` — extracts text AND table content from PDFs
- **Fallback**: `PyPDF2` — lightweight fallback if pdfplumber fails on a page
- **DOCX**: `python-docx` — paragraphs + table cells
- **Text cleanup**: removes null bytes, collapses whitespace, normalizes newlines

### 2. Skill Extractor (`backend/skill_extractor.py`)
- Calls OpenAI `chat/completions` with `response_format: {type: json_object}`
- Temperature = 0.1 for deterministic extraction
- Separate prompts for Resume (proficiency_level 1-5, evidence) vs JD (required_level, is_mandatory)
- Falls back to regex pattern matching against taxonomy aliases when no API key is set
- Runs resume + JD extraction **concurrently** using `asyncio.create_task`

### 3. Gap Analyzer (`backend/gap_analyzer.py`)
- Builds canonical alias map from `data/skill_taxonomy.json`
- Resolves each JD skill to canonical form, then checks resume for match
- Statuses: `missing` | `below_required` | `met` | `exceeds`
- Priority formula: `mandatory_weight × gap_magnitude × required_level`
- Produces human-readable reasoning trace

### 4. Pathway Engine (`backend/pathway_engine.py`)
- Builds `nx.DiGraph` of course prerequisites (edges = "must do before")
- Includes transitive prereqs automatically
- **Kahn's Algorithm** with a max-priority-heap for tie-breaking
- Groups ordered courses into phases (≤40h each)
- Attaches reasoning trace to every course
- **Strict grounding**: only catalog courses ever appear in output

### 5. Course Catalog (`data/course_catalog.json`)
- 35+ courses across 8 domains
- Each course: id, name, description, skills_taught, prerequisites, duration_hours, level, category, tags, modules
- Prerequisites define the dependency graph edges

### 6. Skill Taxonomy (`data/skill_taxonomy.json`)
- Canonical skill names + alias lists for normalization
- 7 categories with proficiency level descriptions
- Used by gap_analyzer for fuzzy skill matching

### 7. FastAPI Backend (`backend/main.py`)
- `/analyze` — core endpoint (multipart form: resume + JD files)
- `/extract/resume`, `/extract/jd` — standalone extraction endpoints
- `/catalog`, `/catalog/{id}` — catalog inspection
- CORS enabled for all origins (restrict in production)
- Global exception handler prevents stack traces leaking to client
- Request timing middleware logs every request with latency

### 8. Frontend (`frontend/`)
- Vanilla HTML/CSS/JS — zero framework dependencies
- Drag & drop file upload with type/size validation
- Loading state with animated step progression
- Skill gap radar chart (Chart.js)
- Phase-grouped pathway with module drill-down
- Error toast system

## Data Flow Diagram

```
User Browser
     │
     │  POST /analyze (multipart)
     ▼
FastAPI (main.py)
     │
     ├──▶ parsers.py ──▶ extract_text(resume)
     │                   extract_text(jd)
     │
     ├──▶ skill_extractor.py ──▶ OpenAI API (async)
     │         resume_data                │
     │         jd_data       ◀────────────┘
     │
     ├──▶ gap_analyzer.py
     │         alias_map ◀── skill_taxonomy.json
     │         gap_analysis (gaps + priority scores)
     │
     └──▶ pathway_engine.py
               catalog ◀── course_catalog.json
               graph (NetworkX DiGraph)
               topo_sort + phase_grouping
               pathway (phases + reasoning)
     │
     ▼
JSON Response ──▶ app.js ──▶ Chart.js + DOM rendering
```
