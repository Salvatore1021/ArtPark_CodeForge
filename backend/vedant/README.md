# AI-Driven Candidate Onboarding Engine

A modular, production-ready Python system that evaluates candidate resumes,
identifies skill gaps, and generates a personalised 8-week learning roadmap —
supporting both **CSV datasets** and **individual PDF resumes** as input.

---

## Project Structure

```
onboarding_engine/
├── main.py                  # CLI entry point — ties everything together
├── config.py                # Paths, thresholds, benchmarks, dependency edges
├── skills_taxonomy.py       # Master skill taxonomy (all domains)
├── pdf_parser.py            # PDF text extraction + resume structure parsing
├── skill_extractor.py       # Skill library build, extraction, TF-IDF fitting
├── dependency_graph.py      # NetworkX DAG construction and graph queries
├── candidate_evaluator.py   # 70/30 Weighted Fit + Vector Similarity scoring
├── gap_prioritizer.py       # Graph-enhanced skill gap ordering
├── roadmap_builder.py       # 8-week narrative roadmap generation
├── dashboard.py             # Matplotlib 4-panel onboarding dashboard
├── requirements.txt         # Python dependencies
└── data/                    # (you create this) place Resume.csv and PDFs here
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your files in a `data/` folder:
```
data/
├── Resume.csv           # columns: ID, Category, Resume_str
└── job_description.pdf  # optional — improves vectorizer coverage
```

---

## Usage

### 1. Evaluate a candidate from the CSV by ID
```bash
python main.py --csv data/Resume.csv --id 16237710 --job data/job_desc.pdf
```

### 2. Evaluate a candidate directly from their PDF resume  *(new feature)*
```bash
python main.py --resume_pdf path/to/john_doe_resume.pdf \
               --category ACCOUNTANT \
               --job data/job_desc.pdf
```
The engine will automatically:
- Extract name, email, phone, LinkedIn, GitHub from the PDF
- Parse Education, Experience, Skills, and Certifications sections
- Run the full evaluation + roadmap pipeline on the extracted text

### 3. Batch evaluate all candidates in the CSV
```bash
python main.py --csv data/Resume.csv --batch --job data/job_desc.pdf
# → writes output/batch_results.csv
```

### 4. Suppress dashboard save (display only)
```bash
python main.py --csv data/Resume.csv --id 16237710 --no_save
```

---

## How It Works

| Step | Module | What Happens |
|------|--------|-------------|
| 1 | `skill_extractor.py` | Builds a unified skill library from `skills_taxonomy.py` across all domains; fits TF-IDF vectorizer |
| 2 | `dependency_graph.py` | Constructs a NetworkX DAG mapping prerequisite → advanced skill relationships |
| 3 | `candidate_evaluator.py` | Computes 70% Hard / 30% Soft Weighted Fit Score + TF-IDF Cosine Similarity → Composite Score → Grade |
| 4 | `gap_prioritizer.py` | Orders skill gaps by graph depth (foundational first) with cosine similarity tie-breaking |
| 5 | `roadmap_builder.py` | Distributes gaps across 8 weeks (4 foundational → 4 advanced) with objectives & success criteria |
| 6 | `dashboard.py` | Renders 4-panel matplotlib dashboard: profile, readiness donut, skill bridge graph, roadmap table |

---

## Skills Taxonomy Integration

`skills_taxonomy.py` provides 20+ domain categories powering all skill extraction:

- `programming_languages`, `web_development`, `databases`, `cloud_devops`, `data_science_ml`
- `agriculture_core`, `agriculture_tech`
- `management`, `soft_skills`, `research`
- `languages`, `office_tools`, `finance_business`
- `education_training`, `development_ngo`
- `healthcare_clinical`, `clinical_research_tools`
- `administrative`, `it_support`, `telecom_rf`, `design_creative`
- `hospitality_service`, `education_pedagogy`

Extend by editing `SKILLS_TAXONOMY` in `skills_taxonomy.py` — no other file changes needed.

---

## Extending Category Benchmarks

Edit `CATEGORY_BENCHMARKS` in `config.py` to add new role types:

```python
CATEGORY_BENCHMARKS["DATA ENGINEER"] = [
    "Python", "SQL", "Spark", "Kafka", "Airflow",
    "AWS", "Docker", "Communication",
]
```

---

## Output

- `output/dashboard_<ID>.png` — visual onboarding dashboard
- `output/batch_results.csv`  — batch evaluation summary
- Console roadmap text printed for every single-candidate run
