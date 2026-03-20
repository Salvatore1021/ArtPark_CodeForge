# AI-Driven Candidate Onboarding Engine — v2.1

---

## What's new vs v2

### Optimisations
| Area | Before | After | Speedup |
|---|---|---|---|
| `extract_skills()` | One `re.search` per skill (215 calls) | Single combined OR-regex | **54×** |
| `precompute_skill_vectors()` | Loop `vectorizer.transform([skill])` | One batch `vectorizer.transform(list)` | **7×** |
| TF-IDF n-grams | unigrams only | bigrams `(1,2)` — captures "machine learning" as one token | better accuracy |
| Skill library lookup | `list` iteration | `frozenset` O(1) membership | instant |

### New features

| Module | Feature |
|---|---|
| `skill_confidence.py` | Per-skill confidence score (0–1) replaces binary match. `confidence = 0.5×text_match + 0.3×TF-IDF_cosine + 0.2×frequency`. Gaps are now any skill below the **strong** threshold (0.70) — partial knowledge (0.40–0.70) is visible and tracked. |
| `role_recommender.py` | Auto-detect best-fit roles. Scores every job title in the taxonomy and returns top-N by composite. Use `--recommend` flag. |
| `progress_tracker.py` | JSON history per candidate (`output/<id>_history.json`). Every run is appended. Re-run to see composite delta, new skills gained, gaps closed. |
| `report_exporter.py` | Exports `output/<id>_report.json` (full machine-readable record) and `output/<id>_report.txt` (formatted human-readable report) after every run. |
| `dashboard.py` | New **5th panel**: horizontal skill confidence bar chart showing every benchmark skill coloured by tier (green=strong, amber=partial, red=weak) with O*NET weight markers. |

---

## Project structure

```
onboarding_engine/
├── main.py                # CLI entry point — all features wired
├── config.py              # Constants, category→job map, edges, learning meta
├── enhanced_taxonomy.py   # O*NET + tech skills taxonomy (900+ job titles)
├── taxonomy_adapter.py    # Clean API: merged benchmarks, fuzzy match, discovery
├── pdf_parser.py          # PDF text extraction + resume structure parsing
├── skill_extractor.py     # Skill library, fast combined-regex extraction, TF-IDF
├── dependency_graph.py    # NetworkX DAG (91 nodes, 85 edges)
├── skill_confidence.py    # Per-skill confidence scoring  [NEW]
├── candidate_evaluator.py # O*NET weighted scoring with confidence fit
├── gap_prioritizer.py     # Three-key gap ordering with confidence integration
├── roadmap_builder.py     # Week-by-week roadmap generation
├── role_recommender.py    # Auto role recommendation  [NEW]
├── progress_tracker.py    # Run history & delta reporting  [NEW]
├── report_exporter.py     # JSON + text report export  [NEW]
├── dashboard.py           # 5-panel matplotlib dashboard (adds confidence chart)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Evaluate a PDF resume
```bash
# With known job title
python main.py --resume_pdf Aditya_CV.pdf --job_title "Machine Learning"

# Auto-recommend best role from taxonomy
python main.py --resume_pdf Aditya_CV.pdf --recommend
```

### Evaluate from CSV
```bash
python main.py --csv data/Resume.csv --id 16237710
python main.py --csv data/Resume.csv --id 16237710 --recommend
```

### Batch evaluate
```bash
python main.py --csv data/Resume.csv --batch
```

### Discovery
```bash
python main.py --list_sectors
python main.py --list_jobs --sector "Data & AI"
python main.py --list_jobs --sector "Backend"
```

---

## Output files (all auto-generated)

| File | Contents |
|---|---|
| `output/dashboard_<ID>.png` | 5-panel visual dashboard |
| `output/<ID>_report.json` | Full machine-readable evaluation record |
| `output/<ID>_report.txt` | Formatted human-readable onboarding summary |
| `output/<ID>_history.json` | Run history for progress tracking |
| `output/batch_results.csv` | Batch evaluation summary |

---

## Scoring model

```
confidence_i = 0.5 × text_match + 0.3 × TF-IDF_cosine + 0.2 × frequency

Confidence Fit  = Σ(onet_weight_i × confidence_i) / Σ(onet_weight_i)
Weighted Cosine = cosine(resume_vec, weighted_bench_vec)

Composite = (Confidence Fit + Weighted Cosine) / 2
```

Confidence tiers:
- **Strong** (≥ 0.70) — skill clearly demonstrated
- **Partial** (0.40–0.70) — adjacent / implied knowledge
- **Weak** (< 0.40) — gap, goes into learning plan

---

## Gap priority sort key

```
(graph_level ASC, onet_weight DESC, gap_score DESC)
```

where `gap_score = 1 - confidence` — the less the candidate knows about a skill, the sooner it's addressed.
