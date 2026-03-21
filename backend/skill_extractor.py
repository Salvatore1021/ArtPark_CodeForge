"""
skill_extractor.py
------------------
Skill extraction pipeline — LLM only.

The LLM defines everything (skill, category, proficiency, confidence).
No handcoded skill lists. No taxonomy. No rules. No fallback.

If no LLM backend is reachable, extraction raises a RuntimeError.
"""

import re
from collections import defaultdict, Counter
from llm_extractor import extract_skills_llm, map_to_taxonomy


# ════════════════════════════════════════════════════════════════════
# Experience years calculator
# ════════════════════════════════════════════════════════════════════

MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,
    "june":6,"july":7,"august":8,"september":9,
    "october":10,"november":11,"december":12,
}

def _parse_date(s: str):
    import datetime
    if not s:
        return None
    s = s.strip().lower()
    m = re.match(r"([a-z]+)\s+(\d{4})", s)
    if m:
        return int(m.group(2)) + MONTH_MAP.get(m.group(1), 6) / 12.0
    m = re.match(r"(\d{4})", s)
    if m:
        return float(m.group(1))
    if "present" in s or "current" in s:
        return float(datetime.date.today().year)
    return None


def _compute_experience(experience: list) -> dict:
    import datetime
    current_year = float(datetime.date.today().year)
    total_months = 0
    years_per_title = {}

    for job in experience:
        start = _parse_date(job.get("start_date", ""))
        end   = _parse_date(job.get("end_date",   "")) or current_year
        if start and end and end > start:
            yrs = end - start
            total_months += yrs * 12
            title = job.get("title", "")
            if title:
                years_per_title[title] = round(yrs, 1)

    total = round(total_months / 12, 1)
    if total < 2:
        seniority = "Junior (0–2 yrs)"
    elif total < 5:
        seniority = "Mid-Level (2–5 yrs)"
    elif total < 10:
        seniority = "Senior (5–10 yrs)"
    else:
        seniority = "Expert (10+ yrs)"

    return {
        "total_years":     total,
        "years_per_title": years_per_title,
        "seniority_level": seniority,
    }


# ════════════════════════════════════════════════════════════════════
# Domain classifier — derived from LLM categories, no hardcoding
# ════════════════════════════════════════════════════════════════════

def _classify_domains(skills: list) -> list:
    """
    Derive domain expertise ranking purely from LLM-assigned categories.
    Returns top 5 domains ordered by number of skills in each.
    """
    cat_counts = Counter(s["category"] for s in skills)
    return [cat for cat, _ in cat_counts.most_common(5)]


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def extract_all_skills(parsed_resume: dict,
                       taxonomy_skills: list = None) -> dict:
    """
    Extract skills from a parsed resume using the LLM.
    If taxonomy_skills is provided, maps extracted skills to the taxonomy.

    Raises RuntimeError if no LLM backend is reachable.
    """
    raw_text   = parsed_resume.get("raw_text", "")
    experience = parsed_resume.get("experience", [])

    # ── LLM extraction ────────────────────────────────────────────────
    llm_result = extract_skills_llm(raw_text)

    if not llm_result["llm_available"]:
        raise RuntimeError(
            f"LLM extraction failed: {llm_result.get('error', 'Ollama unavailable')}. "
            "Make sure Ollama is running: ollama serve"
        )

    skills       = llm_result["skills"]
    backend_used = llm_result["backend_used"]
    model_used   = llm_result["model_used"]

    # ── Reasoning trace ───────────────────────────────────────────────
    reasoning_trace = [
        {
            "layer":       "llm",
            "skill":       s["skill"],
            "category":    s["category"],
            "proficiency": s["proficiency"],
            "confidence":  s["confidence"],
            "reasoning":   s["reasoning"],
            "backend":     backend_used,
            "model":       model_used,
        }
        for s in skills
    ]

    # ── Filter hallucinations — drop skills with no real reasoning ────────
    INVALID_REASONING = {"", "not specified", "n/a", "none", "not mentioned"}
    skills = [
        s for s in skills
        if s.get("reasoning", "").strip().lower() not in INVALID_REASONING
    ]

    # ── Taxonomy mapping (optional) ───────────────────────────────────────
    if taxonomy_skills:
        skill_lookup = {s.lower().strip(): s for s in taxonomy_skills}
        mapping      = map_to_taxonomy(skills, taxonomy_skills)
        mapped = []
        for s in skills:
            matched = mapping.get(s["skill"])
            if matched:
                canonical = skill_lookup.get(matched.lower().strip(), matched)
                if canonical in taxonomy_skills:
                    mapped.append({
                        **s,
                        "original_skill": s["skill"],
                        "skill":          canonical,
                        "category":       s["category"],
                    })
                else:
                    print(f"  [taxonomy] removed '{s['skill']}' (no match)")
            else:
                print(f"  [taxonomy] removed '{s['skill']}' (no match)")
        skills = mapped

    # ── Dedup ─────────────────────────────────────────────────────────────
    seen, deduped = set(), []
    for s in skills:
        if s["skill"] not in seen:
            seen.add(s["skill"])
            deduped.append(s)
    skills = deduped

    # ── Group by category ─────────────────────────────────────────────
    by_category = defaultdict(list)
    for s in skills:
        by_category[s["category"]].append(s["skill"])

    # ── Experience info ───────────────────────────────────────────────
    experience_info = _compute_experience(experience)

    # ── Domains ───────────────────────────────────────────────────────
    domains = _classify_domains(skills)

    return {
        "skills":             skills,
        "skills_by_category": dict(by_category),
        "experience_info":    experience_info,
        "domains":            domains,
        "languages":          parsed_resume.get("languages", []),
        "confidence_scores":  {s["skill"]: s["confidence"] for s in skills},
        "reasoning_trace":    reasoning_trace,
        "backend_used":       backend_used,
        "model_used":         model_used,
    }
