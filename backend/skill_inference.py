"""
skill_inference.py
------------------
Infers additional skills from explicitly extracted ones using:
  1. Co-occurrence rules    — "if A then likely B"
  2. Role-based inference   — job titles imply skill bundles
  3. Education inference    — degree field implies domain skills

Every inferred skill is tagged with:
  - method: "inferred"
  - source: the skill that triggered the inference
  - confidence: lower than directly extracted skills (0.4–0.6)

This is FULLY TRANSPARENT — every inference has a traceable source.
"""

from collections import defaultdict


# ════════════════════════════════════════════════════════════════════
# 1. Skill Co-occurrence Rules
#    Format: "trigger_skill": [("implied_skill", confidence), ...]
# ════════════════════════════════════════════════════════════════════

SKILL_COOCCURRENCE = {

    # Programming language implies ecosystem
    "python":           [("data analysis", 0.6), ("scripting", 0.5)],
    "tensorflow":       [("python", 0.9), ("machine learning", 0.9),
                         ("deep learning", 0.8), ("neural networks", 0.7)],
    "pytorch":          [("python", 0.9), ("deep learning", 0.9),
                         ("machine learning", 0.8)],
    "scikit-learn":     [("python", 0.9), ("machine learning", 0.8),
                         ("data analysis", 0.7)],
    "pandas":           [("python", 0.9), ("data analysis", 0.8),
                         ("numpy", 0.7)],
    "react":            [("javascript", 0.9), ("html", 0.8), ("css", 0.8)],
    "angular":          [("javascript", 0.9), ("typescript", 0.7),
                         ("html", 0.8)],
    "node.js":          [("javascript", 0.9), ("rest api", 0.7)],
    "django":           [("python", 0.9), ("rest api", 0.7), ("sql", 0.6)],
    "fastapi":          [("python", 0.9), ("rest api", 0.8)],
    "kubernetes":       [("docker", 0.9), ("devops", 0.8), ("linux", 0.7)],
    "docker":           [("linux", 0.7), ("devops", 0.7)],
    "aws":              [("cloud", 0.8), ("linux", 0.6), ("devops", 0.6)],
    "spark":            [("hadoop", 0.7), ("python", 0.6), ("sql", 0.6),
                         ("data analysis", 0.7)],
    "tableau":          [("data analysis", 0.8), ("sql", 0.6)],
    "power bi":         [("data analysis", 0.8), ("sql", 0.6), ("excel", 0.7)],

    # Agriculture domain
    "irrigation":       [("crop management", 0.6), ("soil science", 0.5)],
    "plant disease":    [("plant protection", 0.8), ("crop management", 0.6)],
    "plant protection": [("crop management", 0.7), ("pest control", 0.6)],
    "crop management":  [("agronomy", 0.6), ("farm management", 0.5)],
    "livestock":        [("farm management", 0.6), ("agronomy", 0.5)],

    # Management domain
    "strategic planning": [("management", 0.8), ("leadership", 0.7),
                            ("planning", 0.8)],
    "capacity building":  [("training", 0.8), ("mentoring", 0.7),
                            ("leadership", 0.6)],
    "project management": [("planning", 0.8), ("leadership", 0.7),
                            ("stakeholder management", 0.6)],
    "team management":    [("leadership", 0.8), ("mentoring", 0.6),
                            ("communication", 0.6)],

    # Office tools
    "excel":            [("data analysis", 0.6), ("microsoft office", 0.8)],
    "word":             [("microsoft office", 0.8), ("reporting", 0.5)],
    "power point":      [("microsoft office", 0.8), ("presentation", 0.7)],

    # Soft skills
    "presentation":     [("communication", 0.7), ("public speaking", 0.6)],
    "coordination":     [("communication", 0.7), ("stakeholder management", 0.6)],
    "monitoring":       [("evaluation", 0.7), ("reporting", 0.6), ("research", 0.5)],
    "logical framework": [("project management", 0.7), ("planning", 0.6)],

    # Translation / language work
    "english":          [("communication", 0.5)],
    "pashto":           [("translation", 0.6)],
    "dari":             [("translation", 0.6)],
}


# ════════════════════════════════════════════════════════════════════
# 2. Role-based Inference
#    Job titles imply a bundle of expected skills
# ════════════════════════════════════════════════════════════════════

ROLE_SKILL_BUNDLES = {
    "software engineer":     [("git", 0.8), ("linux", 0.7), ("sql", 0.6),
                               ("rest api", 0.6), ("problem solving", 0.7)],
    "data scientist":        [("python", 0.9), ("machine learning", 0.9),
                               ("statistics", 0.8), ("data analysis", 0.9),
                               ("sql", 0.7), ("data mining", 0.7)],
    "data analyst":          [("sql", 0.9), ("excel", 0.8), ("data analysis", 0.9),
                               ("reporting", 0.7), ("tableau", 0.6)],
    "devops engineer":       [("docker", 0.9), ("kubernetes", 0.8), ("linux", 0.9),
                               ("ci/cd", 0.8), ("aws", 0.7)],
    "project manager":       [("project management", 0.9), ("planning", 0.9),
                               ("stakeholder management", 0.8), ("leadership", 0.8),
                               ("reporting", 0.7), ("risk management", 0.7)],
    "agriculture advisor":   [("crop management", 0.8), ("farm management", 0.7),
                               ("agronomy", 0.7), ("research", 0.6),
                               ("training", 0.6), ("coordination", 0.6)],
    "provincial manager":    [("management", 0.9), ("planning", 0.8),
                               ("reporting", 0.8), ("leadership", 0.8),
                               ("coordination", 0.7)],
    "plant protection manager": [("plant protection", 0.9), ("pest control", 0.8),
                                  ("crop management", 0.7), ("farm management", 0.6)],
    "language officer":      [("translation", 0.9), ("communication", 0.8),
                               ("reporting", 0.6)],
    "teacher":               [("curriculum development", 0.8), ("training", 0.8),
                               ("communication", 0.7), ("mentoring", 0.7)],
    "researcher":            [("research", 0.9), ("data collection", 0.8),
                               ("reporting", 0.7), ("statistics", 0.6)],
}


# ════════════════════════════════════════════════════════════════════
# 3. Education-based Inference
# ════════════════════════════════════════════════════════════════════

EDUCATION_SKILL_MAP = {
    "agriculture":          [("agronomy", 0.7), ("crop management", 0.7),
                              ("soil science", 0.6), ("plant protection", 0.6),
                              ("farm management", 0.6)],
    "computer science":     [("programming", 0.8), ("algorithms", 0.7),
                              ("data structures", 0.7), ("software development", 0.7)],
    "data science":         [("machine learning", 0.8), ("statistics", 0.8),
                              ("python", 0.7), ("data analysis", 0.8)],
    "business":             [("management", 0.7), ("marketing", 0.6),
                              ("business development", 0.6), ("finance", 0.5)],
    "information technology": [("networking", 0.7), ("software development", 0.6),
                                 ("database", 0.6)],
    "mathematics":          [("statistics", 0.8), ("data analysis", 0.6)],
    "economics":            [("data analysis", 0.6), ("research", 0.6),
                              ("statistics", 0.6)],
}


# ════════════════════════════════════════════════════════════════════
# Inference Engine
# ════════════════════════════════════════════════════════════════════

def infer_from_cooccurrence(extracted_skills: list[dict]) -> list[dict]:
    """Apply co-occurrence rules to infer additional skills."""
    known = {s["skill"].lower() for s in extracted_skills}
    inferred = []

    for skill_entry in extracted_skills:
        trigger = skill_entry["skill"].lower()
        if trigger in SKILL_COOCCURRENCE:
            for implied_skill, conf in SKILL_COOCCURRENCE[trigger]:
                if implied_skill not in known:
                    inferred.append({
                        "skill":      implied_skill,
                        "category":   _lookup_category(implied_skill),
                        "method":     "inferred_cooccurrence",
                        "confidence": conf,
                        "source":     trigger,
                        "reasoning":  f"'{trigger}' typically implies '{implied_skill}'"
                    })
                    known.add(implied_skill)

    return inferred


def infer_from_roles(job_titles: list[str],
                     extracted_skills: list[dict]) -> list[dict]:
    """Infer skills from job titles the candidate has held."""
    known = {s["skill"].lower() for s in extracted_skills}
    inferred = []

    for title in job_titles:
        title_lower = title.lower().strip()

        # Match job title to role bundles using full-phrase or majority-word match.
        # Single-word matching ("manager" in "provincial manager") caused false
        # positives — e.g. "Sales Manager" triggering agriculture skills.
        for role, skill_bundle in ROLE_SKILL_BUNDLES.items():
            # Only keep meaningful words (length > 3) for matching
            role_words = [w for w in role.split() if len(w) > 3]
            if not role_words:
                continue
            # Require ALL meaningful role words to appear in the title
            if all(word in title_lower for word in role_words):
                for skill, conf in skill_bundle:
                    if skill not in known:
                        inferred.append({
                            "skill":      skill,
                            "category":   _lookup_category(skill),
                            "method":     "inferred_role",
                            "confidence": conf,
                            "source":     title,
                            "reasoning":  f"Role '{title}' typically requires '{skill}'"
                        })
                        known.add(skill)

    return inferred


def infer_from_education(education_list: list[dict],
                          extracted_skills: list[dict]) -> list[dict]:
    """Infer skills from education field/degree."""
    known = {s["skill"].lower() for s in extracted_skills}
    inferred = []

    for edu in education_list:
        field = (edu.get("field", "") + " " + edu.get("degree", "")).lower()

        for edu_field, skill_bundle in EDUCATION_SKILL_MAP.items():
            if edu_field in field:
                for skill, conf in skill_bundle:
                    if skill not in known:
                        inferred.append({
                            "skill":      skill,
                            "category":   _lookup_category(skill),
                            "method":     "inferred_education",
                            "confidence": conf,
                            "source":     edu.get("field", ""),
                            "reasoning":  (f"Degree in '{edu.get('field', '')}'"
                                           f" typically includes '{skill}'")
                        })
                        known.add(skill)

    return inferred


def run_all_inference(extracted_skills: list[dict],
                      job_titles:        list[str],
                      education:         list[dict]) -> list[dict]:
    """
    Run all three inference passes and return combined inferred skills.

    Deduplicates across passes — a skill inferred by multiple passes
    gets its confidence boosted.
    """
    # Run all passes
    co_skills  = infer_from_cooccurrence(extracted_skills)
    role_skills = infer_from_roles(job_titles, extracted_skills + co_skills)
    edu_skills  = infer_from_education(education,
                                        extracted_skills + co_skills + role_skills)

    all_inferred = co_skills + role_skills + edu_skills

    # Deduplicate: if skill appears in multiple passes, boost confidence
    merged = {}
    for s in all_inferred:
        key = s["skill"]
        if key in merged:
            merged[key]["confidence"] = min(
                0.85,
                merged[key]["confidence"] + 0.1
            )
            merged[key]["reasoning"] += f" | Also: {s['reasoning']}"
        else:
            merged[key] = s

    return list(merged.values())


def _lookup_category(skill: str) -> str:
    """Look up category from taxonomy; return 'general' if not found."""
    try:
        from skills_taxonomy import SKILL_TO_CATEGORY
        return SKILL_TO_CATEGORY.get(skill.lower(), "general")
    except ImportError:
        return "general"
