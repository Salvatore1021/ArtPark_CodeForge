"""
skill_extractor.py
------------------
Full 5-layer skill extraction pipeline — NO LLM required.

Layer 1 — Exact / phrase match        (taxonomy regex, high precision)
Layer 2 — TF-IDF keyword extraction   (corpus-aware, handles paraphrasing)
Layer 3 — Semantic char n-gram match  (catches morphological variants)
Layer 4 — Proficiency detection       (beginner -> expert per skill)
Layer 5 — Skill inference             (co-occurrence + roles + education)

Every extracted/inferred skill carries:
  - method       : which layer found it
  - confidence   : 0.0-1.0
  - proficiency  : { level, score, signals, evidence }
  - reasoning    : human-readable explanation (powers the Reasoning Trace)
"""

import re
import math
from collections import defaultdict, Counter
from skills_taxonomy import SKILLS_TAXONOMY, SKILL_TO_CATEGORY
from proficiency_detector import detect_all_proficiencies
from skill_inference import run_all_inference
from semantic_matcher import SemanticSkillMatcher


# ════════════════════════════════════════════════════════════════════
# LAYER 1 — Exact Phrase Matching
# ════════════════════════════════════════════════════════════════════

def _build_phrase_patterns():
    all_skills = sorted(SKILL_TO_CATEGORY.keys(), key=len, reverse=True)
    return {
        skill: re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
        for skill in all_skills
    }

_PATTERNS = _build_phrase_patterns()


def extract_by_exact_match(text):
    found, seen = [], set()
    text_lower = text.lower()

    for skill, pattern in _PATTERNS.items():
        if skill in seen:
            continue
        match = pattern.search(text_lower)
        if match:
            start = max(0, match.start() - 80)
            end   = min(len(text_lower), match.end() + 80)
            found.append({
                "skill":      skill,
                "category":   SKILL_TO_CATEGORY[skill],
                "method":     "exact_match",
                "confidence": 0.7,
                "reasoning":  "Exact phrase found in resume text",
            })
            seen.add(skill)
    return found


# ════════════════════════════════════════════════════════════════════
# LAYER 2 — TF-IDF Keyword Extraction
# ════════════════════════════════════════════════════════════════════

def _tokenize(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z+#./-]{1,}\b", text.lower())
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def compute_tfidf_keywords(docs, top_n=40):
    tf_per_doc = []
    for doc in docs:
        tokens = _tokenize(doc)
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        tf_per_doc.append({t: c / total for t, c in tf.items()})

    df = Counter()
    for tf in tf_per_doc:
        for term in tf:
            df[term] += 1

    N = len(docs)
    results = []
    for tf in tf_per_doc:
        scores = {
            term: tf_val * (math.log((N + 1) / (df[term] + 1)) + 1)
            for term, tf_val in tf.items()
        }
        top = sorted(scores, key=scores.get, reverse=True)[:top_n]
        results.append(top)
    return results


def extract_by_tfidf(tfidf_keywords, known_skills):
    found = []
    for kw in tfidf_keywords:
        kw_lower = kw.lower().strip()
        if kw_lower in SKILL_TO_CATEGORY and kw_lower not in known_skills:
            found.append({
                "skill":      kw_lower,
                "category":   SKILL_TO_CATEGORY[kw_lower],
                "method":     "tfidf",
                "confidence": 0.5,
                "reasoning":  "High TF-IDF weight term matched skill taxonomy",
            })
    return found


# ════════════════════════════════════════════════════════════════════
# LAYER 3 — Semantic Matching (char n-gram similarity)
# ════════════════════════════════════════════════════════════════════

_semantic_matcher = SemanticSkillMatcher(n=3, threshold=0.58)


def extract_by_semantic(text, known_skills):
    matches = _semantic_matcher.extract_from_text(text, known_skills)
    return [
        {
            "skill":      m["skill"],
            "category":   m["category"],
            "method":     "semantic_match",
            "confidence": m["confidence"],
            "reasoning":  m.get("reasoning", ""),
        }
        for m in matches
    ]


# ════════════════════════════════════════════════════════════════════
# Experience Years Helper
# ════════════════════════════════════════════════════════════════════

MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,
    "june":6,"july":7,"august":8,"september":9,
    "october":10,"november":11,"december":12,
}

def _parse_date(s):
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
        return 2025.0
    return None


def compute_total_experience(experience_list):
    total_months = 0
    years_per_title = {}

    for job in experience_list:
        start = _parse_date(job.get("start_date", ""))
        end   = _parse_date(job.get("end_date", "")) or 2025.0
        if start and end and end > start:
            yrs = end - start
            total_months += yrs * 12
            title = job.get("title", "Unknown")
            if title:
                years_per_title[title] = round(yrs, 1)

    total_years = round(total_months / 12, 1)
    if total_years < 2:
        seniority = "Junior (0-2 yrs)"
    elif total_years < 5:
        seniority = "Mid-Level (2-5 yrs)"
    elif total_years < 10:
        seniority = "Senior (5-10 yrs)"
    else:
        seniority = "Expert (10+ yrs)"

    return {
        "total_years":     total_years,
        "years_per_title": years_per_title,
        "seniority_level": seniority,
    }


# ════════════════════════════════════════════════════════════════════
# Domain Classification
# ════════════════════════════════════════════════════════════════════

DOMAIN_KEYWORDS = {
    "Agriculture":              ["agriculture","crop","farm","irrigation",
                                  "livestock","agronomy","plant","soil",
                                  "harvest","fertilizer","pest"],
    "Information Technology":   ["software","programming","developer",
                                  "engineer","cloud","database","network",
                                  "cybersecurity","devops","machine learning"],
    "Data Science":             ["data","analytics","machine learning",
                                  "statistics","modelling","tableau","python",
                                  "sql","visualization"],
    "Education & Training":     ["teacher","training","curriculum",
                                  "instructor","university","school",
                                  "education","workshop","seminar"],
    "International Development":["ngo","humanitarian","community",
                                  "development","provincial","coordination",
                                  "donor","capacity building"],
    "Management":               ["manager","management","director","head",
                                  "supervisor","coordinator","lead",
                                  "strategic","planning"],
    "Language & Translation":   ["translation","interpreter","language",
                                  "pashto","dari","english","linguistic"],
    "Finance & Business":       ["finance","accounting","budget","marketing",
                                  "business","sales","procurement"],
    "Healthcare & Clinical":    ["clinical","patient","hospital","medical",
                                  "healthcare","oncology","irb","trial",
                                  "regulatory","compliance","audit","research",
                                  "nursing","pharmacy","outpatient","inpatient"],
    "Information Technology":   ["technical support","help desk","network","server",
                                  "windows","linux","firewall","vpn","active directory",
                                  "itil","infrastructure","sysadmin","it support"],
    "Telecom & RF":             ["rf","lte","cdma","wireless","telecom","antenna",
                                  "spectrum","base station","network optimization",
                                  "drive test","5g","4g","3g","evdo"],
    "Design & Creative":        ["graphic","design","adobe","illustrator","photoshop",
                                  "indesign","branding","typography","logo","ui","ux",
                                  "wireframe","mockup","creative","visual"],
    "Hospitality & Service":    ["customer service","hospitality","restaurant","dining",
                                  "front of house","guest","reservations","food","event"],
    "Education":                ["teacher","classroom","curriculum","lesson","student",
                                  "assessment","literacy","instruction","education",
                                  "school","learning","pedagogy","coaching"],
    "BPO & Call Center":        ["call center","bpo","inbound","outbound","customer support",
                                  "ticket","escalation","kpi","first call resolution",
                                  "workforce management","voice process","helpdesk"],
    "Consulting & Strategy":    ["consulting","consultant","strategy","process improvement",
                                  "six sigma","lean","operational excellence","gap analysis",
                                  "stakeholder","transformation","proposal","swot"],
    "Sales & Business Dev":     ["sales","crm","lead generation","revenue","digital marketing",
                                  "seo","sem","account management","territory","dealership",
                                  "sales manager","business development","pipeline"],
    "Fitness & Wellness":       ["fitness","wellness","gym","trainer","yoga","pilates",
                                  "nutrition","exercise","group fitness","personal training",
                                  "health coaching","cpr","certified instructor"],
    "Retail & Logistics":       ["retail","warehouse","logistics","inventory","forklift",
                                  "merchandising","stock","order fulfillment","osha",
                                  "shipping","receiving","supply chain"],
    "Automotive":               ["automotive","vehicle","dealership","car sales","auto",
                                  "finance and insurance","f&i","test drive","trade-in",
                                  "lease","dealer","dms","reynolds"],
    "Culinary & Food":          ["chef","cook","cooking","kitchen","catering","culinary",
                                  "menu","food preparation","recipe","baking","pastry",
                                  "food cost","haccp","food hygiene","banquet"],
    "Digital Media & Ad Tech":  ["programmatic","media buying","media planning","dsp",
                                  "digital advertising","display","remarketing","ad tech",
                                  "media strategy","audience targeting","campaign",
                                  "media analyst","media planner","ad trafficking"],
}

# ── Domain → allowed taxonomy categories ──────────────────────────────────
# Defines which skill categories are valid for each detected domain.
# Semantic matches from disallowed categories are filtered out as noise.
# Exact matches and explicitly listed skills always pass through regardless.

DOMAIN_ALLOWED_CATEGORIES = {
    "Healthcare & Clinical": {
        "healthcare_clinical", "clinical_research_tools", "administrative",
        "research", "management", "soft_skills", "office_tools",
        "office_tools_extended", "education_training", "data_science_ml",
        "finance_business", "languages",
    },
    "Information Technology": {
        "programming_languages", "web_development", "databases",
        "cloud_devops", "data_science_ml", "office_tools",
        "office_tools_extended", "management", "soft_skills",
        "research", "administrative", "finance_business", "languages",
    },
    "Agriculture": {
        "agriculture_core", "agriculture_tech", "management", "research",
        "soft_skills", "office_tools", "education_training",
        "development_ngo", "languages", "administrative",
    },
    "Data Science": {
        "data_science_ml", "programming_languages", "databases",
        "cloud_devops", "research", "management", "soft_skills",
        "office_tools", "finance_business", "languages", "administrative",
    },
    "Finance & Business": {
        "finance_business", "management", "office_tools",
        "office_tools_extended", "data_science_ml", "soft_skills",
        "research", "administrative", "languages", "education_training",
    },
    "Management": {
        "management", "soft_skills", "finance_business", "office_tools",
        "office_tools_extended", "research", "education_training",
        "development_ngo", "administrative", "languages", "data_science_ml",
    },
    "Education & Training": {
        "education_training", "management", "soft_skills", "research",
        "office_tools", "office_tools_extended", "development_ngo",
        "administrative", "languages", "finance_business",
    },
    "International Development": {
        "development_ngo", "management", "soft_skills", "research",
        "education_training", "agriculture_core", "finance_business",
        "office_tools", "languages", "administrative",
    },
    "Language & Translation": {
        "languages", "soft_skills", "office_tools", "management",
        "education_training", "administrative",
    },
    "Information Technology": {
        "it_support", "programming_languages", "web_development", "databases",
        "cloud_devops", "data_science_ml", "office_tools", "office_tools_extended",
        "management", "soft_skills", "research", "administrative",
        "finance_business", "languages", "telecom_rf",
    },
    "Telecom & RF": {
        "telecom_rf", "it_support", "programming_languages", "cloud_devops",
        "management", "soft_skills", "office_tools", "research",
        "administrative", "languages",
    },
    "Design & Creative": {
        "design_creative", "web_development", "soft_skills", "office_tools",
        "office_tools_extended", "management", "finance_business",
        "education_training", "administrative", "languages",
        "hospitality_service",
    },
    "Hospitality & Service": {
        "hospitality_service", "soft_skills", "management", "office_tools",
        "finance_business", "education_training", "administrative", "languages",
    },
    "Education": {
        "education_pedagogy", "education_training", "management", "soft_skills",
        "research", "office_tools", "office_tools_extended", "development_ngo",
        "administrative", "languages", "finance_business",
    },
    "BPO & Call Center": {
        "bpo_call_center", "hospitality_service", "it_support", "management",
        "soft_skills", "office_tools", "office_tools_extended", "administrative",
        "research", "finance_business", "languages", "databases",
        "consulting_strategy", "education_training",
    },
    "Consulting & Strategy": {
        "consulting_strategy", "management", "finance_business", "soft_skills",
        "research", "office_tools", "office_tools_extended", "administrative",
        "education_training", "development_ngo", "languages", "data_science_ml",
    },
    "Sales & Business Dev": {
        "sales_extended", "finance_business", "management", "soft_skills",
        "hospitality_service", "office_tools", "office_tools_extended",
        "administrative", "education_training", "research", "languages",
        "consulting_strategy", "data_science_ml",
    },
    "Fitness & Wellness": {
        "fitness_wellness", "hospitality_service", "education_training",
        "soft_skills", "management", "office_tools", "administrative",
        "languages", "healthcare_clinical",
    },
    "Retail & Logistics": {
        "retail_logistics", "hospitality_service", "soft_skills", "management",
        "office_tools", "administrative", "finance_business", "languages",
        "it_support",
    },
    "Automotive": {
        "automotive", "sales_extended", "finance_business", "management",
        "soft_skills", "office_tools", "administrative", "it_support",
        "hospitality_service", "languages", "consulting_strategy",
    },
    "Culinary & Food": {
        "culinary", "hospitality_service", "retail_logistics", "management",
        "soft_skills", "administrative", "languages", "finance_business",
    },
    "Digital Media & Ad Tech": {
        "digital_media", "sales_extended", "finance_business", "management",
        "soft_skills", "data_science_ml", "office_tools", "research",
        "administrative", "languages", "design_creative",
    },
}

# Categories that are always allowed regardless of domain (universal skills)
UNIVERSAL_CATEGORIES = {
    "soft_skills", "office_tools", "office_tools_extended",
    "languages", "administrative",
}

# Exact match and explicit_section methods always bypass the domain filter
BYPASS_METHODS = {"exact_match", "explicit_section"}


def apply_domain_filter(skills, detected_domains, reasoning_trace):
    """
    Filter out semantic-only skill matches that do not belong to the
    detected domain(s) of the resume.

    Rules:
    - Exact matches and explicit section skills always pass through.
    - Universal categories (soft skills, office tools) always pass through.
    - If no domain is detected, no filtering is applied.
    - A skill is kept if its category is allowed by ANY of the top-2 domains.
    - Filtered skills are logged to reasoning_trace as 'domain_filter' entries.
    """
    if not detected_domains:
        return skills

    # Build the combined allowed set from top-2 detected domains
    allowed_categories = set(UNIVERSAL_CATEGORIES)
    for domain in detected_domains[:2]:
        allowed_categories.update(
            DOMAIN_ALLOWED_CATEGORIES.get(domain, set())
        )

    passed, filtered = [], []
    for s in skills:
        method = s.get("method", "")
        category = s.get("category", "")

        # Always keep exact matches and explicit section entries
        is_bypass = any(m in method for m in BYPASS_METHODS)
        if is_bypass or category in allowed_categories:
            passed.append(s)
        else:
            filtered.append(s)
            reasoning_trace.append({
                "layer":     "domain_filter",
                "skill":     s["skill"],
                "reasoning": (
                    f"Removed: category '{category}' not relevant to "
                    f"detected domains {detected_domains[:2]}"
                ),
            })

    return passed


def classify_domain(raw_text, job_titles):
    combined = (raw_text + " " + " ".join(job_titles)).lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(
            len(re.findall(r"\b" + re.escape(kw) + r"\b", combined))
            for kw in keywords
        )
        if score > 0:
            scores[domain] = score
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def extract_all_skills(parsed_resume,
                       all_resume_texts=None,
                       resume_index=0):
    """
    Run the full 5-layer pipeline on a single parsed resume.

    Returns
    -------
    {
        skills             : directly extracted skills with full metadata
        inferred_skills    : skills inferred via co-occurrence/roles/edu
        skills_by_category : { category: [skill, ...] }
        experience_info    : { total_years, years_per_title, seniority }
        domains            : ordered list of domain expertise
        languages          : list of languages
        education          : list of education entries
        confidence_scores  : { skill: float }
        reasoning_trace    : full audit log (one entry per decision)
    }
    """

    raw_text       = parsed_resume.get("raw_text", "")
    experience     = parsed_resume.get("experience", [])
    education      = parsed_resume.get("education", [])
    section_skills = parsed_resume.get("skills", [])

    reasoning_trace = []

    # ── LAYER 1: Exact Match ─────────────────────────────────────────
    exact_skills = extract_by_exact_match(raw_text)
    for s in exact_skills:
        reasoning_trace.append({
            "layer":     "1_exact_match",
            "skill":     s["skill"],
            "reasoning": s["reasoning"],
        })
    known = {s["skill"] for s in exact_skills}

    # ── LAYER 2: TF-IDF ──────────────────────────────────────────────
    if all_resume_texts:
        all_keywords = compute_tfidf_keywords(all_resume_texts)
        keywords = all_keywords[resume_index]
    else:
        keywords = list(set(re.findall(r"\b\w[\w+#.-]{1,}\b", raw_text.lower())))

    tfidf_skills = extract_by_tfidf(keywords, known)
    for s in tfidf_skills:
        reasoning_trace.append({
            "layer":     "2_tfidf",
            "skill":     s["skill"],
            "reasoning": s["reasoning"],
        })
    known.update(s["skill"] for s in tfidf_skills)

    # ── LAYER 3: Semantic Matching ────────────────────────────────────
    semantic_skills = extract_by_semantic(raw_text, known)
    for s in semantic_skills:
        reasoning_trace.append({
            "layer":     "3_semantic",
            "skill":     s["skill"],
            "reasoning": s["reasoning"],
        })
    known.update(s["skill"] for s in semantic_skills)

    # ── LAYER 3b: Domain Filter ───────────────────────────────────────
    # Detect domains early (using raw text + job titles) so the filter
    # can run before the merge step.
    job_titles_early = [job.get("title", "") for job in experience]
    detected_domains  = classify_domain(raw_text, job_titles_early)

    semantic_skills = apply_domain_filter(
        semantic_skills, detected_domains, reasoning_trace
    )
    # Also re-filter tfidf skills for consistency
    tfidf_skills = apply_domain_filter(
        tfidf_skills, detected_domains, reasoning_trace
    )

    # ── Merge layers 1-3 ─────────────────────────────────────────────
    skill_registry = {}

    for skill_list, base_conf in [
        (exact_skills,    0.70),
        (tfidf_skills,    0.50),
        (semantic_skills, 0.55),
    ]:
        for s in skill_list:
            key = s["skill"]
            if key in skill_registry:
                skill_registry[key]["confidence"] = min(
                    1.0, skill_registry[key]["confidence"] + 0.15
                )
                skill_registry[key]["method"] += f"+{s['method']}"
            else:
                skill_registry[key] = {**s, "confidence": base_conf}

    # Boost skills explicitly listed in Skills section
    for raw_skill in section_skills:
        key = raw_skill.lower().strip()
        if key in SKILL_TO_CATEGORY:
            if key not in skill_registry:
                skill_registry[key] = {
                    "skill":      key,
                    "category":   SKILL_TO_CATEGORY[key],
                    "method":     "explicit_section",
                    "confidence": 0.9,
                    "reasoning":  "Explicitly listed in resume Skills section",
                }
            else:
                skill_registry[key]["confidence"] = min(
                    1.0, skill_registry[key]["confidence"] + 0.2
                )

    directly_found = list(skill_registry.values())

    # ── LAYER 4: Proficiency Detection ───────────────────────────────
    directly_found = detect_all_proficiencies(directly_found, raw_text)
    for s in directly_found:
        reasoning_trace.append({
            "layer":     "4_proficiency",
            "skill":     s["skill"],
            "level":     s["proficiency"]["level"],
            "signals":   s["proficiency"]["signals"],
            "reasoning": (
                f"Proficiency '{s['proficiency']['level']}' inferred "
                f"from context signals: {s['proficiency']['signals']}"
            ),
        })

    # ── LAYER 5: Skill Inference ──────────────────────────────────────
    job_titles      = job_titles_early  # already computed above
    inferred_skills = run_all_inference(directly_found, job_titles, education)
    inferred_skills = detect_all_proficiencies(inferred_skills, raw_text)

    for s in inferred_skills:
        reasoning_trace.append({
            "layer":     "5_inference",
            "skill":     s["skill"],
            "source":    s.get("source", ""),
            "reasoning": s.get("reasoning", ""),
        })

    # ── Aggregate by category ─────────────────────────────────────────
    all_skills  = directly_found + inferred_skills
    by_category = defaultdict(list)
    for s in all_skills:
        by_category[s["category"]].append(s["skill"])

    experience_info = compute_total_experience(experience)
    domains         = detected_domains  # already computed in layer 3b

    return {
        "skills":             directly_found,
        "inferred_skills":    inferred_skills,
        "skills_by_category": dict(by_category),
        "experience_info":    experience_info,
        "domains":            domains,
        "languages":          parsed_resume.get("languages", []),
        "confidence_scores":  {s["skill"]: s["confidence"] for s in all_skills},
        "reasoning_trace":    reasoning_trace,
    }
