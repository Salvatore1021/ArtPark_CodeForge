"""
taxonomy_adapter.py
--------------------
Wraps ENHANCED_TAXONOMY_DATA and exposes clean APIs used by the rest of the engine.

Taxonomy structure
------------------
Two kinds of job entries coexist in the taxonomy:

  A) O*NET-weighted roles (Industrial, Engineering, Healthcare, Education, etc.)
     → "WEIGHTS": {"Critical Thinking": 4.0, "Programming": 3.75, ...}
     → Benchmark is built directly from these weights.

  B) Tech-tool roles (Data & AI, Frontend/Web, Backend, Mobile, IT Infrastructure)
     → "WEIGHTS": {}  (empty)
     → Skills live in "Unnamed: 2" ... "Unnamed: 8" flat lists

Merging strategy
----------------
For every job title the final benchmark is assembled in this priority order:

  1. Own O*NET WEIGHTS  (if present)
  2. Proxy O*NET role's WEIGHTS  (if own WEIGHTS are empty — via PROXY_MAP)
  3. Own tech lists from "Unnamed: X" columns  (weight = DEFAULT_TECH_WEIGHT)
  4. TECH_SKILL_AUGMENTATION  (curated tech stacks for roles missing tool lists)

This ensures every role gets both competency weights (critical thinking, programming…)
AND a realistic set of technology keywords so extraction and TF-IDF matching work.

Public API
----------
get_all_skills(min_weight)           → sorted list of every known skill string
get_job_benchmark(title, top_n)      → {skill_lower: weight}
build_weighted_benchmark_text(title) → weighted text string for TF-IDF
find_job_title(query)                → fuzzy-match to nearest job title
list_sectors() / list_jobs(sector)   → discovery helpers
get_tech_skills(title)               → flat tech tool list
SKILL_TO_JOBS                        → reverse map  skill → [job_titles]
"""

from __future__ import annotations
import re
from functools import lru_cache
from enhanced_taxonomy import ENHANCED_TAXONOMY_DATA

_EMOJI_RE = re.compile(
    "["
    u"\U0001F600-\U0001FFFF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def _clean(name: str) -> str:
    return _EMOJI_RE.sub("", name).strip()


# ── Default weight assigned to tech-list skills ───────────────────────────────
DEFAULT_TECH_WEIGHT      = 3.0   # base weight for Unnamed: tool-list skills
AUGMENTATION_TECH_WEIGHT = 3.5   # slightly higher so curated tech stacks rank
                                  # above the lowest O*NET competencies in top-N


# ── Supplementary tech skills for roles that have no Unnamed: tool lists ──────
# These are curated tool/framework stacks per role, added at DEFAULT_TECH_WEIGHT.
# Keys must match the cleaned job title string after emoji removal.
TECH_SKILL_AUGMENTATION: dict[str, list[str]] = {
    "Machine Learning": [
        "python", "pytorch", "tensorflow", "keras", "scikit-learn",
        "pandas", "numpy", "matplotlib", "seaborn", "jupyter",
        "sql", "statistics", "machine learning", "deep learning",
        "natural language processing", "computer vision", "feature engineering",
        "model tuning", "transformers", "hugging face",
        "git / github", "docker", "aws", "gcp", "azure",
        "spark", "data visualization", "mlflow", "wandb",
    ],
    "Software Engineer": [
        "python", "java", "javascript", "typescript", "c++", "go",
        "algorithms", "data structures", "system design",
        "rest apis", "graphql", "sql", "postgresql", "mongodb",
        "docker", "kubernetes", "ci/cd", "git / github",
        "linux", "aws", "unit testing", "agile", "microservices",
    ],
    "Database Administrator": [
        "sql", "postgresql", "mysql", "oracle db", "mongodb",
        "redis", "elasticsearch", "cassandra", "ms sql server",
        "backup and recovery", "performance tuning", "indexing",
        "data modeling", "replication", "linux", "stored procedures",
    ],
    "Full Stack Developer": [
        "javascript", "typescript", "react", "node.js", "html5", "css3",
        "python", "sql", "postgresql", "mongodb", "redis",
        "docker", "aws", "git / github", "rest apis", "graphql",
        "webpack", "next.js", "tailwind",
    ],
    "Backend Developer": [
        "python", "java", "javascript", "sql", "postgresql", "mongodb",
        "redis", "docker", "kubernetes", "rest apis", "microservices",
        "git / github", "linux", "ci/cd", "aws", "message queues",
    ],
    "PHP Developer": [
        "php", "laravel", "symfony", "mysql", "postgresql",
        "rest apis", "html5", "css3", "javascript", "git / github",
        "linux", "docker", "composer", "wordpress",
    ],
    "Django Developer": [
        "python", "django", "django rest framework", "postgresql",
        "redis", "celery", "sql", "rest apis", "docker",
        "git / github", "linux", "aws", "nginx", "gunicorn",
    ],
    "JavaScript Developer": [
        "javascript", "typescript", "html5", "css3", "react",
        "angular", "vue.js", "node.js", "webpack", "rest apis",
        "graphql", "git / github", "npm / yarn", "firebase",
        "mongodb", "postgresql", "jest", "cypress",
    ],
    "Flutter Developer": [
        "flutter", "dart", "ios", "android", "firebase",
        "rest apis", "git / github", "state management",
        "bloc", "provider", "sqlite", "push notifications",
    ],
    "iOS Developer": [
        "swift", "objective-c", "xcode", "ios", "core data",
        "rest apis", "git / github", "swiftui", "cocoapods",
        "combine", "uikit", "app store connect",
    ],
    "Network Administrator": [
        "tcp/ip", "dns", "dhcp", "vpn", "firewall", "linux",
        "windows server", "active directory", "network monitoring",
        "cisco", "troubleshooting", "it support", "vmware",
        "backup and recovery", "snmp", "vlan",
    ],
}


# ── Proxy map: tech-only job → best-matching O*NET-weighted role ──────────────
# When a tech role has empty WEIGHTS, competency weights are inherited from proxy.
PROXY_MAP: dict[str, str] = {
    "Machine Learning":     "Computer and Information Research Scientists",
    "Software Engineer":    "Computer Programmers",
    "Database Administrator": "Database Administrators and Architects",
    "JavaScript Developer": "Web Developers",
    "Full Stack Developer": "Web Developers",
    "Backend Developer":    "Computer Programmers",
    "DevOps Engineer":      "Computer Network Support Specialists",
    "Java Developer":       "Computer Programmers",
    "PHP Developer":        "Computer Programmers",
    "Django Developer":     "Computer Programmers",
    "Flutter Developer":    "Software Developers and Software Quality Assurance Analysts and Testers",
    "iOS Developer":        "Software Developers and Software Quality Assurance Analysts and Testers",
    "Network Administrator":"Computer Network Support Specialists",
}


# ── Internal flat index ───────────────────────────────────────────────────────
_JOB_INDEX:  dict[str, dict[str, float]] = {}   # clean_title → own {skill: weight}
_JOB_SECTOR: dict[str, str]              = {}   # clean_title → sector
_JOB_TECH:   dict[str, list[str]]        = {}   # clean_title → [raw tech strings]


def _build_index() -> None:
    for sector, jobs in ENHANCED_TAXONOMY_DATA.items():
        if sector == "Metadata/Header":
            continue
        for raw_title, body in jobs.items():
            title   = _clean(raw_title)
            weights: dict[str, float] = {}
            tech:    list[str]        = []

            w_dict = body.get("WEIGHTS", {})
            if isinstance(w_dict, dict):
                for skill, w in w_dict.items():
                    weights[skill.lower().strip()] = float(w)

            for key, val in body.items():
                if key.startswith("Unnamed:") and isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            tech.append(item.strip())

            _JOB_INDEX[title]  = weights
            _JOB_SECTOR[title] = sector
            _JOB_TECH[title]   = tech


_build_index()


# ── Merged weight builder ─────────────────────────────────────────────────────

def _merged_weights(title: str) -> dict[str, float]:
    """
    Return the full {skill_lower: weight} for a title by merging in order:
    1. Own O*NET WEIGHTS
    2. Proxy O*NET WEIGHTS  (if own is empty)
    3. Own Unnamed: tech lists  (weight = DEFAULT_TECH_WEIGHT)
    4. TECH_SKILL_AUGMENTATION  (curated stacks for tech roles)
    """
    own_weights = _JOB_INDEX.get(title, {})
    own_tech    = _JOB_TECH.get(title, [])

    if own_weights:
        # Has real O*NET weights — start from those
        merged: dict[str, float] = dict(own_weights)
    else:
        # Inherit competency weights from proxy role
        proxy_title = PROXY_MAP.get(title)
        merged = {}
        if proxy_title and proxy_title in _JOB_INDEX:
            merged = dict(_JOB_INDEX[proxy_title])
        # Also pull proxy's own tech list
        if proxy_title and _JOB_TECH.get(proxy_title):
            for t in _JOB_TECH[proxy_title]:
                k = t.lower().strip()
                if k not in merged:
                    merged[k] = DEFAULT_TECH_WEIGHT

    # Layer 3: this role's own Unnamed: tech list
    for t in own_tech:
        k = t.lower().strip()
        if k not in merged:
            merged[k] = DEFAULT_TECH_WEIGHT

    # Layer 4: curated augmentation tech skills (slightly higher weight)
    for t in TECH_SKILL_AUGMENTATION.get(title, []):
        k = t.lower().strip()
        if k not in merged:
            merged[k] = AUGMENTATION_TECH_WEIGHT

    return merged


# ── Reverse lookup  skill → [job titles] ─────────────────────────────────────
SKILL_TO_JOBS: dict[str, list[str]] = {}
for _title in _JOB_INDEX:
    for _skill in _merged_weights(_title):
        SKILL_TO_JOBS.setdefault(_skill, []).append(_title)


# ── Public API ────────────────────────────────────────────────────────────────

def list_sectors() -> list[str]:
    """Return sorted sector names (excl. Metadata/Header)."""
    return sorted({s for s in ENHANCED_TAXONOMY_DATA if s != "Metadata/Header"})


def list_jobs(sector: str | None = None) -> list[str]:
    """Return clean job titles, optionally filtered to a sector."""
    if sector is None:
        return sorted(_JOB_INDEX.keys())
    s_low = sector.lower()
    return sorted(t for t, s in _JOB_SECTOR.items() if s.lower() == s_low)


def find_job_title(query: str) -> str | None:
    """
    Fuzzy-match a query string to the nearest known job title.
    Tries: exact → starts-with → substring → token overlap.
    """
    q = query.lower().strip()
    for title in _JOB_INDEX:
        if title.lower() == q:
            return title
    for title in _JOB_INDEX:
        if title.lower().startswith(q):
            return title
    for title in _JOB_INDEX:
        if q in title.lower():
            return title
    q_tokens  = set(q.split())
    best, best_score = None, 0
    for title in _JOB_INDEX:
        score = len(q_tokens & set(title.lower().split()))
        if score > best_score:
            best, best_score = title, score
    return best if best_score > 0 else None


@lru_cache(maxsize=512)
def get_job_benchmark(job_title: str, top_n: int = 20) -> dict[str, float]:
    """
    Return the top-N skills by weight for a job title.
    Uses the fully merged benchmark (O*NET + tech lists + augmentation).
    Falls back to fuzzy match if title not found exactly.
    """
    title = job_title if job_title in _JOB_INDEX else find_job_title(job_title)
    if title is None:
        return {}
    merged = _merged_weights(title)
    top    = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(top)


def get_tech_skills(job_title: str) -> list[str]:
    """Return the raw tech tool list for a job title (from Unnamed: columns)."""
    title = job_title if job_title in _JOB_TECH else find_job_title(job_title)
    if not title:
        return []
    return (_JOB_TECH.get(title, [])
            + TECH_SKILL_AUGMENTATION.get(title, []))


def get_all_skills(min_weight: float = 0.0) -> list[str]:
    """
    Return a deduplicated sorted list of every skill string across the taxonomy,
    including O*NET competencies, tech tool lists, and augmentation stacks.
    Optionally filter to skills appearing with weight >= min_weight somewhere.
    """
    skills: set[str] = set()
    for title in _JOB_INDEX:
        for skill, w in _merged_weights(title).items():
            if w >= min_weight:
                skills.add(skill.lower())
    return sorted(skills)


def build_weighted_benchmark_text(job_title: str, top_n: int = 20) -> str:
    """
    Build a benchmark text string for TF-IDF vectorisation where each skill
    is repeated proportional to its importance weight.

    A skill with weight 4.0 appears 4 times; weight 1.0 appears once.
    The resulting TF-IDF vector naturally emphasises critical skills.
    """
    benchmark = get_job_benchmark(job_title, top_n=top_n)
    tokens: list[str] = []
    for skill, weight in benchmark.items():
        repeats = max(1, round(weight))
        tokens.extend([skill] * repeats)
    return " ".join(tokens)


def get_sector(job_title: str) -> str | None:
    """Return the sector for a job title."""
    title = job_title if job_title in _JOB_SECTOR else find_job_title(job_title)
    return _JOB_SECTOR.get(title) if title else None
