"""
skill_classifier.py
--------------------
Classifies benchmark skills as HARD or SOFT and applies category-aware
hard/soft weighting ratios to the fit score.

Why this matters
----------------
An Accountant's "critical thinking" is a soft interpersonal skill worth 30% of
the fit weight.  A Data Scientist's "machine learning" is a hard technical skill
worth 80%.  Using the same flat score for both distorts the evaluation.

Hard/soft split ratios by job category type
-------------------------------------------
  TECH         : 95% hard  / 05% soft
  MANAGEMENT   : 50% hard  / 50% soft
  HEALTHCARE   : 65% hard  / 35% soft
  EDUCATION    : 55% hard  / 45% soft
  CREATIVE     : 60% hard  / 40% soft
  NON_TECH     : 70% hard  / 30% soft   ← the classic default

Skill classification rules (priority order)
--------------------------------------------
1. Skill appears in TECH_SKILL_AUGMENTATION for this job title → HARD
2. Skill is in ONET_HARD_SKILLS set → HARD
3. Skill is in ONET_SOFT_SKILLS set → SOFT
4. Default → SOFT  (conservative: unknown skills don't inflate hard score)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from taxonomy_adapter import TECH_SKILL_AUGMENTATION, get_sector


# ── Category types ────────────────────────────────────────────────────────────

class CategoryType(str, Enum):
    TECH       = "TECH"
    MANAGEMENT = "MANAGEMENT"
    HEALTHCARE = "HEALTHCARE"
    EDUCATION  = "EDUCATION"
    CREATIVE   = "CREATIVE"
    NON_TECH   = "NON_TECH"


# ── Hard/soft split ratios ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillWeightRatio:
    hard:  float
    soft:  float
    label: str

    def __post_init__(self):
        assert abs(self.hard + self.soft - 1.0) < 1e-6, "hard + soft must sum to 1.0"


CATEGORY_RATIOS: dict[str, SkillWeightRatio] = {
    CategoryType.TECH.value:       SkillWeightRatio(hard=0.95, soft=0.05, label="Tech (95/5)"),
    CategoryType.MANAGEMENT.value: SkillWeightRatio(hard=0.50, soft=0.50, label="Management (50/50)"),
    CategoryType.HEALTHCARE.value: SkillWeightRatio(hard=0.65, soft=0.35, label="Healthcare (65/35)"),
    CategoryType.EDUCATION.value:  SkillWeightRatio(hard=0.55, soft=0.45, label="Education (55/45)"),
    CategoryType.CREATIVE.value:   SkillWeightRatio(hard=0.60, soft=0.40, label="Creative (60/40)"),
    CategoryType.NON_TECH.value:   SkillWeightRatio(hard=0.70, soft=0.30, label="Non-Tech (70/30)"),
}


# ── Sector → CategoryType map ─────────────────────────────────────────────────

SECTOR_TO_CATEGORY_TYPE: dict[str, str] = {
    "Data & AI":                  CategoryType.TECH.value,
    "Backend":                    CategoryType.TECH.value,
    "Frontend/Web":               CategoryType.TECH.value,
    "Mobile":                     CategoryType.TECH.value,
    "IT Infrastructure":          CategoryType.TECH.value,
    "Engineering & Tech":         CategoryType.TECH.value,
    "Management & Business":      CategoryType.MANAGEMENT.value,
    "Healthcare":                 CategoryType.HEALTHCARE.value,
    "Education":                  CategoryType.EDUCATION.value,
    "Arts & Media":               CategoryType.CREATIVE.value,
    "Industrial & Construction":  CategoryType.NON_TECH.value,
    "Other Professional Services":CategoryType.NON_TECH.value,
}

# CSV category label → CategoryType (fast exact match)
CATEGORY_TYPE_BY_LABEL: dict[str, str] = {
    # Tech
    "DATA SCIENCE":         CategoryType.TECH.value,
    "DATA SCIENTIST":       CategoryType.TECH.value,
    "MACHINE LEARNING":     CategoryType.TECH.value,
    "AI ENGINEER":          CategoryType.TECH.value,
    "SOFTWARE ENGINEER":    CategoryType.TECH.value,
    "SOFTWARE DEVELOPER":   CategoryType.TECH.value,
    "BACKEND DEVELOPER":    CategoryType.TECH.value,
    "DEVOPS":               CategoryType.TECH.value,
    "DEVOPS ENGINEER":      CategoryType.TECH.value,
    "JAVA DEVELOPER":       CategoryType.TECH.value,
    "FULL STACK":           CategoryType.TECH.value,
    "FULL STACK DEVELOPER": CategoryType.TECH.value,
    "WEB DEVELOPER":        CategoryType.TECH.value,
    "FRONTEND":             CategoryType.TECH.value,
    "FRONTEND DEVELOPER":   CategoryType.TECH.value,
    "IOS DEVELOPER":        CategoryType.TECH.value,
    "FLUTTER DEVELOPER":    CategoryType.TECH.value,
    "NETWORK ADMIN":        CategoryType.TECH.value,
    "DATABASE ADMIN":       CategoryType.TECH.value,
    "DBA":                  CategoryType.TECH.value,
    "UPLOADED_PDF":         CategoryType.TECH.value,
    # Management
    "HR":                   CategoryType.MANAGEMENT.value,
    "HUMAN RESOURCES":      CategoryType.MANAGEMENT.value,
    "MARKETING":            CategoryType.MANAGEMENT.value,
    "OPERATIONS":           CategoryType.MANAGEMENT.value,
    "ANALYST":              CategoryType.MANAGEMENT.value,
    # Healthcare
    "NURSE":                CategoryType.HEALTHCARE.value,
    "NURSING":              CategoryType.HEALTHCARE.value,
    "DOCTOR":               CategoryType.HEALTHCARE.value,
    "HEALTHCARE":           CategoryType.HEALTHCARE.value,
    # Education
    "TEACHER":              CategoryType.EDUCATION.value,
    "EDUCATION":            CategoryType.EDUCATION.value,
    # Non-tech
    "ACCOUNTANT":           CategoryType.NON_TECH.value,
    "ACCOUNTING":           CategoryType.NON_TECH.value,
    "LAWYER":               CategoryType.NON_TECH.value,
    "LEGAL":                CategoryType.NON_TECH.value,
    "ENGINEER":             CategoryType.NON_TECH.value,
    "MECHANICAL ENGINEER":  CategoryType.NON_TECH.value,
    "ELECTRICAL ENGINEER":  CategoryType.NON_TECH.value,
    "CIVIL ENGINEER":       CategoryType.NON_TECH.value,
}


# ── Soft skill set (universal interpersonal / cognitive) ──────────────────────

ONET_SOFT_SKILLS: frozenset = frozenset({
    # Interpersonal
    "active listening", "speaking", "social perceptiveness", "coordination",
    "persuasion", "negotiation", "service orientation", "instructing",
    # Cognitive / self-management
    "critical thinking", "complex problem solving", "judgment and decision making",
    "active learning", "learning strategies", "monitoring", "time management",
    "writing", "reading comprehension",
    # People management
    "management of personnel resources", "management of financial resources",
    "management of material resources",
    # Augmentation soft skills
    "communication", "leadership", "teamwork", "adaptability",
    "problem solving", "attention to detail", "collaboration",
    "analytical thinking", "team collaboration",
})

# ── Hard skill set (domain-specific / technical) ─────────────────────────────

ONET_HARD_SKILLS: frozenset = frozenset({
    # O*NET technical competencies
    "programming", "mathematics", "science", "technology design",
    "systems analysis", "systems evaluation", "operations analysis",
    "operations monitoring", "operation and control", "quality control analysis",
    "equipment maintenance", "troubleshooting", "repairing",
    "equipment selection", "installation",
    # Accounting / finance
    "accounting", "auditing", "taxation", "financial reporting",
    "financial analysis", "budgeting", "payroll",
    # Healthcare / clinical
    "patient care", "clinical research", "good clinical practice",
    "pharmacology", "pharmacovigilance", "medical records",
    "electronic health records", "hipaa",
    # Education / training
    "curriculum design", "instructional design", "lesson planning",
    "student assessment",
    # General domain hard
    "data analysis", "research", "project management", "strategic planning",
    "statistics", "data science", "machine learning", "deep learning",
})


# ── Public helpers ────────────────────────────────────────────────────────────

def classify_skill(skill: str, job_title: str | None = None) -> str:
    """Return 'hard' or 'soft' for a skill string."""
    s = skill.lower().strip()

    # 1. Job-specific tech augmentation → always hard
    if job_title:
        tech_list = {t.lower() for t in TECH_SKILL_AUGMENTATION.get(job_title, [])}
        if s in tech_list:
            return "hard"

    if s in ONET_HARD_SKILLS:
        return "hard"

    if s in ONET_SOFT_SKILLS:
        return "soft"

    return "soft"  # conservative default


def get_category_type(category_label: str, job_title: str | None = None) -> str:
    """Resolve a category label to a CategoryType string value."""
    upper = category_label.upper().strip()
    if upper in CATEGORY_TYPE_BY_LABEL:
        return CATEGORY_TYPE_BY_LABEL[upper]

    if job_title:
        sector = get_sector(job_title)
        if sector and sector in SECTOR_TO_CATEGORY_TYPE:
            return SECTOR_TO_CATEGORY_TYPE[sector]

    return CategoryType.NON_TECH.value


def get_ratio(category_label: str, job_title: str | None = None) -> SkillWeightRatio:
    """Return the SkillWeightRatio for a category/job combination."""
    cat_type = get_category_type(category_label, job_title)
    return CATEGORY_RATIOS[cat_type]


def split_benchmark(
    benchmark: dict[str, float],
    job_title: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Split a benchmark dict into (hard_skills, soft_skills)."""
    hard: dict[str, float] = {}
    soft: dict[str, float] = {}
    for skill, weight in benchmark.items():
        if classify_skill(skill, job_title) == "hard":
            hard[skill] = weight
        else:
            soft[skill] = weight
    return hard, soft


def compute_split_weighted_fit(
    benchmark: dict[str, float],
    confidence_scores: dict,
    category_label: str,
    job_title: str | None = None,
) -> dict:
    """
    Compute the hard/soft split weighted fit score.

    Formula
    -------
    hard_fit  = Σ(onet_w_i × conf_i) / Σ(onet_w_i)   [hard skills only]
    soft_fit  = Σ(onet_w_i × conf_i) / Σ(onet_w_i)   [soft skills only]
    split_fit = ratio.hard × hard_fit + ratio.soft × soft_fit

    Returns
    -------
    dict with split_fit, hard_fit, soft_fit, ratio details, matched/gap lists
    """
    from skill_confidence import STRONG_THRESHOLD

    ratio      = get_ratio(category_label, job_title)
    cat_type   = get_category_type(category_label, job_title)
    hard_bench, soft_bench = split_benchmark(benchmark, job_title)

    def _sub_fit(sub_bench: dict[str, float]) -> float:
        if not sub_bench:
            return 1.0
        total_w  = sum(sub_bench.values())
        scored_w = sum(
            sub_bench[s] * confidence_scores.get(s, {}).get("confidence", 0.0)
            for s in sub_bench
        )
        return scored_w / total_w if total_w > 0 else 0.0

    hard_fit  = _sub_fit(hard_bench)
    soft_fit  = _sub_fit(soft_bench)
    split_fit = ratio.hard * hard_fit + ratio.soft * soft_fit

    hard_matched = [s for s in hard_bench
                    if confidence_scores.get(s, {}).get("confidence", 0.0) >= STRONG_THRESHOLD]
    soft_matched = [s for s in soft_bench
                    if confidence_scores.get(s, {}).get("confidence", 0.0) >= STRONG_THRESHOLD]
    hard_gaps    = [s for s in hard_bench
                    if confidence_scores.get(s, {}).get("confidence", 0.0) < STRONG_THRESHOLD]
    soft_gaps    = [s for s in soft_bench
                    if confidence_scores.get(s, {}).get("confidence", 0.0) < STRONG_THRESHOLD]

    return {
        "split_fit":    round(split_fit, 4),
        "hard_fit":     round(hard_fit, 4),
        "soft_fit":     round(soft_fit, 4),
        "hard_weight":  ratio.hard,
        "soft_weight":  ratio.soft,
        "category_type":cat_type,
        "ratio_label":  ratio.label,
        "hard_skills":  hard_bench,
        "soft_skills":  soft_bench,
        "hard_matched": sorted(hard_matched),
        "soft_matched": sorted(soft_matched),
        "hard_gaps":    sorted(hard_gaps),
        "soft_gaps":    sorted(soft_gaps),
    }


def format_split_report(split_result: dict) -> str:
    """Pretty-print the hard/soft split breakdown."""
    r = split_result
    lines = [
        f"\n{'─'*65}",
        f"  Hard/Soft Split  [{r['ratio_label']}]  Type: {r['category_type']}",
        f"{'─'*65}",
        f"  Hard Skills ({len(r['hard_skills'])})  —  weight {r['hard_weight']*100:.0f}%",
        f"    Fit score  :  {r['hard_fit']:.4f}",
        f"    Matched    :  {', '.join(r['hard_matched']) or 'none'}",
        f"    Gaps       :  {', '.join(r['hard_gaps'][:7]) or 'none'}",
        f"",
        f"  Soft Skills ({len(r['soft_skills'])})  —  weight {r['soft_weight']*100:.0f}%",
        f"    Fit score  :  {r['soft_fit']:.4f}",
        f"    Matched    :  {', '.join(r['soft_matched']) or 'none'}",
        f"    Gaps       :  {', '.join(r['soft_gaps'][:7]) or 'none'}",
        f"",
        f"  ► Split-Weighted Fit : {r['split_fit']:.4f}",
        f"    = {r['hard_weight']*100:.0f}% × {r['hard_fit']:.4f}  (hard)"
        f"   +   {r['soft_weight']*100:.0f}% × {r['soft_fit']:.4f}  (soft)",
        f"{'─'*65}",
    ]
    return "\n".join(lines)
