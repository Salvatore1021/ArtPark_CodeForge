"""
skill_extractor.py — LLM-based skill extraction from resume and job description text.

Uses OpenAI-compatible API (gpt-4o / gpt-4o-mini).
Falls back to regex/taxonomy matching if API is unavailable.
"""

import os
import json
import re
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESUME_SYSTEM_PROMPT = """You are an expert HR analyst and skill extractor.
Given a resume, extract ALL technical and soft skills mentioned.
Respond ONLY with a valid JSON object — no markdown, no explanation.

Output schema:
{
  "candidate_name": "string or null",
  "years_total_experience": number or null,
  "current_role": "string or null",
  "skills": [
    {
      "skill_name": "canonical skill name (e.g. Python, Machine Learning, SQL)",
      "category": "programming | data-science | data-engineering | cloud-devops | web | soft-skills | security | other",
      "proficiency_level": 1-5,
      "evidence": "brief quote or phrase from resume that supports this skill"
    }
  ]
}

Proficiency levels:
1=Aware (mentioned once, no depth), 2=Beginner (coursework/exposure),
3=Intermediate (used in projects, some production work),
4=Advanced (production systems, led work, multiple years), 5=Expert (published, architected, thought leader)

Be generous in extraction — capture every skill implied by the experience, tools, and education.
Normalize skill names to canonical form (e.g., "sklearn" → "Scikit-learn", "pytorch" → "PyTorch").
"""

JD_SYSTEM_PROMPT = """You are an expert job description analyst.
Given a job description (JD), extract the required and preferred skills.
Respond ONLY with a valid JSON object — no markdown, no explanation.

Output schema:
{
  "job_title": "string",
  "department": "string or null",
  "seniority": "junior | mid | senior | lead | principal | director",
  "skills": [
    {
      "skill_name": "canonical skill name",
      "category": "programming | data-science | data-engineering | cloud-devops | web | soft-skills | security | other",
      "required_level": 1-5,
      "is_mandatory": true/false,
      "context": "why this skill is needed (brief)"
    }
  ]
}

is_mandatory=true for "required/must-have", false for "preferred/nice-to-have".
Normalize skill names to canonical form.
Extract EVERY skill mentioned — both explicit and implied by technologies listed.
"""

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_llm(system_prompt: str, user_content: str, max_tokens: int = 2000) -> dict[str, Any]:
    """Call OpenAI-compatible API and parse JSON response."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — using regex fallback")
        return {}

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content[:12000]},  # token guard
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,  # low temp for structured extraction
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        return json.loads(raw_text)


# ---------------------------------------------------------------------------
# Regex / taxonomy fallback
# ---------------------------------------------------------------------------

def _load_taxonomy_skills() -> list[str]:
    """Load all canonical skill names from taxonomy for fallback matching."""
    try:
        taxonomy_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "skill_taxonomy.json"
        )
        with open(taxonomy_path, "r") as f:
            taxonomy = json.load(f)
        skills = []
        for category in taxonomy["categories"].values():
            for skill_name, skill_data in category["skills"].items():
                skills.append(skill_name)
                skills.extend(skill_data.get("aliases", []))
        return skills
    except Exception as e:
        logger.error(f"Taxonomy load failed: {e}")
        return []


def _regex_extract_skills(text: str) -> list[dict]:
    """Simple regex-based skill extraction as fallback."""
    taxonomy_skills = _load_taxonomy_skills()
    text_lower = text.lower()
    found = []
    seen = set()

    for skill in taxonomy_skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower) and skill.lower() not in seen:
            seen.add(skill.lower())
            found.append({
                "skill_name": skill,
                "category": "other",
                "proficiency_level": 2,
                "evidence": f"Matched '{skill}' in text",
            })
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_skills_from_resume(resume_text: str) -> dict[str, Any]:
    """
    Extract structured skills from resume text.
    Returns a dict with keys: candidate_name, years_total_experience,
    current_role, skills (list of skill objects).
    """
    try:
        result = await _call_llm(RESUME_SYSTEM_PROMPT, resume_text, max_tokens=2500)
        if result and "skills" in result:
            result["extraction_method"] = "llm"
            return result
    except Exception as e:
        logger.error(f"LLM resume extraction failed: {e}")

    # Fallback
    return {
        "candidate_name": None,
        "years_total_experience": None,
        "current_role": None,
        "skills": _regex_extract_skills(resume_text),
        "extraction_method": "regex_fallback",
    }


async def extract_skills_from_jd(jd_text: str) -> dict[str, Any]:
    """
    Extract structured skills from job description text.
    Returns a dict with keys: job_title, seniority, skills (list).
    """
    try:
        result = await _call_llm(JD_SYSTEM_PROMPT, jd_text, max_tokens=2500)
        if result and "skills" in result:
            result["extraction_method"] = "llm"
            return result
    except Exception as e:
        logger.error(f"LLM JD extraction failed: {e}")

    # Fallback
    return {
        "job_title": "Unknown Role",
        "department": None,
        "seniority": "mid",
        "skills": [
            {
                "skill_name": s["skill_name"],
                "category": s["category"],
                "required_level": 3,
                "is_mandatory": True,
                "context": "Extracted via regex fallback",
            }
            for s in _regex_extract_skills(jd_text)
        ],
        "extraction_method": "regex_fallback",
    }
