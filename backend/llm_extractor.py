"""
llm_extractor.py
----------------
Pure LLM-based skill extraction via Ollama.

The LLM defines everything:
  - skill name
  - category
  - proficiency level
  - confidence score
  - reasoning (evidence quote)

No hardcoded skill lists. No taxonomy. No rules.

Requires Ollama running locally:
  install from ollama.com, then run: ollama pull llama3.2

Usage:
    python main.py resume.html
    OLLAMA_MODEL=llama3.1 python main.py resume.html
"""

import os
import json
import re
import urllib.request


# ── Backend config ────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

CHUNK_SIZE   = 1000   # smaller chunks = fewer skills per response = less output tokens needed


# ── The single extraction prompt ──────────────────────────────────────────────
#
# This prompt does ALL the work — no taxonomy, no rules, no post-processing.
# The LLM decides categories, proficiency, and confidence entirely on its own.

SYSTEM_PROMPT = """You are an expert resume analyst. Extract only skills that are explicitly present in the provided resume text.

STRICT RULES:
- Only extract skills you can directly quote from the resume text.
- If you cannot find a direct quote supporting a skill, do NOT include it.
- Never invent or assume skills. If the resume says "basic Python", proficiency is beginner, not advanced.
- reasoning MUST be a direct quote from the resume text (under 15 words). Never write "not specified".
- skill name must be 1-4 words, lowercase. No sentences.
- Do NOT extract: hardware, file formats, course names, degree names, company names, job titles.
- Extract the concept, not the implementation detail:
    "cuda-accelerated batch processing" -> "cuda"
    "monte carlo simulation" -> "monte carlo simulation" (this IS a valid skill)
    "indexed sqlite schema" -> "sqlite"

For each skill determine:
1. skill       : short skill name, 1-4 words, lowercase
2. category    : domain (e.g. "programming", "machine learning", "quantitative finance", "mathematics")
3. proficiency : beginner | intermediate | advanced | expert — based strictly on resume wording
4. confidence  : 0.9 = explicitly listed, 0.7 = clearly used in project, 0.5 = weakly implied
5. reasoning   : verbatim quote from the resume under 15 words — REQUIRED, never "not specified"

Return ONLY a valid JSON array. No explanation. No markdown. No preamble.

Output format:
[
  {
    "skill": "skill name",
    "category": "category name",
    "proficiency": "intermediate",
    "confidence": 0.9,
    "reasoning": "exact short quote from resume"
  }
]"""


def _chunk_text(plain: str) -> list:
    """Split plain text into chunks of CHUNK_SIZE chars, splitting on newlines where possible."""
    chunks = []
    while len(plain) > CHUNK_SIZE:
        split_at = plain.rfind("\n", 0, CHUNK_SIZE)
        if split_at == -1:
            split_at = CHUNK_SIZE
        chunks.append(plain[:split_at].strip())
        plain = plain[split_at:].strip()
    if plain:
        chunks.append(plain)
    return chunks


def _build_messages(chunk: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Extract all skills from this resume section:\n\n{chunk}"}
    ]


# ── Backend: Ollama ───────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _call_ollama(raw_text: str) -> list:
    chunks = _chunk_text(raw_text)
    print(f"  [ollama] Processing {len(chunks)} chunk(s) ({len(raw_text)} chars total)")

    all_skills = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [ollama] Chunk {i}/{len(chunks)}...")
        payload = json.dumps({
            "model":    OLLAMA_MODEL,
            "messages": _build_messages(chunk),
            "stream":   False,
            "options":  {"temperature": 0.1, "num_predict": 4096}
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            content = json.loads(r.read())["message"]["content"]
            print(f"  [ollama] Chunk {i} raw (first 200 chars): {content[:200]}")
            chunk_skills = _parse(content)
            print(f"  [ollama] Chunk {i} parsed {len(chunk_skills)} skills")
            if chunk_skills:
                all_skills.extend(chunk_skills)

    return all_skills


# ── Response parser ───────────────────────────────────────────────────────────

def _parse(raw: str) -> list:
    """Extract JSON array from LLM response, handling common formatting issues."""
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        # Try fixing single quotes
        try:
            return json.loads(match.group().replace("'", '"'))
        except Exception:
            return []


# ── Main function ─────────────────────────────────────────────────────────────

def extract_skills_llm(raw_text: str) -> dict:
    """
    Extract all skills from resume text using Ollama.

    The LLM determines skill names, categories, proficiency, and confidence
    entirely from the text — no taxonomy or rules involved.

    Returns
    -------
    {
        "skills":        list[dict]  — extracted skills with full metadata
        "llm_available": bool
        "backend_used":  str
        "model_used":    str
        "error":         str | None
    }
    """
    result = {
        "skills":        [],
        "llm_available": False,
        "backend_used":  "ollama",
        "model_used":    OLLAMA_MODEL,
        "error":         None,
    }

    if not _ollama_available():
        result["error"] = "Ollama is not running. Start it with: ollama serve"
        return result

    try:
        raw_skills = _call_ollama(raw_text)
    except Exception as e:
        result["error"] = f"[ollama] {e}"
        return result

    if not raw_skills:
        result["error"] = "[ollama] Empty or unparseable response"
        return result

    # Normalise and deduplicate
    seen, skills = set(), []
    for s in raw_skills:
        name = str(s.get("skill", "")).lower().strip()
        if not name or name in seen:
            continue
        seen.add(name)
        skills.append({
            "skill":       name,
            "category":    str(s.get("category",    "general")).lower().strip(),
            "proficiency": str(s.get("proficiency", "intermediate")).lower().strip(),
            "confidence":  round(float(s.get("confidence", 0.7)), 2),
            "reasoning":   str(s.get("reasoning",   "")).strip(),
            "method":      "llm_extracted",
            "backend":     "ollama",
        })

    result["skills"]        = skills
    result["llm_available"] = True
    return result


# ── Taxonomy mapper ───────────────────────────────────────────────────────────

def map_to_taxonomy(skills: list, taxonomy_skills: list,
                    batch_size: int = 10) -> dict:
    """
    Semantically map a list of extracted skills to the closest taxonomy skill
    using Ollama. Returns a dict: { original_skill: matched_taxonomy_skill_or_None }

    taxonomy_skills : flat list of all skill names from the taxonomy file.
    """
    # Case-insensitive lookup for validation
    skill_lookup = {s.lower().strip(): s for s in taxonomy_skills}
    taxonomy_str = "\n".join(f"- {s}" for s in taxonomy_skills)
    mapping      = {}

    skill_names = [s["skill"] for s in skills]

    for i in range(0, len(skill_names), batch_size):
        batch      = skill_names[i:i + batch_size]
        resume_str = "\n".join(f"- {s}" for s in batch)

        prompt = f"""You are a skill mapping expert. Map each resume skill to the single most semantically similar skill from the taxonomy list.

TAXONOMY SKILLS:
{taxonomy_str}

RESUME SKILLS TO MAP:
{resume_str}

Rules:
- Match by meaning, not spelling.
- Examples:
    "employee relations" -> "Management of Personnel Resources"
    "payroll" -> "Management of Financial Resources"
    "scheduling" -> "Time Management"
    "hiring" -> "Management of Personnel Resources"
    "bookkeeping" -> "Management of Financial Resources"
    "pytorch" -> "Python"
    "sqlite" -> "SQL"
    "debugging" -> "Troubleshooting"
    "coaching" -> "Instructing"
    "document management" -> "Writing"
    "osha reporting" -> "Quality Control Analysis"
- Return null if there is genuinely no reasonable match.
- Return ONLY valid JSON object. No explanation. No markdown. No preamble.

Output format:
{{
  "resume_skill_1": "Matched Taxonomy Skill",
  "resume_skill_2": null
}}"""

        payload = json.dumps({
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options":  {"temperature": 0.0, "num_predict": 512}
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        print(f"  [taxonomy] Mapping batch {i//batch_size + 1} ({len(batch)} skills)...")
        with urllib.request.urlopen(req, timeout=120) as r:
            raw     = json.loads(r.read())["message"]["content"]
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    batch_map = json.loads(match.group())
                    for orig, matched in batch_map.items():
                        if matched:
                            # Normalize to canonical casing
                            canonical = skill_lookup.get(
                                matched.lower().strip(), matched
                            )
                            mapping[orig] = canonical if canonical in taxonomy_skills else None
                        else:
                            mapping[orig] = None
                except json.JSONDecodeError:
                    pass

    return mapping
