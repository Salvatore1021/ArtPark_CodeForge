"""
main.py
-------
Entry point for the resume skill extraction pipeline.

Usage:
    python main.py resume.pdf
    python main.py resume.pdf enhanced_taxonomy.py
    python main.py resume1.pdf resume2.pdf --taxonomy enhanced_taxonomy.py
"""

import json
import sys
import os
from resume_parser import parse_resume_pdf, parse_resume_html, parse_resume_plaintext
from skill_extractor import extract_all_skills


def load_resume(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_taxonomy_skills(taxonomy_path: str) -> list:
    """Load all unique skill names from the taxonomy file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("enhanced_taxonomy", taxonomy_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    taxonomy_data = mod.ENHANCED_TAXONOMY_DATA

    skill_set = set()
    for domain, roles in taxonomy_data.items():
        for role, data in roles.items():
            if not isinstance(data, dict):
                continue
            for key, val in data.items():
                if key == "WEIGHTS" and isinstance(val, dict):
                    for s in val:
                        if "\n" not in s and len(s) > 1:
                            skill_set.add(s)
                elif key.startswith("Unnamed") and isinstance(val, list):
                    for s in val:
                        if "\n" not in s and len(s) > 1:
                            skill_set.add(s)
    print(f"  [taxonomy] Loaded {len(skill_set)} skills from {os.path.basename(taxonomy_path)}")
    return sorted(skill_set)


def process_resumes(filepaths: list, taxonomy_skills: list = None) -> list:
    profiles = []
    for fp in filepaths:
        if fp.lower().endswith(".pdf"):
            parsed = parse_resume_pdf(fp)
        else:
            raw = load_resume(fp)
            if fp.endswith(".html") or raw.strip().startswith('"<') or raw.strip().startswith("<"):
                parsed = parse_resume_html(raw)
            else:
                sections = parse_resume_plaintext(raw)
                parsed = {
                    "raw_text": raw, "summary": sections.get("summary", ""),
                    "experience": [], "skills": [], "languages": [],
                    "accomplishments": [], "additional": "",
                }

        extraction = extract_all_skills(
            parsed_resume=parsed,
            taxonomy_skills=taxonomy_skills,
        )
        profiles.append({
            "source_file": os.path.basename(fp),
            **extraction,
        })

    return profiles


def print_profile(profile: dict):
    W = 70
    print("=" * W)
    print(f"  FILE : {profile['source_file']}")
    print("=" * W)

    print(f"\n  LLM  : {profile['model_used']} via {profile['backend_used']}")

    # Experience
    ei = profile["experience_info"]
    print(f"\n EXPERIENCE")
    print(f"   Total Years  : {ei['total_years']}")
    print(f"   Seniority    : {ei['seniority_level']}")
    for title, yrs in ei["years_per_title"].items():
        if title:
            print(f"     {title}: {yrs} yrs")

    # Domains
    if profile["domains"]:
        print(f"\n DOMAINS  (by skill count)")
        for i, d in enumerate(profile["domains"], 1):
            count = len(profile["skills_by_category"].get(d, []))
            print(f"   {i}. {d}  ({count} skills)")

    # Skills by category
    print(f"\n EXTRACTED SKILLS")
    for cat, skill_names in profile["skills_by_category"].items():
        skill_objs = [s for s in profile["skills"] if s["skill"] in skill_names]
        if not skill_objs:
            continue
        label = cat.replace("_", " ").title()
        print(f"\n   [{label}]")
        for s in skill_objs:
            prof = s.get("proficiency", "intermediate")
            conf = s.get("confidence", 0.7)
            print(f"     • {s['skill']:<32} {prof:<14} conf={conf:.2f}")

    # Reasoning trace summary
    print(f"\n REASONING TRACE")
    print(f"   {'LLM extraction':<35}: {len(profile['reasoning_trace'])} decisions")

    print()


def export_json(profiles: list, path: str = None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, default=str)
    print(f"  JSON saved to: {path}\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse --taxonomy flag
    taxonomy_path  = None
    resume_files   = []
    skip_next      = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--taxonomy" and i + 1 < len(args):
            taxonomy_path = args[i + 1]
            skip_next     = True
        elif arg.endswith(".py") and os.path.exists(arg):
            taxonomy_path = arg
        else:
            resume_files.append(arg)

    if not resume_files:
        print("Usage: python main.py resume.pdf")
        print("       python main.py resume.pdf enhanced_taxonomy.py")
        print("       python main.py resume1.pdf resume2.pdf --taxonomy enhanced_taxonomy.py")
        sys.exit(1)

    # Load taxonomy if provided
    taxonomy_skills = None
    if taxonomy_path:
        taxonomy_skills = load_taxonomy_skills(taxonomy_path)

    try:
        profiles = process_resumes(resume_files, taxonomy_skills)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    for profile in profiles:
        print_profile(profile)

    export_json(profiles)
