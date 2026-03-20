"""
main.py
-------
Entry point for the resume skill extraction pipeline.

Usage:
    python main.py resume1.html resume2.html ...
    LLM_BACKEND=groq python main.py resume.html
"""

import json
import sys
import os
from resume_parser import parse_resume_pdf, parse_resume_html, parse_resume_plaintext
from skill_extractor import extract_all_skills


def load_resume(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def process_resumes(filepaths: list) -> list:
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

        extraction = extract_all_skills(parsed_resume=parsed)
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
    files = sys.argv[1:] if len(sys.argv) > 1 else []

    if not files:
        print("Usage: python main.py resume1.html resume2.html ...\n")
        sys.exit(1)

    try:
        profiles = process_resumes(files)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    for profile in profiles:
        print_profile(profile)

    export_json(profiles)
