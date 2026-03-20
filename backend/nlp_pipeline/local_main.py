"""
main.py
-------
Entry point: runs the full 5-layer extraction pipeline on one or
more resume files and prints a structured report.

Usage:
    python main.py resume1.html resume2.html ...
"""

import json
import sys
import os
from resume_parser import parse_resume_html, parse_resume_plaintext
from skill_extractor import extract_all_skills


def load_resume(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def process_resumes(filepaths):
    """
    Full batch pipeline:
      1. Parse HTML/text into structured sections
      2. Run 5-layer skill extraction (TF-IDF uses all resumes as corpus)
      3. Return enriched profile dicts
    """
    parsed_resumes = []
    for fp in filepaths:
        raw = load_resume(fp)
        if fp.endswith(".html") or raw.strip().startswith('"<') or raw.strip().startswith("<"):
            parsed = parse_resume_html(raw)
        else:
            sections = parse_resume_plaintext(raw)
            parsed = {
                "raw_text": raw, "summary": sections.get("summary", ""),
                "experience": [], "education": [], "skills": [],
                "languages": [], "additional": "",
            }
        parsed_resumes.append(parsed)

    all_texts = [p["raw_text"] for p in parsed_resumes]

    profiles = []
    for i, (fp, parsed) in enumerate(zip(filepaths, parsed_resumes)):
        extraction = extract_all_skills(
            parsed_resume=parsed,
            all_resume_texts=all_texts,
            resume_index=i,
        )
        profiles.append({
            "source_file": os.path.basename(fp),
            **extraction,
        })
    return profiles


def print_profile(profile):
    W = 70
    print("=" * W)
    print(f"  FILE: {profile['source_file']}")
    print("=" * W)

    # Experience
    ei = profile["experience_info"]
    print(f"\n EXPERIENCE")
    print(f"   Total Years  : {ei['total_years']}")
    print(f"   Seniority    : {ei['seniority_level']}")
    for title, yrs in ei["years_per_title"].items():
        if title:
            print(f"     {title}: {yrs} yrs")

    # Domain expertise
    if profile["domains"]:
        print(f"\n DOMAIN EXPERTISE  (ranked by evidence strength)")
        for i, d in enumerate(profile["domains"][:5], 1):
            print(f"   {i}. {d}")

    # Skills by category (directly extracted)
    print(f"\n EXTRACTED SKILLS  (Layer 1-3)")
    for cat, skills in profile["skills_by_category"].items():
        direct = [s for s in profile["skills"] if s["skill"] in skills]
        if direct:
            label = cat.replace("_", " ").title()
            print(f"   [{label}]")
            for s in direct:
                prof  = s["proficiency"]["level"]
                conf  = s["confidence"]
                method = s["method"]
                print(f"     • {s['skill']:<30} proficiency={prof:<14} "
                      f"conf={conf:.2f}  [{method}]")

    # Inferred skills
    if profile["inferred_skills"]:
        print(f"\n INFERRED SKILLS  (Layer 5 - co-occurrence / role / education)")
        for s in profile["inferred_skills"]:
            print(f"   • {s['skill']:<30} conf={s['confidence']:.2f}  "
                  f"← {s.get('reasoning','')[:60]}")

    # Languages
    if profile["languages"]:
        print(f"\n LANGUAGES: {', '.join(profile['languages'])}")

    # High confidence skills
    high = [s for s in profile["skills"] if s["confidence"] >= 0.85]
    if high:
        print(f"\n HIGH-CONFIDENCE SKILLS  (>= 0.85, found by multiple layers)")
        for s in high:
            print(f"   • {s['skill']}  [{s['category']}]  conf={s['confidence']:.2f}")

    # Reasoning trace summary
    trace = profile["reasoning_trace"]
    layer_counts = {}
    for entry in trace:
        layer_counts[entry["layer"]] = layer_counts.get(entry["layer"], 0) + 1

    print(f"\n REASONING TRACE SUMMARY")
    layer_labels = {
        "1_exact_match": "Layer 1 — Exact Match",
        "2_tfidf":       "Layer 2 — TF-IDF",
        "3_semantic":    "Layer 3 — Semantic",
        "4_proficiency": "Layer 4 — Proficiency",
        "5_inference":   "Layer 5 — Inference",
    }
    for layer_key in sorted(layer_counts.keys()):
        label = layer_labels.get(layer_key, layer_key)
        print(f"   {label:<35}: {layer_counts[layer_key]} decisions")

    print()


def export_json(profiles, output_path="output.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, default=str)
    print(f"  JSON saved to: {output_path}\n")


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else []

    if not files:
        print("Usage: python main.py resume1.html resume2.html ...\n")
        print("Running demo on sample resume...\n")
        files = ["/mnt/user-data/uploads/New_Text_Document.html"]

    profiles = process_resumes(files)

    for profile in profiles:
        print_profile(profile)

    export_json(profiles, "/home/claude/resume_parser/output.json")
