"""
main.py
-------
AI-Driven Candidate Onboarding Engine — entry point.

Usage
-----
  python main.py --resume_pdf Aditya_CV.pdf --job_title "Machine Learning"
  python main.py --resume_pdf resume.pdf --recommend
  python main.py --csv data/Resume.csv --id 16237710
  python main.py --list_sectors
  python main.py --list_jobs --sector "Data & AI"
"""

from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import RESUME_CSV_PATH, JOB_DESC_PDF_PATH, OUTPUT_DIR
from taxonomy_adapter import (
    list_sectors,
    list_jobs,
    find_job_title,
    build_weighted_benchmark_text,
    get_job_benchmark,
    get_sector,
)
from resume_parser import extract_text_from_pdf, parse_resume_pdf, resume_profile_to_dataframe_row
from skill_extractor import (
    FULL_SKILL_LIBRARY,
    extract_all_skills,
    extract_skills,
    build_and_fit_vectorizer,
    precompute_skill_vectors,
)
from dependency_graph import build_skill_dependency_graph, get_prerequisites
from candidate_evaluator import evaluate_candidate, batch_evaluate
from gap_prioritizer import prioritize_gaps
from roadmap_builder import build_roadmap
from dashboard import generate_dashboard
from role_recommender import recommend_roles
from report_exporter import export_all
from skill_classifier import classify_skill, split_benchmark, get_category_type

import os
import importlib.util

def load_taxonomy_skills() -> list[str]:
    taxonomy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhanced_taxonomy.py")
    if not os.path.exists(taxonomy_path):
        print("  [taxonomy] enhanced_taxonomy.py not found — skipping mapping")
        return []
    spec = importlib.util.spec_from_file_location("enhanced_taxonomy", taxonomy_path)
    if spec is None or spec.loader is None:
        print("  [taxonomy] Failed to load enhanced_taxonomy.py — skipping mapping")
        return []

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"  [taxonomy] Failed to import enhanced_taxonomy.py: {exc}")
        return []

    taxonomy_data = getattr(mod, "ENHANCED_TAXONOMY_DATA", None)
    if not isinstance(taxonomy_data, dict):
        print("  [taxonomy] ENHANCED_TAXONOMY_DATA missing or invalid — skipping mapping")
        return []

    skill_set = set()
    for domain, roles in taxonomy_data.items():
        for role, data in roles.items():
            if not isinstance(data, dict): continue
            for key, val in data.items():
                if key == "WEIGHTS" and isinstance(val, dict):
                    for s in val:
                        if "\n" not in s and len(s) > 1: skill_set.add(s)
                elif key.startswith("Unnamed") and isinstance(val, list):
                    for s in val:
                        if "\n" not in s and len(s) > 1: skill_set.add(s)
    print(f"  [taxonomy] Loaded {len(skill_set)} skills")
    return sorted(skill_set)


# ── Readiness description ─────────────────────────────────────────────────────

def _readiness_label(grade: str) -> str:
    return {
        "A": "Excellent  — ready to hit the ground running",
        "B": "Good       — strong foundation, minor gaps",
        "C": "Developing — solid base, some key areas to build",
        "D": "Needs Work — important foundations to establish",
        "F": "Beginner   — great starting point for structured learning",
    }.get(grade, "Unknown")


def _importance_label(weight: float) -> str:
    if weight >= 3.75: return "Essential"
    if weight >= 3.25: return "Very Important"
    if weight >= 2.75: return "Important"
    return "Useful"


def _progress_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "#" * filled + "-" * (width - filled)


# ── Pretty print helpers ──────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _line() -> None:
    print(f"  {'-'*58}")


def _prepare_vectorizer(
    resume_df: pd.DataFrame,
    job_text: str = "",
    extra_benchmark_texts: list[str] | None = None,
):
    vectorizer = build_and_fit_vectorizer(
        resume_df,
        job_description_text=job_text,
        skill_library=FULL_SKILL_LIBRARY,
        extra_benchmark_texts=extra_benchmark_texts,
    )
    skill_vectors = precompute_skill_vectors(vectorizer, FULL_SKILL_LIBRARY)
    return vectorizer, skill_vectors


def _proficiency_to_level(proficiency: str) -> int:
    return {
        "beginner": 1,
        "intermediate": 3,
        "advanced": 4,
        "expert": 5,
    }.get((proficiency or "").lower(), 2)


def _weight_to_required_level(weight: float) -> int:
    if weight >= 3.75:
        return 5
    if weight >= 3.25:
        return 4
    if weight >= 2.75:
        return 3
    if weight >= 2.25:
        return 2
    return 1


def _extract_text_from_upload(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".txt" or suffix == ".md":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX support requires python-docx to be installed.") from exc
        document = Document(str(file_path))
        return "\n".join(p.text for p in document.paragraphs).strip()
    raise RuntimeError(f"Unsupported file type: {suffix}. Use PDF, DOCX, TXT, or MD.")


def _parse_resume_upload(file_path: Path) -> dict:
    if file_path.suffix.lower() == ".pdf":
        return parse_resume_pdf(file_path)

    text = _extract_text_from_upload(file_path)
    return {
        "name": file_path.stem.replace("_", " ").strip() or "Candidate",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "summary_text": "",
        "education": [],
        "experience": [],
        "skills_raw": "",
        "certifications": [],
        "projects": [],
        "full_text": text,
    }


def _build_api_payload(
    metrics: dict,
    profile: dict,
    roadmap_df: pd.DataFrame,
    recommendations: list[dict],
    benchmark: dict[str, float],
    llm_output: dict | None,
) -> dict:
    confidence_scores = metrics.get("Confidence_Scores", {})
    llm_lookup = {
        s["skill"].lower(): s for s in (llm_output or {}).get("skills", [])
    }

    candidate_skills = []
    for skill in metrics.get("Extracted_Skills", []):
        info = confidence_scores.get(skill.lower(), {})
        llm_skill = llm_lookup.get(skill.lower(), {})
        candidate_skills.append({
            "skill_name": skill.title(),
            "proficiency_level": _proficiency_to_level(llm_skill.get("proficiency", "")),
            "confidence": round(float(info.get("confidence", 0.0)), 2),
            "evidence": llm_skill.get("reasoning", ""),
        })

    required_skills = [
        {
            "skill_name": skill.title(),
            "required_level": _weight_to_required_level(weight),
            "importance": _importance_label(weight),
            "weight": round(float(weight), 2),
        }
        for skill, weight in sorted(benchmark.items(), key=lambda item: item[1], reverse=True)[:10]
    ]

    roadmap = [
        {
            "week": row["Week"],
            "skill": row["Skill"].title(),
            "objective": row["Objective"],
            "success": row["Success"],
            "weight": row["ONET_Weight"],
            "priority": next(
                (gap["Priority"] for gap in metrics.get("Prioritized_Gaps", []) if gap["Skill"].lower() == row["Skill"].lower()),
                None,
            ),
        }
        for _, row in roadmap_df.iterrows()
    ]

    prioritized_gaps = [
        {
            "priority": gap["Priority"],
            "skill": gap["Skill"].title(),
            "type": "Technical" if classify_skill(gap["Skill"], metrics.get("Job_Title", "")) == "hard" else "Soft Skill",
            "importance": _importance_label(gap["ONET_Weight"]),
            "weight": gap["ONET_Weight"],
            "level": gap["Level"],
            "gap_score": gap.get("Gap_Score", 0.0),
        }
        for gap in metrics.get("Prioritized_Gaps", [])[:12]
    ]
    estimated_weeks = len(roadmap)

    split = metrics.get("Split_Details", {})
    recommended_roles = [
        {
            "title": rec["job_title"],
            "sector": rec["sector"],
            "match_pct": round(rec["composite"] * 100),
            "matched_skills": [s.title() for s in rec.get("matched_skills", [])[:5]],
            "gap_skills": [s.title() for s in rec.get("gap_skills", [])[:4]],
        }
        for rec in recommendations[:3]
    ]

    benchmark_lookup = {skill.lower(): weight for skill, weight in benchmark.items()}
    candidate_lookup = {skill["skill_name"].lower(): skill for skill in candidate_skills}
    graph_nodes: list[dict] = []
    graph_edges: list[dict] = []
    skill_graph = build_skill_dependency_graph()

    def add_node(node_id: str, label: str, group: str, **extra) -> None:
        node = {"id": node_id, "label": label, "group": group}
        node.update(extra)
        graph_nodes.append(node)

    target_role_title = metrics.get("Job_Title", "")
    add_node(
        "role:target",
        target_role_title,
        "target-role",
        sector=get_sector(target_role_title) or "Professional",
        readiness=metrics.get("Grade", ""),
    )

    for idx, rec in enumerate(recommendations[:3], start=1):
        rec_id = f"role:recommended:{idx}"
        add_node(
            rec_id,
            rec["job_title"],
            "recommended-role",
            sector=rec["sector"],
            match_pct=round(rec["composite"] * 100),
        )
        graph_edges.append({
            "source": "role:target",
            "target": rec_id,
            "type": "role-match",
            "strength": round(rec["composite"], 4),
        })

    top_graph_skills = []
    seen_graph_skills = set()
    for skill_name, weight in sorted(benchmark.items(), key=lambda item: item[1], reverse=True):
        key = skill_name.lower()
        if key in seen_graph_skills:
            continue
        seen_graph_skills.add(key)
        top_graph_skills.append((skill_name, weight))
        if len(top_graph_skills) >= 8:
            break

    highlighted_gaps = {gap["skill"].lower() for gap in prioritized_gaps[:6]}
    for skill_name, weight in top_graph_skills:
        skill_key = skill_name.lower()
        candidate_info = candidate_lookup.get(skill_key)
        is_gap = skill_key in highlighted_gaps or skill_key not in candidate_lookup
        skill_node_id = f"skill:{skill_key}"
        add_node(
            skill_node_id,
            skill_name.title(),
            "gap-skill" if is_gap else "candidate-skill",
            weight=round(float(weight), 2),
            proficiency=(candidate_info or {}).get("proficiency_level", 0),
            confidence=(candidate_info or {}).get("confidence", 0.0),
        )
        graph_edges.append({
            "source": "role:target",
            "target": skill_node_id,
            "type": "requires-skill",
            "strength": round(float(weight) / 5.0, 4),
        })

        for prereq in get_prerequisites(skill_graph, skill_key)[:3]:
            prereq_node_id = f"skill:{prereq}"
            if not any(node["id"] == prereq_node_id for node in graph_nodes):
                add_node(
                    prereq_node_id,
                    prereq.title(),
                    "foundation-skill",
                    weight=round(float(benchmark_lookup.get(prereq, 2.5)), 2),
                    proficiency=(candidate_lookup.get(prereq) or {}).get("proficiency_level", 0),
                    confidence=(candidate_lookup.get(prereq) or {}).get("confidence", 0.0),
                )
            graph_edges.append({
                "source": prereq_node_id,
                "target": skill_node_id,
                "type": "prerequisite",
                "strength": 0.7,
            })

    return {
        "candidate_profile": {
            "name": profile.get("name") or f"Candidate {metrics['ID']}",
            "email": profile.get("email"),
            "phone": profile.get("phone"),
            "linkedin": profile.get("linkedin"),
            "education": profile.get("education", []),
            "experience": profile.get("experience", []),
            "skills": candidate_skills,
        },
        "summary": {
            "target_role": metrics.get("Job_Title", ""),
            "coverage_pct": round(metrics.get("Composite_Score", 0.0) * 100),
            "gaps": len(metrics.get("Gaps", [])),
            "weeks": estimated_weeks,
            "readiness_label": _readiness_label(metrics.get("Grade", "")),
            "readiness_grade": metrics.get("Grade", ""),
            "technical_fit_pct": round(split.get("hard_fit", 0.0) * 100),
            "soft_fit_pct": round(split.get("soft_fit", 0.0) * 100),
            "technical_weight_pct": round(split.get("hard_weight", 0.0) * 100),
            "soft_weight_pct": round(split.get("soft_weight", 0.0) * 100),
            "estimated_hours": estimated_weeks * 6,
        },
        "target_role": {
            "title": metrics.get("Job_Title", ""),
            "required_skills": required_skills,
            "sector": get_sector(metrics.get("Job_Title", "")) or "Professional",
        },
        "gap_analysis": {
            "prioritized_gaps": prioritized_gaps,
            "reasoning": f"{len(metrics.get('Gaps', []))} skill gaps identified and prioritized by dependency level, importance, and confidence.",
            "gap_summary": {
                "coverage_pct": round(metrics.get("Composite_Score", 0.0) * 100),
                "total_gaps": len(metrics.get("Gaps", [])),
            },
        },
        "recommendations": recommended_roles,
        "learning_pathway": {
            "estimated_weeks": estimated_weeks,
            "roadmap": roadmap,
            "reasoning_summary": metrics.get("Pathway_Depth", ""),
            "estimated_total_hours": estimated_weeks * 6,
        },
        "visual_breakdown": {
            "readiness": {
                "grade": metrics.get("Grade", ""),
                "label": _readiness_label(metrics.get("Grade", "")),
                "coverage_pct": round(metrics.get("Composite_Score", 0.0) * 100),
                "technical_fit_pct": round(split.get("hard_fit", 0.0) * 100),
                "soft_fit_pct": round(split.get("soft_fit", 0.0) * 100),
            },
            "recommendations": recommended_roles,
            "top_strengths": [skill["skill_name"] for skill in candidate_skills[:8]],
            "top_gaps": [gap["skill"] for gap in prioritized_gaps[:8]],
        },
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
    }


def _run_analysis_for_api(
    resume_path: Path,
    job_description_path: Path | None = None,
    category: str | None = None,
    job_title: str | None = None,
) -> dict:
    profile = _parse_resume_upload(resume_path)
    row_dict = resume_profile_to_dataframe_row(profile)
    resume_text = profile.get("full_text", "")
    if not resume_text.strip():
        raise RuntimeError("The uploaded resume could not be read.")

    llm_output = None
    extracted_skills = []
    try:
        taxonomy_skills = load_taxonomy_skills()
        llm_output = extract_all_skills(profile, taxonomy_skills=taxonomy_skills or None)
        extracted_skills = [s["skill"] for s in llm_output.get("skills", [])]
    except Exception:
        extracted_skills = extract_skills(resume_text, FULL_SKILL_LIBRARY)

    row_dict["Extracted_Skills"] = extracted_skills
    row_dict["Category"] = category or job_title or "UPLOADED_PDF"
    cand_series = pd.Series(row_dict)

    resume_df = pd.DataFrame([{
        "ID": row_dict["ID"],
        "Category": row_dict["Category"],
        "Resume_str": resume_text,
        "Extracted_Skills": extracted_skills, 
    }])

    vec, svecs = _prepare_vectorizer(resume_df)
    recommendations = recommend_roles(
        resume_text,
        vec,
        top_n=3,
        min_match_count=1,
        candidate_id=row_dict["ID"],
    )

    resolved_job_title = job_title
    if not resolved_job_title and category:
        resolved_job_title = find_job_title(category) or category
    if not resolved_job_title:
        resolved_job_title = recommendations[0]["job_title"] if recommendations else "Software Engineer"

    job_text = ""
    jd_skills = None
    if job_description_path is not None:
        job_text = _extract_text_from_upload(job_description_path)
        if job_text.strip():
            jd_skills = extract_skills(job_text, FULL_SKILL_LIBRARY)

    vec, svecs = _prepare_vectorizer(
        resume_df,
        job_text,
        [build_weighted_benchmark_text(resolved_job_title)],
    )
    graph = build_skill_dependency_graph()

    metrics = evaluate_candidate(
        cand_series,
        vec,
        svecs,
        category=resolved_job_title,
        jd_skills=jd_skills,
        llm_json=llm_output,
    )
    cand_vec = metrics.pop("_cand_vec")
    metrics["Prioritized_Gaps"] = prioritize_gaps(
        metrics["Gaps"],
        metrics["Gap_Weights"],
        graph,
        cand_vec,
        svecs,
        confidence_scores=metrics.get("Confidence_Scores"),
    )
    roadmap_df = build_roadmap(metrics)
    metrics["Duration"] = len(roadmap_df)

    benchmark = metrics.get("Gap_Weights", {}).copy()
    for skill, info in metrics.get("Confidence_Scores", {}).items():
        benchmark.setdefault(skill, info.get("onet_weight", 0.0))

    return _build_api_payload(
        metrics,
        profile,
        roadmap_df,
        recommendations,
        benchmark,
        llm_output,
    )


# ── Core display functions ────────────────────────────────────────────────────

def _print_candidate_profile(profile: dict | None, cand_id, category: str, job_title: str) -> None:
    _section("CANDIDATE PROFILE")
    if profile:
        print(f"  Name          :  {profile.get('name', 'Not found')}")
        print(f"  Email         :  {profile.get('email', 'Not found')}")
        print(f"  Phone         :  {profile.get('phone', 'Not found')}")
        if profile.get("linkedin"):
            print(f"  LinkedIn      :  {profile['linkedin']}")
        if profile.get("education"):
            print(f"  Education     :  {profile['education'][0]}")
        if profile.get("experience"):
            print(f"  Experience    :  {profile['experience'][0]}")
    else:
        print(f"  Candidate ID  :  {cand_id}")

    _section("JOB IDENTIFICATION")
    sector = ""
    from taxonomy_adapter import get_sector
    sector = get_sector(job_title) or "Professional"
    cat_type = get_category_type(category, job_title)
    cat_label = {
        "TECH": "Technical / Technology",
        "MANAGEMENT": "Management & Business",
        "HEALTHCARE": "Healthcare & Clinical",
        "EDUCATION": "Education & Training",
        "CREATIVE": "Arts & Creative",
        "NON_TECH": "Professional Services",
    }.get(cat_type, cat_type)
    
    print(f"  Target Role   :  {job_title}")
    print(f"  Industry      :  {sector}")
    print(f"  Category      :  {cat_label}")


def _print_readiness(metrics: dict) -> None:
    _section("OVERALL READINESS")
    grade   = metrics["Grade"]
    score   = metrics["Composite_Score"]
    pct     = round(score * 100)
    label   = _readiness_label(grade)
    bar     = _progress_bar(score, width=30)
    pathway = metrics["Pathway_Depth"]
    weeks   = metrics["Duration"]

    print(f"  Readiness     :  {pct}%  [{bar}]")
    print(f"  Status        :  {label}")
    print(f"  Learning Plan :  {pathway}")
    print(f"  Duration      :  {weeks} Weeks")

    # Hard vs soft breakdown in plain language
    split = metrics.get("Split_Details", {})
    if split:
        hard_pct = round(split.get("hard_fit", 0) * 100)
        soft_pct = round(split.get("soft_fit", 0) * 100)
        h_weight = round(split.get("hard_weight", 0) * 100)
        s_weight = round(split.get("soft_weight", 0) * 100)
        print()
        print(f"  Technical Skills Score  :  {hard_pct}%  (counts for {h_weight}% of your score)")
        print(f"  Soft Skills Score       :  {soft_pct}%  (counts for {s_weight}% of your score)")


def _print_skills(metrics: dict, job_title: str) -> None:
    _section("SKILLS IDENTIFIED IN YOUR CV")

    from skill_classifier import classify_skill
    extracted    = sorted(metrics.get("Extracted_Skills", []))
    tech_skills = sorted([s for s in extracted if classify_skill(s, job_title) == "hard"])
    soft_skills = sorted([s for s in extracted if classify_skill(s, job_title) == "soft"])
    conf_scores  = metrics.get("Confidence_Scores", {})
    hard_bench, soft_bench = split_benchmark(
        metrics.get("Gap_Weights", {}) | {
            s: conf_scores.get(s, {}).get("onet_weight", 3.0) for s in extracted
        },
        job_title,
    )

    print(f"  Total skills found  :  {len(extracted)}")
    print()
    if hard_bench:
        # Wrap into lines of ~55 chars
        line, lines = "", []
        for s in tech_skills:
            chunk = (", " if line else "") + s.title()
            if len(line) + len(chunk) > 55:
                lines.append(line)
                line = s.title()
            else:
                line += chunk
        if line: lines.append(line)
        if lines:
            print(f"  Technical Skills  :  {lines[0]}")
            for l in lines[1:]:
                print(f"                     {l}")

    print()
    if soft_bench:
        line, lines = "", []
        for s in soft_skills:
            chunk = (", " if line else "") + s.title()
            if len(line) + len(chunk) > 55:
                lines.append(line)
                line = s.title()
            else:
                line += chunk
        if line: lines.append(line)
        if lines:
            print(f"  Soft / People     :  {lines[0]}")
            for l in lines[1:]:
                print(f"                     {l}")


def _print_skill_gaps(metrics: dict) -> None:
    prio_gaps = metrics.get("Prioritized_Gaps", [])
    if not prio_gaps:
        return

    _section(f"AREAS TO DEVELOP  ({len(prio_gaps)} identified)")
    print(f"  {'#':<4} {'Skill':<35} {'Type':<12} {'Importance'}")
    _line()

    from skill_classifier import classify_skill
    job_title = metrics.get("Job_Title", "")
    for g in prio_gaps:
        skill     = g["Skill"].title()
        weight    = g["ONET_Weight"]
        imp       = _importance_label(weight)
        skill_type= "Technical" if classify_skill(g["Skill"], job_title) == "hard" else "Soft Skill"
        print(f"  {g['Priority']:<4} {skill:<35} {skill_type:<12} {imp}")


def _print_recommendations(recommendations: list[dict]) -> None:
    if not recommendations:
        return

    top3 = recommendations[:3]
    _section("TOP 3 ROLES THAT MATCH YOUR PROFILE")

    for i, r in enumerate(top3, 1):
        pct   = round(r["composite"] * 100)
        bar   = _progress_bar(r["composite"], width=20)
        title = r["job_title"]
        sec   = r["sector"]
        matched = [s.title() for s in r["matched_skills"][:5]]
        gaps    = [s.title() for s in r["gap_skills"][:3]]

        print(f"\n  #{i}  {title}  [{sec}]")
        print(f"       Match  :  {pct}%  {bar}")
        print(f"       You have  :  {', '.join(matched) or 'building foundations'}")
        print(f"       To learn  :  {', '.join(gaps) or 'minor gaps only'}")

    print()


def _print_learning_pathway(metrics: dict, roadmap_df: pd.DataFrame) -> None:
    _section(f"YOUR PERSONALISED {metrics['Duration']}-WEEK LEARNING PATHWAY")

    print(f"  This plan focuses on your most important skill gaps,")
    print(f"  starting with foundations and building towards advanced topics.")
    print()
    print(f"  {'Week':<8} {'What to Learn':<32} {'Goal'}")
    _line()

    for _, row in roadmap_df.iterrows():
        skill   = row["Skill"].title()
        obj     = row["Objective"]
        success = row["Success"]
        # Truncate long objectives for clean display
        if len(obj) > 38:
            obj = obj[:36] + ".."
        print(f"  {row['Week']:<8} {skill:<32} {obj}")
        print(f"  {'':<8} {'':32} - {success}")
        print()


def _print_footer(paths: dict) -> None:
    _section("REPORTS SAVED")
    for label, path in [("Full Report (text)", paths.get("text")),
                         ("Data Export (JSON)", paths.get("json")),
                         ("Visual Dashboard",   paths.get("dashboard"))]:
        if path:
            print(f"  {label:<22}:  {path}")
    print()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(
    cand_series,
    vectorizer, skill_vectors, graph,
    profile=None,
    category=None,
    save_dashboard=True,
    label="",
    jd_skills=None,
    llm_json=None,        # ← NEW: LLM extraction output, None in CSV/batch mode
) -> dict:
 
    # ── Evaluate (job matching + gap analysis) ────────────────────────────────
    metrics  = evaluate_candidate(
        cand_series, vectorizer, skill_vectors,
        category=category, jd_skills=jd_skills,
        llm_json=llm_json,    # ← passed through to evaluator
    )
    cand_vec = metrics.pop("_cand_vec")
 
    # ── Prioritise gaps ───────────────────────────────────────────────────────
    prio_gaps = prioritize_gaps(
        metrics["Gaps"], metrics["Gap_Weights"], graph, cand_vec, skill_vectors,
        confidence_scores=metrics.get("Confidence_Scores"),
    )
    metrics["Prioritized_Gaps"] = prio_gaps
 
    # ── Role recommendations ──────────────────────────────────────────────────
    resume_text     = str(cand_series.get("Resume_str", ""))
    recommendations = recommend_roles(
        resume_text,
        vectorizer,
        top_n=3,
        min_match_count=1,
        candidate_id=metrics["ID"],
    )
 
    # ── Roadmap ───────────────────────────────────────────────────────────────
    roadmap_df = build_roadmap(metrics)
    metrics["Duration"] = len(roadmap_df)   # keep Duration honest after roadmap fix
 
    # ── Print ─────────────────────────────────────────────────────────────────
    cand_id   = metrics["ID"]
    job_title = metrics["Job_Title"]
    raw_cat   = category or str(cand_series.get("Category", ""))
 
    print("\n")
    print("=" * 62)
    print("  CANDIDATE ONBOARDING REPORT")
    print("=" * 62)
 
    _print_candidate_profile(profile, cand_id, raw_cat, job_title)
    _print_readiness(metrics)
    _print_skills(metrics, job_title)
    _print_skill_gaps(metrics)
    _print_recommendations(recommendations)
    _print_learning_pathway(metrics, roadmap_df)
 
    # ── Save ──────────────────────────────────────────────────────────────────
    from pathlib import Path
    from config import OUTPUT_DIR
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_png = Path(OUTPUT_DIR) / f"dashboard_{cand_id}.png" if save_dashboard else None
 
    generate_dashboard(
        metrics, graph, recommendations,
        profile=profile,
        save_path=out_png, show=False,
    )
 
    paths = export_all(
        metrics, roadmap_df,
        confidence_scores=metrics.get("Confidence_Scores"),
        recommendations=recommendations,
        output_dir=OUTPUT_DIR,
    )
    if out_png:
        paths["dashboard"] = out_png
 
    _print_footer(paths)
    return metrics

def _llm_result_to_series(
    llm_output: dict,
    resume_text: str,
    profile: dict,
    job_title: str,
    row_dict: dict,
) -> pd.Series:
    """
    Bridge LLM extraction output → pd.Series expected by evaluate_candidate.
    Merges the contact/ID fields already built by resume_profile_to_dataframe_row
    with the skill list from the LLM.
    """
    row_dict["Category"]         = job_title
    row_dict["Extracted_Skills"] = [s["skill"] for s in llm_output["skills"]]
    return pd.Series(row_dict)

# ── Single PDF ────────────────────────────────────────────────────────────────

def run_single_pdf(
    resume_pdf_path, job_title,
    job_pdf_path, save=True,
) -> None:
    # Step 1: parse PDF → structured dict
    profile  = parse_resume_pdf(resume_pdf_path)
    row_dict = resume_profile_to_dataframe_row(profile)
 
    # Step 2: LLM extraction → JSON
    print(f"\n  [pipeline] Running LLM extraction via Ollama...")
    llm_output = extract_all_skills(profile)
 
    # Step 3: bridge LLM JSON → evaluator-compatible Series
    cand_series = _llm_result_to_series(
        llm_output,
        profile["full_text"],
        profile,
        job_title,
        row_dict,
    )
 
    # Step 4: build TF-IDF vectorizer (still needed for cosine similarity)
    pdf_df   = pd.DataFrame([{
        "Resume_str": profile["full_text"],
        "ID":         row_dict["ID"],
        "Category":   job_title,
    }])
    job_text  = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    jd_skills = extract_skills(job_text, FULL_SKILL_LIBRARY) if job_text else None
    vec, svecs = _prepare_vectorizer(
        pdf_df,
        job_text,
        [build_weighted_benchmark_text(job_title)],
    )
    graph = build_skill_dependency_graph()
 
    # Steps 5–6: job matching → gap analysis → roadmap
    # llm_json passed so evaluator uses LLM confidence, not TF-IDF vectors
    _run_pipeline(
        cand_series, vec, svecs, graph,
        profile=profile, category=job_title,
        save_dashboard=save, label=profile["name"],
        jd_skills=jd_skills,
        llm_json=llm_output,    # ← the key change
    )


# ── Single CSV ────────────────────────────────────────────────────────────────

def run_single_csv(
    csv_path: str, candidate_id: int,
    job_pdf_path: str, save: bool = True,
) -> None:
    df       = pd.read_csv(csv_path)
    df["Extracted_Skills"] = df["Resume_str"].apply(lambda t: extract_skills(t, FULL_SKILL_LIBRARY))

    cand_row = df[df["ID"] == candidate_id]
    if cand_row.empty:
        print(f"Candidate {candidate_id} not found."); sys.exit(1)

    row      = cand_row.iloc[0]
    category = str(row.get("Category", ""))
    resolved = find_job_title(category) or category
    job_text = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    jd_skills = extract_skills(job_text, FULL_SKILL_LIBRARY) if job_text else None

    extra    = [build_weighted_benchmark_text(resolved)]
    vec, svecs = _prepare_vectorizer(df, job_text, extra)
    graph    = build_skill_dependency_graph()

    _run_pipeline(
        row, vec, svecs, graph,
        category=category, save_dashboard=save, label=candidate_id,
        jd_skills=jd_skills,
    )


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(csv_path: str, job_pdf_path: str) -> None:
    df  = pd.read_csv(csv_path)
    df["Extracted_Skills"] = df["Resume_str"].apply(lambda t: extract_skills(t, FULL_SKILL_LIBRARY))
    
    job_text = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    jd_skills = extract_skills(job_text, FULL_SKILL_LIBRARY) if job_text else None
    
    vec, svecs = _prepare_vectorizer(df, job_text)

    all_metrics = batch_evaluate(df, vec, svecs, jd_skills=jd_skills)
    rows = [{
        "ID":       m.get("ID"),
        "Name":     m.get("ID"),
        "Role":     m.get("Job_Title",""),
        "Grade":    m.get("Grade",""),
        "Readiness":f"{round(m.get('Composite_Score',0)*100)}%",
        "Pathway":  m.get("Pathway_Depth",""),
        "Weeks":    m.get("Duration",""),
        "Gaps":     len(m.get("Gaps",[])),
    } for m in all_metrics if "error" not in m]

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / "batch_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nResults saved to {out}")


def run_recommend_only(resume_pdf_path: str, save: bool = True) -> None:
    profile = parse_resume_pdf(resume_pdf_path)
    resume_text = profile.get("full_text", "")
    row_dict = resume_profile_to_dataframe_row(profile)
    pdf_df = pd.DataFrame([row_dict])
    vec, _ = _prepare_vectorizer(pdf_df)

    recommendations = recommend_roles(
        resume_text,
        vec,
        top_n=5,
        min_match_count=1,
        candidate_id=row_dict["ID"],
    )
    _print_recommendations(recommendations)


app = FastAPI(title="SkillDeck Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _persist_upload(upload: UploadFile | None) -> Path | None:
    if upload is None:
        return None

    suffix = Path(upload.filename or "").suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        return Path(tmp.name)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: UploadFile | None = File(default=None),
    category: str | None = Form(default=None),
    job_title: str | None = Form(default=None),
):
    resume_path = await _persist_upload(resume)
    jd_path = await _persist_upload(job_description)
    try:
        return JSONResponse(
            _run_analysis_for_api(
                resume_path=resume_path,
                job_description_path=jd_path,
                category=category,
                job_title=job_title,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for path in [resume_path, jd_path]:
            if path and path.exists():
                path.unlink(missing_ok=True)


@app.post("/analyze/visual")
async def analyze_resume_visual(
    resume: UploadFile = File(...),
    job_description: UploadFile | None = File(default=None),
    category: str | None = Form(default=None),
    job_title: str | None = Form(default=None),
):
    resume_path = await _persist_upload(resume)
    jd_path = await _persist_upload(job_description)
    try:
        payload = _run_analysis_for_api(
            resume_path=resume_path,
            job_description_path=jd_path,
            category=category,
            job_title=job_title,
        )
        return {
            "visual_breakdown": payload.get("visual_breakdown", {}),
            "graph": payload.get("graph", {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for path in [resume_path, jd_path]:
            if path and path.exists():
                path.unlink(missing_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Candidate Onboarding Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv",        type=str, default=RESUME_CSV_PATH)
    src.add_argument("--resume_pdf", type=str, help="Path to candidate PDF resume")

    p.add_argument("--id",           type=int,  help="Candidate ID (CSV mode)")
    p.add_argument("--job_title",    type=str,  default=None,
                   help="Target job title e.g. 'Machine Learning', 'Accountants and Auditors'")
    p.add_argument("--job",          type=str,  default=JOB_DESC_PDF_PATH)
    p.add_argument("--batch",        action="store_true", help="Evaluate all candidates in CSV")
    p.add_argument("--recommend",    action="store_true", help="Only show role recommendations for a PDF resume")
    p.add_argument("--no_save",      action="store_true", help="Don't save dashboard image")
    p.add_argument("--list_sectors", action="store_true", help="Show all available sectors")
    p.add_argument("--list_jobs",    action="store_true", help="Show available job titles")
    p.add_argument("--sector",       type=str,  default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    save = not args.no_save

    if args.list_sectors:
        print("\nAvailable sectors:")
        for s in list_sectors(): print(f"  {s}")
        return

    if args.list_jobs:
        print(f"\nJob titles{' in ' + args.sector if args.sector else ''}:")
        for j in list_jobs(args.sector): print(f"  {j}")
        return

    if args.resume_pdf:
        job_title = args.job_title or "Machine Learning"
        if args.recommend:
            run_recommend_only(args.resume_pdf, save = save)
        else:
            run_single_pdf(args.resume_pdf, job_title, args.job, save=save)

    elif args.batch:
        run_batch(args.csv, args.job)

    else:
        if args.id is None:
            build_parser().error("--id is required when using --csv mode.")
        run_single_csv(args.csv, args.id, args.job, save=save)


if __name__ == "__main__":
    main()
