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
from pathlib import Path

import pandas as pd

from config import RESUME_CSV_PATH, JOB_DESC_PDF_PATH, OUTPUT_DIR
from taxonomy_adapter import list_sectors, list_jobs, find_job_title, build_weighted_benchmark_text
from resume_parser import extract_text_from_pdf, parse_resume_pdf, resume_profile_to_dataframe_row
from skill_extractor import FULL_SKILL_LIBRARY, extract_skills, build_and_fit_vectorizer, precompute_skill_vectors
from dependency_graph import build_skill_dependency_graph
from candidate_evaluator import evaluate_candidate, batch_evaluate
from gap_prioritizer import prioritize_gaps
from roadmap_builder import build_roadmap
from dashboard import generate_dashboard
from role_recommender import recommend_roles
from report_exporter import export_all
from skill_classifier import split_benchmark, get_category_type



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
    return "█" * filled + "░" * (width - filled)


# ── Pretty print helpers ──────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def _line() -> None:
    print(f"  {'─'*58}")


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

    extracted    = sorted(metrics.get("Extracted_Skills", []))
    conf_scores  = metrics.get("Confidence_Scores", {})
    hard_bench, soft_bench = split_benchmark(
        metrics.get("Gap_Weights", {}) | {
            s: conf_scores.get(s, {}).get("onet_weight", 3.0) for s in extracted
        },
        job_title,
    )

    hard_found = [s for s in extracted if s in hard_bench or
                  s not in {g.lower() for g in metrics.get("Gaps", [])}]
    soft_found = [s for s in extracted if s.lower() not in hard_found]

    # Simpler: classify each extracted skill
    from skill_classifier import classify_skill
    tech_skills = sorted([s for s in extracted if classify_skill(s, job_title) == "hard"])
    soft_skills = sorted([s for s in extracted if classify_skill(s, job_title) == "soft"])

    print(f"  Total skills found  :  {len(extracted)}")
    print()
    if tech_skills:
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
        print(f"  Technical Skills  :  {lines[0]}")
        for l in lines[1:]:
            print(f"                     {l}")

    print()
    if soft_skills:
        line, lines = "", []
        for s in soft_skills:
            chunk = (", " if line else "") + s.title()
            if len(line) + len(chunk) > 55:
                lines.append(line)
                line = s.title()
            else:
                line += chunk
        if line: lines.append(line)
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
        print(f"  {'':<8} {'':32} ✓ {success}")
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
    cand_series: pd.Series,
    vectorizer, skill_vectors: dict, graph,
    profile: dict | None = None,
    category: str | None = None,
    save_dashboard: bool = True,
    label: str = "",
    jd_skills: set[str] | list[str] | None = None,
) -> dict:

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics  = evaluate_candidate(
        cand_series, vectorizer, skill_vectors, 
        category=category, jd_skills=jd_skills
    )
    cand_vec = metrics.pop("_cand_vec")

    # ── Prioritise gaps ───────────────────────────────────────────────────────
    prio_gaps = prioritize_gaps(
        metrics["Gaps"], metrics["Gap_Weights"], cand_vec, skill_vectors, graph,
        confidence_scores=metrics.get("Confidence_Scores"),
    )
    metrics["Prioritized_Gaps"] = prio_gaps

    # ── Role recommendations (always on, top 3) ───────────────────────────────
    resume_text    = str(cand_series.get("Resume_str", ""))
    recommendations = recommend_roles(resume_text, vectorizer, top_n=3, min_match_count=1)

    # ── Roadmap ───────────────────────────────────────────────────────────────
    roadmap_df = build_roadmap(metrics)

    # ── Print clean output ────────────────────────────────────────────────────
    cand_id   = metrics["ID"]
    job_title = metrics["Job_Title"]
    raw_cat   = category or str(cand_series.get("Category", ""))

    print("\n")
    print("█" * 62)
    print("  CANDIDATE ONBOARDING REPORT")
    print("█" * 62)

    _print_candidate_profile(profile, cand_id, raw_cat, job_title)
    _print_readiness(metrics)
    _print_skills(metrics, job_title)
    _print_skill_gaps(metrics)
    _print_recommendations(recommendations)
    _print_learning_pathway(metrics, roadmap_df)

    # ── Save dashboard & exports ──────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_png = Path(OUTPUT_DIR) / f"dashboard_{cand_id}.png" if save_dashboard else None

    generate_dashboard(
        metrics, graph, recommendations,
        profile=profile,
        save_path=out_png, show=True,
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


# ── Single PDF ────────────────────────────────────────────────────────────────

def run_single_pdf(
    resume_pdf_path: str, job_title: str,
    job_pdf_path: str, save: bool = True,
) -> None:
    profile  = parse_resume_pdf(resume_pdf_path)
    row_dict = resume_profile_to_dataframe_row(profile)
    row_dict["Category"]        = job_title
    row_dict["Extracted_Skills"]= extract_skills(profile["full_text"], FULL_SKILL_LIBRARY)
    cand_series = pd.Series(row_dict)

    pdf_df  = pd.DataFrame([{"Resume_str": profile["full_text"],
                              "ID": row_dict["ID"], "Category": job_title}])
    job_text = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    jd_skills = extract_skills(job_text, FULL_SKILL_LIBRARY) if job_text else None

    extra   = [build_weighted_benchmark_text(job_title)]
    vec     = build_and_fit_vectorizer(pdf_df, job_text, FULL_SKILL_LIBRARY,
                                        extra_benchmark_texts=extra)
    svecs   = precompute_skill_vectors(vec, FULL_SKILL_LIBRARY)
    graph   = build_skill_dependency_graph()

    _run_pipeline(
        cand_series, vec, svecs, graph,
        profile=profile, category=job_title,
        save_dashboard=save, label=profile["name"],
        jd_skills=jd_skills,
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
    vec      = build_and_fit_vectorizer(df, job_text, FULL_SKILL_LIBRARY,
                                         extra_benchmark_texts=extra)
    svecs    = precompute_skill_vectors(vec, FULL_SKILL_LIBRARY)
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
    
    vec, svecs = _quick_vectorizer(df, job_text)

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


def _quick_vectorizer(df, job_text):
    vec   = build_and_fit_vectorizer(df, job_text, FULL_SKILL_LIBRARY)
    svecs = precompute_skill_vectors(vec, FULL_SKILL_LIBRARY)
    return vec, svecs


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
        run_single_pdf(args.resume_pdf, job_title, args.job, save=save)

    elif args.batch:
        run_batch(args.csv, args.job)

    else:
        if args.id is None:
            build_parser().error("--id is required when using --csv mode.")
        run_single_csv(args.csv, args.id, args.job, save=save)


if __name__ == "__main__":
    main()
