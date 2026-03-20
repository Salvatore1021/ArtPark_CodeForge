"""
main.py
-------
AI-Driven Onboarding Engine — entry point.

Usage
-----
# Evaluate a candidate from the Resume CSV by ID:
    python main.py --csv data/Resume.csv --id 16237710 --job data/job_desc.pdf

# Evaluate a candidate from a PDF resume directly:
    python main.py --resume_pdf path/to/candidate_resume.pdf --category ACCOUNTANT --job data/job_desc.pdf

# Batch evaluate all candidates in the CSV:
    python main.py --csv data/Resume.csv --batch --job data/job_desc.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Internal modules ──────────────────────────────────────────────────────────
from config import RESUME_CSV_PATH, JOB_DESC_PDF_PATH, OUTPUT_DIR
from pdf_parser import extract_text_from_pdf, parse_resume_pdf, resume_profile_to_dataframe_row
from skill_extractor import (
    build_skill_library,
    FULL_SKILL_LIBRARY,
    extract_skills,
    build_and_fit_vectorizer,
    precompute_skill_vectors,
)
from dependency_graph import build_skill_dependency_graph, graph_summary
from candidate_evaluator import evaluate_candidate, batch_evaluate
from gap_prioritizer import prioritize_gaps
from roadmap_builder import build_roadmap, roadmap_to_text
from dashboard import generate_dashboard


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _load_resume_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[CSV] Loaded {len(df)} resumes from {csv_path}")
    return df


def _apply_skill_extraction(df: pd.DataFrame, skill_library: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["Extracted_Skills"] = df["Resume_str"].apply(
        lambda t: extract_skills(t, skill_library)
    )
    return df


# ── Single candidate (from CSV) ───────────────────────────────────────────────

def run_single_csv(
    csv_path: str,
    candidate_id: int,
    job_pdf_path: str,
    save_dashboard: bool = True,
) -> None:
    _print_section("Step 1 — Initialise Environment")

    # Load data
    resume_df       = _load_resume_csv(csv_path)
    job_text        = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    skill_library   = FULL_SKILL_LIBRARY

    # Extract skills
    resume_df = _apply_skill_extraction(resume_df, skill_library)

    # Fit vectorizer
    vectorizer      = build_and_fit_vectorizer(resume_df, job_text, skill_library)
    skill_vectors   = precompute_skill_vectors(vectorizer, skill_library)
    print(f"[Vectorizer] Vocabulary: {len(vectorizer.get_feature_names_out())} features")

    _print_section("Step 2 — Build Skill Dependency Graph")
    graph   = build_skill_dependency_graph()
    summary = graph_summary(graph)
    print(f"[Graph] Nodes: {summary['nodes']} | Edges: {summary['edges']} | DAG: {summary['is_dag']}")

    _print_section("Step 3 — Evaluate Candidate")
    cand_row = resume_df[resume_df["ID"] == candidate_id]
    if cand_row.empty:
        print(f"[ERROR] Candidate {candidate_id} not found in {csv_path}")
        sys.exit(1)

    metrics = evaluate_candidate(cand_row.iloc[0], vectorizer)
    print(f"Grade: {metrics['Grade']} | Composite: {metrics['Composite_Score']:.4f}")
    print(f"Pathway: {metrics['Pathway_Depth']} ({metrics['Duration']} weeks)")
    print(f"Gaps: {metrics['Gaps']}")

    _print_section("Step 4 — Prioritise Skill Gaps")
    cand_vec            = metrics.pop("_cand_vec")   # extract sparse vector
    prioritized_gaps    = prioritize_gaps(metrics["Gaps"], cand_vec, skill_vectors, graph)
    metrics["Prioritized_Gaps"] = prioritized_gaps

    for g in prioritized_gaps:
        print(f"  Priority {g['Priority']:>2}: {g['Skill']:<30} Level={g['Level']}  Sim={g['Similarity']:.4f}")

    _print_section("Step 5 — Build Roadmap")
    roadmap_df = build_roadmap(metrics)
    print(roadmap_to_text(roadmap_df, candidate_id))

    _print_section("Step 6 — Generate Dashboard")
    out_path = Path(OUTPUT_DIR) / f"dashboard_{candidate_id}.png" if save_dashboard else None
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    generate_dashboard(metrics, graph, save_path=out_path, show=True)


# ── Single candidate (from PDF resume) ────────────────────────────────────────

def run_single_pdf(
    resume_pdf_path: str,
    category: str,
    csv_path: str | None,
    job_pdf_path: str,
    save_dashboard: bool = True,
) -> None:
    _print_section("Step 1 — Parse PDF Resume")

    profile = parse_resume_pdf(resume_pdf_path)
    print(f"[PDF] Candidate Name  : {profile['name']}")
    print(f"[PDF] Email           : {profile['email']}")
    print(f"[PDF] Phone           : {profile['phone']}")
    print(f"[PDF] LinkedIn        : {profile['linkedin']}")
    if profile["education"]:
        print(f"[PDF] Education       : {profile['education'][0]}")
    if profile["experience"]:
        print(f"[PDF] Experience (1st): {profile['experience'][0]}")

    # Build a minimal DataFrame row for this candidate
    row_dict                 = resume_profile_to_dataframe_row(profile)
    row_dict["Category"]     = category.upper()
    skill_library            = FULL_SKILL_LIBRARY
    row_dict["Extracted_Skills"] = extract_skills(profile["full_text"], skill_library)
    cand_series              = pd.Series(row_dict)

    # Load CSV corpus for better vectorizer coverage (optional)
    corpus_df = pd.DataFrame()
    if csv_path and Path(csv_path).exists():
        corpus_df = _load_resume_csv(csv_path)
        corpus_df = _apply_skill_extraction(corpus_df, skill_library)

    # Combine PDF candidate with CSV corpus for vectoriser fitting
    pdf_df    = pd.DataFrame([row_dict])
    all_df    = pd.concat([corpus_df, pdf_df], ignore_index=True) if not corpus_df.empty else pdf_df
    all_df["Resume_str"] = all_df["Resume_str"].fillna(all_df.get("full_text", ""))

    job_text     = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    vectorizer   = build_and_fit_vectorizer(all_df, job_text, skill_library)
    skill_vectors= precompute_skill_vectors(vectorizer, skill_library)
    print(f"[Vectorizer] Vocabulary: {len(vectorizer.get_feature_names_out())} features")

    _print_section("Step 2 — Build Skill Dependency Graph")
    graph   = build_skill_dependency_graph()
    summary = graph_summary(graph)
    print(f"[Graph] Nodes: {summary['nodes']} | Edges: {summary['edges']} | DAG: {summary['is_dag']}")

    _print_section("Step 3 — Evaluate Candidate")
    metrics = evaluate_candidate(cand_series, vectorizer, category=category)
    print(f"Grade: {metrics['Grade']} | Composite: {metrics['Composite_Score']:.4f}")
    print(f"Pathway: {metrics['Pathway_Depth']} ({metrics['Duration']} weeks)")
    print(f"Gaps: {metrics['Gaps']}")

    _print_section("Step 4 — Prioritise Skill Gaps")
    cand_vec         = metrics.pop("_cand_vec")
    prioritized_gaps = prioritize_gaps(metrics["Gaps"], cand_vec, skill_vectors, graph)
    metrics["Prioritized_Gaps"] = prioritized_gaps

    for g in prioritized_gaps:
        print(f"  Priority {g['Priority']:>2}: {g['Skill']:<30} Level={g['Level']}  Sim={g['Similarity']:.4f}")

    _print_section("Step 5 — Build Roadmap")
    roadmap_df = build_roadmap(metrics)
    cand_name  = profile["name"]
    print(roadmap_to_text(roadmap_df, f"{metrics['ID']} ({cand_name})"))

    _print_section("Step 6 — Generate Dashboard")
    out_path = (
        Path(OUTPUT_DIR) / f"dashboard_{metrics['ID']}.png"
        if save_dashboard else None
    )
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    generate_dashboard(metrics, graph, save_path=out_path, show=True)


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(csv_path: str, job_pdf_path: str) -> None:
    _print_section("Batch Evaluation Mode")

    resume_df     = _load_resume_csv(csv_path)
    job_text      = extract_text_from_pdf(job_pdf_path) if Path(job_pdf_path).exists() else ""
    skill_library = FULL_SKILL_LIBRARY

    resume_df     = _apply_skill_extraction(resume_df, skill_library)
    vectorizer    = build_and_fit_vectorizer(resume_df, job_text, skill_library)

    all_metrics   = batch_evaluate(resume_df, vectorizer)
    summary_rows  = [
        {k: v for k, v in m.items()
         if k not in ("Gaps", "Extracted_Skills", "_cand_vec", "Prioritized_Gaps")}
        for m in all_metrics
    ]
    summary_df    = pd.DataFrame(summary_rows)

    out_csv = Path(OUTPUT_DIR) / "batch_results.csv"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"\n[Batch] Results saved to {out_csv}")


# ── CLI argument parser ───────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI-Driven Candidate Onboarding Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = p.add_mutually_exclusive_group()
    source.add_argument("--csv",         type=str, default=RESUME_CSV_PATH,
                        help="Path to Resume.csv")
    source.add_argument("--resume_pdf",  type=str,
                        help="Path to a candidate PDF resume file (triggers PDF mode)")

    p.add_argument("--id",       type=int,   default=None,
                   help="Candidate ID to evaluate (CSV mode)")
    p.add_argument("--category", type=str,   default="ACCOUNTANT",
                   help="Role category (PDF mode, e.g. ACCOUNTANT, DATA SCIENCE)")
    p.add_argument("--job",      type=str,   default=JOB_DESC_PDF_PATH,
                   help="Path to job-description PDF")
    p.add_argument("--batch",    action="store_true",
                   help="Batch-evaluate all candidates in the CSV")
    p.add_argument("--no_save",  action="store_true",
                   help="Do not save the dashboard image")

    return p


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    save   = not args.no_save

    if args.resume_pdf:
        # ── PDF resume mode ──────────────────────────────────────────────────
        run_single_pdf(
            resume_pdf_path=args.resume_pdf,
            category=args.category,
            csv_path=args.csv if args.csv != RESUME_CSV_PATH else None,
            job_pdf_path=args.job,
            save_dashboard=save,
        )

    elif args.batch:
        # ── Batch CSV mode ───────────────────────────────────────────────────
        run_batch(csv_path=args.csv, job_pdf_path=args.job)

    else:
        # ── Single CSV candidate mode ────────────────────────────────────────
        if args.id is None:
            parser.error("--id is required in single-candidate CSV mode.")
        run_single_csv(
            csv_path=args.csv,
            candidate_id=args.id,
            job_pdf_path=args.job,
            save_dashboard=save,
        )


if __name__ == "__main__":
    main()
