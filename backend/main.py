"""
main.py — AI-Adaptive Onboarding Engine API
ARTPARK CodeForge Hackathon

Endpoints:
  GET  /health                  — Liveness check
  POST /analyze                 — Full analysis: resume + JD → pathway
  POST /extract/resume          — Extract skills from resume only
  POST /extract/jd              — Extract skills from JD only
  GET  /catalog                 — Return full course catalog
  GET  /catalog/{course_id}     — Return single course details
"""
import tempfile
from fastapi import Form
import pandas as pd

# Updated imports pointing to result_engine
from result_engine.pdf_parser import extract_text_from_pdf as vedant_extract, parse_resume_pdf, resume_profile_to_dataframe_row
from result_engine.skill_extractor import build_skill_library, extract_skills as vedant_extract_skills, build_and_fit_vectorizer, precompute_skill_vectors
from result_engine.dependency_graph import build_skill_dependency_graph
from result_engine.candidate_evaluator import evaluate_candidate
from result_engine.gap_prioritizer import prioritize_gaps
from result_engine.roadmap_builder import build_roadmap
from result_engine.dashboard import generate_dashboard

from gap_analyzer import compute_skill_gap
from metrics import evaluate_pathway
from pathway_engine import generate_learning_pathway, load_course_catalog

# Import the new NLP pipeline modules
from resume_parser import parse_resume_html, parse_resume_plaintext
from skill_extractor import extract_all_skills

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("aurora")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI-Adaptive Onboarding Engine...")
    # Pre-warm catalog cache
    catalog = load_course_catalog()
    logger.info(f"Catalog loaded: {len(catalog)} courses")
    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-Adaptive Onboarding Engine",
    description=(
        "Parse resume + job description → extract skill gaps → "
        "generate personalized, graph-ordered learning pathway."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start, 3)
    response.headers["X-Process-Time"] = str(elapsed)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed}s)")
    return response


# ---------------------------------------------------------------------------
# Allowed file types
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
MAX_FILE_SIZE_MB = 10


def _validate_file(upload: UploadFile) -> None:
    """Raise HTTPException if file type is disallowed."""
    name = upload.filename or ""
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "service": "AI-Adaptive Onboarding Engine",
        "version": "1.0.0",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/catalog", tags=["Catalog"])
def get_catalog():
    """Return the full course catalog used for pathway generation."""
    catalog = load_course_catalog()
    return {
        "total_courses": len(catalog),
        "courses": catalog,
    }


@app.get("/catalog/{course_id}", tags=["Catalog"])
def get_course(course_id: str):
    """Return details for a single course by ID."""
    catalog = load_course_catalog()
    for course in catalog:
        if course["id"] == course_id:
            return course
    raise HTTPException(status_code=404, detail=f"Course '{course_id}' not found")


@app.post("/extract/resume", tags=["Extraction"])
async def extract_resume(file: UploadFile = File(..., description="Resume PDF/DOCX/TXT")):
    """Extract structured skill profile from a resume document."""
    _validate_file(file)
    raw_bytes = await file.read()
    text = extract_text(raw_bytes, file.filename or "resume.pdf")
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the uploaded resume.")
    result = await extract_skills_from_resume(text)
    return {
        "filename": file.filename,
        "char_count": len(text),
        "extraction_method": result.get("extraction_method"),
        "profile": result,
    }


@app.post("/extract/jd", tags=["Extraction"])
async def extract_jd(file: UploadFile = File(..., description="Job Description PDF/DOCX/TXT")):
    """Extract required skills from a job description document."""
    _validate_file(file)
    raw_bytes = await file.read()
    text = extract_text(raw_bytes, file.filename or "jd.pdf")
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the uploaded JD.")
    result = await extract_skills_from_jd(text)
    return {
        "filename": file.filename,
        "char_count": len(text),
        "extraction_method": result.get("extraction_method"),
        "job_profile": result,
    }


@app.post("/analyze", tags=["Core"])
async def analyze(
    resume: UploadFile = File(..., description="Candidate resume (PDF/DOCX/TXT)"),
    job_description: UploadFile = File(..., description="Target job description (PDF/DOCX/TXT)"),
):
    """
    Full end-to-end analysis pipeline:
    1. Extract text from resume and JD
    2. Use LLM to extract structured skills from both
    3. Compute skill gap with priority scoring
    4. Generate adaptive, graph-ordered learning pathway
    5. Return full analysis with reasoning traces
    """
    # --- Validate ---
    _validate_file(resume)
    _validate_file(job_description)

    # --- Extract text ---
    resume_bytes = await resume.read()
    jd_bytes = await job_description.read()

    resume_text = extract_text(resume_bytes, resume.filename or "resume.pdf")
    jd_text = extract_text(jd_bytes, job_description.filename or "jd.pdf")

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Resume text extraction returned empty content.")
    if not jd_text.strip():
        raise HTTPException(status_code=422, detail="JD text extraction returned empty content.")

    # --- Extract skills (LLM / fallback) ---
    resume_data, jd_data = await _extract_both(resume_text, jd_text)

    # --- Skill gap analysis ---
    gap_analysis = compute_skill_gap(resume_data, jd_data)

    # --- Adaptive pathway generation ---
    catalog = load_course_catalog()
    pathway = generate_learning_pathway(gap_analysis, jd_data, catalog)

    # --- Evaluate pathway quality + compute savings ---
    evaluation = evaluate_pathway(gap_analysis, pathway, catalog)

    # --- Build response ---
    return {
        "meta": {
            "resume_filename": resume.filename,
            "jd_filename": job_description.filename,
            "resume_extraction_method": resume_data.get("extraction_method"),
            "jd_extraction_method": jd_data.get("extraction_method"),
        },
        "candidate_profile": {
            "name": resume_data.get("candidate_name"),
            "current_role": resume_data.get("current_role"),
            "years_experience": resume_data.get("years_total_experience"),
            "skills": resume_data.get("skills", []),
        },
        "target_role": {
            "title": jd_data.get("job_title"),
            "department": jd_data.get("department"),
            "seniority": jd_data.get("seniority"),
            "required_skills": jd_data.get("skills", []),
        },
        "gap_analysis": gap_analysis,
        "learning_pathway": pathway,
        "evaluation": evaluation,
    }


async def _extract_both(resume_text: str, jd_text: str):
    """
    Adapter function: Runs the NLP pipeline and maps its complex output 
    to the simple JSON schema expected by the Gap Analyzer.
    """
    # 1. Parse text into sections using the NLP pipeline's parser
    resume_sections = parse_resume_plaintext(resume_text)
    jd_sections = parse_resume_plaintext(jd_text)
    
    # 2. Extract skills using the 5-layer NLP pipeline
    # (Passing a dummy job title list and education list since we only have raw text here)
    raw_resume_profile = extract_all_skills(resume_sections, ["Software Engineer"], [])
    raw_jd_profile = extract_all_skills(jd_sections, ["Target Role"], [])

    # 3. Helper to map 0-3 NLP proficiency to 1-5 System proficiency
    def map_proficiency(nlp_score):
        mapping = {0: 2, 1: 3, 2: 4, 3: 5}
        return mapping.get(nlp_score, 3)

    # 4. Format Resume Skills
    formatted_resume_skills = []
    for skill in raw_resume_profile.get("skills", []) + raw_resume_profile.get("inferred_skills", []):
        formatted_resume_skills.append({
            "skill_name": skill["skill"],
            "category": skill["category"],
            "proficiency_level": map_proficiency(skill.get("proficiency", {}).get("score", 1)),
            "evidence": skill.get("reasoning", "")
        })

    # 5. Format JD Skills
    formatted_jd_skills = []
    for skill in raw_jd_profile.get("skills", []) + raw_jd_profile.get("inferred_skills", []):
        formatted_jd_skills.append({
            "skill_name": skill["skill"],
            "category": skill["category"],
            "required_level": map_proficiency(skill.get("proficiency", {}).get("score", 1)),
            "is_mandatory": True, # Defaulting to true for NLP extraction
            "context": skill.get("reasoning", "")
        })

    # 6. Build final dictionaries matching the old LLM schema
    resume_data = {
        "candidate_name": "Candidate",
        "years_total_experience": raw_resume_profile.get("experience_info", {}).get("total_years", 0),
        "current_role": "Unknown",
        "skills": formatted_resume_skills,
        "extraction_method": "nlp_pipeline"
    }

    jd_data = {
        "job_title": "Target Role",
        "department": "Engineering",
        "seniority": "mid",
        "skills": formatted_jd_skills,
        "extraction_method": "nlp_pipeline"
    }

    return resume_data, jd_data

@app.post("/analyze/visual", tags=["Visualizer"])
async def analyze_visual(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
    category: str = Form("ENGINEER")
):
    """Runs the isolated Result Engine pipeline and returns a Base64 Matplotlib dashboard."""
    
    # 1. Save uploads to temporary files for PyMuPDF processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_res, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
        tmp_res.write(await resume.read())
        tmp_jd.write(await job_description.read())
        res_path = tmp_res.name
        jd_path = tmp_jd.name

    try:
        # 2. Extract and parse using the Result Engine logic
        jd_text = vedant_extract(jd_path)
        jd_req_skills = set(vedant_extract_skills(jd_text, build_skill_library()))

        parsed_res = parse_resume_pdf(res_path)
        cand_row_dict = resume_profile_to_dataframe_row(parsed_res)
        cand_row_dict["Category"] = category
        df = pd.DataFrame([cand_row_dict])

        # 3. Vectorize and compute graph
        vectorizer = build_and_fit_vectorizer(df, job_description_text=jd_text)
        skill_vectors = precompute_skill_vectors(vectorizer)
        graph = build_skill_dependency_graph()

        # 4. Evaluate Candidate
        cand_row = df.iloc
        eval_metrics = evaluate_candidate(cand_row, vectorizer, category=category, jd_skills=jd_req_skills)

        extracted_skills = eval_metrics["Extracted_Skills"]
        gaps = list(jd_req_skills - set(extracted_skills))

        # 5. Prioritize Gaps & Build Roadmap
        prioritized_gaps = prioritize_gaps(gaps, eval_metrics["_cand_vec"], skill_vectors, graph)
        eval_metrics["Prioritized_Gaps"] = prioritized_gaps
        roadmap_df = build_roadmap(eval_metrics)

        # 6. Generate Base64 Dashboard
        base64_image = generate_dashboard(eval_metrics, graph, show=False)

        return {
            "metrics": {
                "Composite_Score": eval_metrics["Composite_Score"],
                "Grade": eval_metrics["Grade"],
                "Pathway_Depth": eval_metrics["Pathway_Depth"],
                "Duration": eval_metrics["Duration"]
            },
            "dashboard_base64": base64_image
        }
    finally:
        # Cleanup temporary files
        import os
        if os.path.exists(res_path): os.remove(res_path)
        if os.path.exists(jd_path): os.remove(jd_path)

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check server logs."},
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("DEV_RELOAD", False)),
        log_level="info",
    )
