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

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gap_analyzer import compute_skill_gap
from metrics import evaluate_pathway
from parsers import extract_text
from pathway_engine import generate_learning_pathway, load_course_catalog
from skill_extractor import extract_skills_from_jd, extract_skills_from_resume

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
    """Run resume + JD extraction concurrently."""
    import asyncio
    resume_task = asyncio.create_task(extract_skills_from_resume(resume_text))
    jd_task = asyncio.create_task(extract_skills_from_jd(jd_text))
    resume_data = await resume_task
    jd_data = await jd_task
    return resume_data, jd_data


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
