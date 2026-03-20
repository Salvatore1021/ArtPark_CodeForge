"""
config.py
---------
Central configuration: file paths, grade thresholds,
category benchmarks, and skill-dependency edge definitions.
"""

# ── File Paths ──────────────────────────────────────────────────────────────
RESUME_CSV_PATH = "data/Resume.csv"
JOB_DESC_PDF_PATH = "data/job_description.pdf"
OUTPUT_DIR = "output"

# ── Grading Thresholds ──────────────────────────────────────────────────────
GRADE_THRESHOLDS = {
    "A": 0.80,
    "B": 0.60,
    "C": 0.40,
    "D": 0.20,
}

# ── Pathway Rules ───────────────────────────────────────────────────────────
FUNDAMENTAL_GRADES = {"D", "F"}
WEEK_COUNT_FUNDAMENTAL = 8
WEEK_COUNT_ADVANCED = 4

# ── Weighted Fit Score Split ────────────────────────────────────────────────
HARD_SKILL_WEIGHT = 0.70
SOFT_SKILL_WEIGHT = 0.30

# ── Category Benchmarks ─────────────────────────────────────────────────────
# Maps resume Category label → list of expected skills for that role
CATEGORY_BENCHMARKS = {
    "ACCOUNTANT": [
        "Taxation", "Auditing", "Financial Reporting", "Payroll",
        "QuickBooks", "Excel", "Accounting", "Communication", "Teamwork",
    ],
    "ENGINEER": [
        "AutoCAD", "SolidWorks", "Project Management", "Manufacturing",
        "Technical Support", "Problem Solving", "Critical Thinking",
    ],
    "HR": [
        "Recruitment", "Onboarding", "Employee Relations",
        "Benefits Administration", "Communication", "Leadership",
    ],
    "MARKETING": [
        "SEO", "Content Strategy", "Digital Strategy", "Social Media",
        "Analytics", "Communication", "Adaptability",
    ],
    "LEGAL": [
        "Compliance", "Intellectual Property", "Contract Law",
        "Regulatory Affairs", "Critical Thinking", "Communication",
    ],
    "HEALTHCARE": [
        "HIPAA", "Patient Care", "Medical Records", "Clinical Research",
        "Communication", "Teamwork", "Adaptability",
    ],
    "DATA SCIENCE": [
        "Python", "Machine Learning", "Data Science", "NLP", "SQL",
        "TensorFlow", "PyTorch", "Statistics", "Communication",
    ],
}

DEFAULT_BENCHMARK = [
    "Communication", "Teamwork", "Problem Solving",
    "Adaptability", "Time Management", "Leadership",
]

# ── Skill Dependency Edges ──────────────────────────────────────────────────
# Format: (Prerequisite, Advanced Skill)
SKILL_DEPENDENCY_EDGES = [
    # Data Science / AI
    ("Python", "Data Science"),
    ("SQL", "Data Science"),
    ("Data Science", "Machine Learning"),
    ("Machine Learning", "Deep Learning"),
    ("Machine Learning", "Model Tuning"),
    ("Deep Learning", "PyTorch"),
    ("Deep Learning", "TensorFlow"),
    ("Data Science", "Data Visualization"),
    ("Statistics", "Machine Learning"),

    # Accounting / Finance
    ("Excel", "Accounting"),
    ("Accounting", "Financial Reporting"),
    ("Accounting", "Auditing"),
    ("Accounting", "Taxation"),
    ("QuickBooks", "Accounting"),
    ("Financial Reporting", "Regulatory Affairs"),

    # Engineering
    ("AutoCAD", "Engineering"),
    ("SolidWorks", "Engineering"),
    ("Engineering", "Manufacturing"),
    ("Engineering", "Operations"),

    # Business / Soft Skills
    ("Communication", "Leadership"),
    ("Teamwork", "Leadership"),
    ("Problem Solving", "Strategic Planning"),
    ("Market Research", "Marketing"),
    ("Marketing", "Digital Strategy"),
    ("SEO", "Content Strategy"),
    ("Content Strategy", "Digital Strategy"),

    # Healthcare
    ("Patient Care", "Clinical Research"),
    ("Medical Records", "HIPAA"),
    ("HIPAA", "Healthcare Compliance"),

    # HR
    ("Recruitment", "Onboarding"),
    ("Onboarding", "Employee Relations"),
]

# ── Learning Objectives & Success Criteria per Skill ───────────────────────
LEARNING_META = {
    "QuickBooks":          {"Objective": "Mastering Automated Ledger Entries",          "Success": "Complete a mock month-end close"},
    "Accounting":          {"Objective": "Advanced Financial Statement Analysis",        "Success": "Accurately interpret a 10-K report"},
    "Auditing":            {"Objective": "Internal Control Testing & Risk Assessment",   "Success": "Identify 3 major risks in a case study"},
    "Taxation":            {"Objective": "Corporate Tax Compliance & Filing",            "Success": "Prepare a basic corporate tax return"},
    "Python":              {"Objective": "Core Python Programming & Data Structures",    "Success": "Build a working ETL pipeline"},
    "Machine Learning":    {"Objective": "Supervised & Unsupervised Learning Algorithms","Success": "Train and evaluate a classification model"},
    "Deep Learning":       {"Objective": "Neural Network Architecture Design",           "Success": "Achieve >90% accuracy on MNIST dataset"},
    "SQL":                 {"Objective": "Database Querying & Schema Design",            "Success": "Write complex joins and window functions"},
    "Communication":       {"Objective": "Professional Communication Strategies",        "Success": "Deliver a mock stakeholder presentation"},
    "Leadership":          {"Objective": "Team Leadership & Delegation",                 "Success": "Lead a simulated sprint retrospective"},
    "Patient Care":        {"Objective": "Patient-Centred Care Standards",               "Success": "Pass GCP certification module"},
    "HIPAA":               {"Objective": "Healthcare Data Privacy Compliance",           "Success": "Complete HIPAA awareness assessment"},
}
LEARNING_META_DEFAULT = {"Objective": "General Proficiency Development", "Success": "Certification or practical assessment"}
