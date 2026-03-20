"""
config.py
---------
Central configuration.
Category benchmarks are now DRIVEN BY the enhanced taxonomy via taxonomy_adapter;
this file keeps only structural constants, edge definitions, and learning meta.
"""

# ── File paths ──────────────────────────────────────────────────────────────
RESUME_CSV_PATH   = "data/Resume.csv"
JOB_DESC_PDF_PATH = "data/job_description.pdf"
OUTPUT_DIR        = "output"

# ── Benchmark settings ──────────────────────────────────────────────────────
BENCHMARK_TOP_N  = 35     # top-N weighted skills pulled per role (covers both O*NET + tech)
MIN_SKILL_WEIGHT = 2.0     # minimum O*NET weight to treat a skill as required

# ── Grading thresholds ──────────────────────────────────────────────────────
GRADE_THRESHOLDS = {"A": 0.80, "B": 0.60, "C": 0.40, "D": 0.20}
FUNDAMENTAL_GRADES     = {"D", "F"}
WEEK_COUNT_FUNDAMENTAL = 8
WEEK_COUNT_ADVANCED    = 4

# ── CSV Category → Taxonomy job title map ───────────────────────────────────
CATEGORY_TO_JOB_TITLE: dict[str, str] = {
    # Tech roles — mapped to taxonomy entries that use proxy O*NET weights
    "DATA SCIENCE":         "Machine Learning",
    "DATA SCIENTIST":       "Machine Learning",
    "MACHINE LEARNING":     "Machine Learning",
    "AI ENGINEER":          "Machine Learning",
    "SOFTWARE ENGINEER":    "Software Engineer",
    "SOFTWARE DEVELOPER":   "Software Engineer",
    "DATABASE ADMIN":       "Database Administrator",
    "DBA":                  "Database Administrator",
    "BACKEND DEVELOPER":    "Backend Developer",
    "DEVOPS":               "DevOps Engineer",
    "DEVOPS ENGINEER":      "DevOps Engineer",
    "JAVA DEVELOPER":       "Java Developer",
    "FULL STACK":           "Full Stack Developer",
    "FULL STACK DEVELOPER": "Full Stack Developer",
    "WEB DEVELOPER":        "JavaScript Developer",
    "FRONTEND":             "JavaScript Developer",
    "FRONTEND DEVELOPER":   "JavaScript Developer",
    "IOS DEVELOPER":        "iOS Developer",
    "FLUTTER DEVELOPER":    "Flutter Developer",
    "NETWORK ADMIN":        "Network Administrator",
    # O*NET-weighted roles (direct taxonomy match)
    "ACCOUNTANT":           "Accountants and Auditors",
    "ACCOUNTING":           "Accountants and Auditors",
    "HR":                   "Human Resources Specialists",
    "HUMAN RESOURCES":      "Human Resources Specialists",
    "MARKETING":            "Market Research Analysts and Marketing Specialists",
    "NURSE":                "Registered Nurses",
    "NURSING":              "Registered Nurses",
    "TEACHER":              "Elementary School Teachers, Except Special Education",
    "EDUCATION":            "Education Administrators, Kindergarten through Secondary",
    "LAWYER":               "Lawyers",
    "LEGAL":                "Lawyers",
    "HEALTHCARE":           "Physicians and Surgeons, All Other",
    "DOCTOR":               "Physicians and Surgeons, All Other",
    "ENGINEER":             "Industrial Machinery Mechanics",
    "MECHANICAL ENGINEER":  "Mechanical Engineers",
    "ELECTRICAL ENGINEER":  "Electrical Engineers",
    "CIVIL ENGINEER":       "Civil Engineers",
    "OPERATIONS":           "Operations Research Analysts",
    "ANALYST":              "Operations Research Analysts",
    "UPLOADED_PDF":         "Software Engineer",
}

# ── Dependency graph edges ──────────────────────────────────────────────────
SKILL_DEPENDENCY_EDGES: list[tuple[str, str]] = [
    # Generic cognitive
    ("active listening",                  "social perceptiveness"),
    ("reading comprehension",             "critical thinking"),
    ("critical thinking",                 "complex problem solving"),
    ("critical thinking",                 "judgment and decision making"),
    ("mathematics",                       "systems analysis"),
    ("systems analysis",                  "systems evaluation"),
    ("systems analysis",                  "operations analysis"),
    ("active learning",                   "learning strategies"),
    ("complex problem solving",           "operations analysis"),
    ("monitoring",                        "quality control analysis"),
    ("speaking",                          "instructing"),
    ("coordination",                      "management of personnel resources"),
    ("management of personnel resources", "management of financial resources"),
    ("management of material resources",  "management of financial resources"),
    # Data Science / ML
    ("mathematics",      "statistics"),
    ("statistics",       "machine learning"),
    ("python",           "data science"),
    ("sql",              "data science"),
    ("data science",     "machine learning"),
    ("machine learning", "deep learning"),
    ("machine learning", "model tuning"),
    ("machine learning", "feature engineering"),
    ("deep learning",    "pytorch"),
    ("deep learning",    "tensorflow"),
    ("deep learning",    "computer vision"),
    ("deep learning",    "natural language processing"),
    ("python",           "pandas"),
    ("python",           "numpy"),
    ("pandas",           "data science"),
    ("numpy",            "machine learning"),
    ("statistics",       "a/b testing"),
    ("python",           "scikit-learn"),
    ("scikit-learn",     "machine learning"),
    ("data science",     "data visualization"),
    ("data visualization","tableau"),
    ("data visualization","matplotlib"),
    # Software Engineering
    ("programming",      "software engineer"),
    ("python",           "django"),
    ("python",           "flask"),
    ("python",           "fastapi"),
    ("javascript",       "node.js"),
    ("javascript",       "react"),
    ("javascript",       "vue.js"),
    ("javascript",       "typescript"),
    ("java",             "spring boot"),
    ("java",             "hibernate"),
    ("java",             "jpa"),
    ("programming",      "algorithms"),
    ("algorithms",       "data structures"),
    ("data structures",  "system design"),
    # DevOps / Cloud
    ("linux",            "bash / shell scripting"),
    ("linux",            "nginx"),
    ("programming",      "ci/cd"),
    ("ci/cd",            "docker"),
    ("docker",           "kubernetes"),
    ("kubernetes",       "helm charts"),
    ("ci/cd",            "jenkins"),
    ("ci/cd",            "gitlab ci / github actions"),
    ("docker",           "terraform modules"),
    ("terraform modules","ansible playbooks"),
    ("monitoring",       "prometheus / grafana"),
    ("monitoring",       "elk stack"),
    # Database
    ("sql",              "mysql"),
    ("sql",              "postgresql"),
    ("sql",              "oracle db"),
    ("programming",      "mongodb"),
    ("mongodb",          "elasticsearch"),
    ("sql",              "redis"),
    # Accounting / Finance
    ("mathematics",      "accounting"),
    ("accounting",       "financial reporting"),
    ("accounting",       "auditing"),
    ("accounting",       "taxation"),
    ("mathematics",      "budgeting"),
    ("budgeting",        "financial analysis"),
    # Healthcare
    ("science",          "patient care"),
    ("patient care",     "clinical research"),
    ("clinical research","good clinical practice"),
    ("science",          "pharmacology"),
    ("pharmacology",     "pharmacovigilance"),
    # Education
    ("instructing",      "learning strategies"),
    ("learning strategies","curriculum design"),
    ("curriculum design","instructional design"),
    # Management
    ("coordination",     "project management"),
    ("project management","strategic planning"),
    ("judgment and decision making","management of personnel resources"),
]

# ── Learning objectives & success criteria ──────────────────────────────────
LEARNING_META: dict[str, dict[str, str]] = {
    "critical thinking":           {"Objective": "Applied Critical Analysis",              "Success": "Resolve a complex case-study scenario"},
    "complex problem solving":     {"Objective": "Structured Problem-Solving Methods",     "Success": "Present root-cause analysis for a mock incident"},
    "systems analysis":            {"Objective": "System Mapping & Bottleneck Identification","Success": "Produce a system analysis report"},
    "operations analysis":         {"Objective": "Process Optimisation Techniques",        "Success": "Identify 3 optimisation opportunities in a workflow"},
    "judgment and decision making":{"Objective": "Evidence-Based Decision Frameworks",     "Success": "Pass a simulated decision-making assessment"},
    "active listening":            {"Objective": "Active Listening & Stakeholder Comms",   "Success": "Conduct a mock stakeholder interview"},
    "instructing":                 {"Objective": "Instructional Design Fundamentals",      "Success": "Deliver a 10-min micro-training session"},
    "mathematics":                 {"Objective": "Applied Business Mathematics",           "Success": "Complete a quantitative reasoning assessment"},
    "programming":                 {"Objective": "Foundational Programming Concepts",      "Success": "Build a working CLI tool"},
    "quality control analysis":    {"Objective": "Quality Assurance Processes",            "Success": "Write and execute a QA checklist"},
    "time management":             {"Objective": "Personal Productivity Systems",          "Success": "Deliver a 2-week mini-project on schedule"},
    "management of personnel resources":{"Objective":"Team Leadership Fundamentals",       "Success": "Lead a simulated sprint retrospective"},
    "management of financial resources":{"Objective":"Budget Planning & Control",          "Success": "Prepare a departmental budget model"},
    "python":                      {"Objective": "Core Python & Data Structures",          "Success": "Build a working ETL pipeline"},
    "machine learning":            {"Objective": "Supervised & Unsupervised Algorithms",   "Success": "Train and evaluate a classification model"},
    "deep learning":               {"Objective": "Neural Network Architecture Design",     "Success": "Achieve >90% accuracy on a benchmark dataset"},
    "pytorch":                     {"Objective": "PyTorch Model Development",              "Success": "Implement and train a CNN from scratch"},
    "tensorflow":                  {"Objective": "TensorFlow & Keras Workflows",           "Success": "Deploy a model with TensorFlow Serving"},
    "natural language processing": {"Objective": "NLP Pipeline Design",                   "Success": "Build a sentiment analysis classifier"},
    "computer vision":             {"Objective": "CV Model Development (YOLO/ResNet)",    "Success": "Train a real-time object detector"},
    "statistics":                  {"Objective": "Inferential Statistics & Hypothesis Testing","Success": "Complete a statistical analysis report"},
    "sql":                         {"Objective": "Advanced SQL Querying",                  "Success": "Write complex window functions and CTEs"},
    "data visualization":          {"Objective": "Dashboard Design Principles",            "Success": "Build a 3-panel interactive dashboard"},
    "docker":                      {"Objective": "Container Fundamentals",                "Success": "Dockerize a multi-service application"},
    "kubernetes":                  {"Objective": "Kubernetes Cluster Management",          "Success": "Deploy and scale a workload on a local cluster"},
    "ci/cd":                       {"Objective": "Pipeline Design & Automation",           "Success": "Build a GitHub Actions pipeline with tests"},
    "terraform modules":           {"Objective": "Infrastructure as Code (Terraform)",     "Success": "Provision a cloud VPC with Terraform"},
    "prometheus / grafana":        {"Objective": "Observability & Alerting Setup",         "Success": "Create a live monitoring dashboard"},
    "accounting":                  {"Objective": "Advanced Financial Statement Analysis",  "Success": "Accurately interpret a 10-K report"},
    "auditing":                    {"Objective": "Internal Control Testing",               "Success": "Identify 3 major risks in a case study"},
    "taxation":                    {"Objective": "Corporate Tax Compliance & Filing",      "Success": "Prepare a basic corporate tax return"},
    "patient care":                {"Objective": "Patient-Centred Care Standards",         "Success": "Pass a GCP certification module"},
    "good clinical practice":      {"Objective": "GCP Compliance & Protocol Adherence",    "Success": "Complete GCP certification"},
    "clinical research":           {"Objective": "Clinical Trial Management",              "Success": "Design a mock Phase II trial protocol"},
}
LEARNING_META_DEFAULT = {
    "Objective": "General Proficiency Development",
    "Success":   "Certification or practical assessment",
}
