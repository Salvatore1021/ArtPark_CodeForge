"""
skills_taxonomy.py
------------------
Master skills taxonomy organized by domain.
In production, replace/extend with O*NET Skills database.
Download from: https://www.onetcenter.org/db_releases.html
"""

SKILLS_TAXONOMY = {

    # ── Software & Programming ──────────────────────────────────────────────
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go",
        "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
        "matlab", "perl", "bash", "shell", "powershell", "vba", "dart",
        "objective-c", "haskell", "elixir", "lua", "groovy",
    ],
    "web_development": [
        "html", "css", "react", "angular", "vue", "node.js", "nodejs",
        "express", "django", "flask", "fastapi", "spring", "asp.net",
        "graphql", "rest api", "restful", "soap", "webpack", "vite",
        "next.js", "nuxt", "svelte", "bootstrap", "tailwind",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "sqlite", "oracle", "cassandra", "dynamodb", "firebase",
        "neo4j", "influxdb", "mariadb", "nosql",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ansible", "jenkins", "ci/cd", "git", "github", "gitlab",
        "bitbucket", "linux", "unix", "nginx", "apache", "heroku",
        "cloudformation", "helm", "prometheus", "grafana",
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "neural networks", "nlp",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "matplotlib", "seaborn", "spark", "hadoop",
        "tableau", "power bi", "data analysis", "data mining",
        "feature engineering", "model training", "statistics",
        "a/b testing", "regression", "classification", "clustering",
        "reinforcement learning", "transformers", "bert", "llm",
    ],

    # ── Agriculture ─────────────────────────────────────────────────────────
    "agriculture_core": [
        "crop management", "plant protection", "soil science", "irrigation",
        "livestock", "agronomy", "horticulture", "pest control",
        "plant disease", "seed technology", "farm management",
        "agricultural extension", "crop production", "fertilizer",
        "greenhouse", "aquaculture", "forestry", "food safety",
        "food sanitation", "home gardening", "kitchen gardening",
        "improved seed", "cultivation", "harvest", "post-harvest",
    ],
    "agriculture_tech": [
        "precision agriculture", "gis", "remote sensing", "drone mapping",
        "drip irrigation", "hydroponics", "vertical farming",
        "agricultural technology", "agritech",
    ],

    # ── Management & Leadership ─────────────────────────────────────────────
    "management": [
        "project management", "strategic planning", "capacity building",
        "team management", "staff supervision", "budget management",
        "stakeholder management", "program management", "operations management",
        "change management", "risk management", "performance management",
        "resource allocation", "planning", "implementation",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "negotiation", "presentation", "public speaking",
        "time management", "adaptability", "conflict resolution",
        "decision making", "mentoring", "coaching",
    ],

    # ── Research & Analysis ─────────────────────────────────────────────────
    "research": [
        "research", "data collection", "needs assessment", "monitoring",
        "evaluation", "field research", "survey design", "qualitative research",
        "quantitative research", "logical framework", "reporting",
        "data analysis", "literature review",
    ],

    # ── Languages & Communication ───────────────────────────────────────────
    "languages": [
        "english", "pashto", "dari", "urdu", "hindi", "arabic", "french",
        "spanish", "german", "chinese", "japanese", "russian", "portuguese",
        "ordo",
    ],

    # ── Office & Tools ──────────────────────────────────────────────────────
    "office_tools": [
        "microsoft word", "microsoft excel", "microsoft powerpoint",
        "microsoft office", "word", "excel", "power point", "powerpoint",
        "outlook", "access", "ms office", "google docs", "google sheets",
        "google slides", "jira", "confluence", "slack", "trello", "notion",
        "salesforce", "sap", "quickbooks",
    ],

    # ── Finance & Business ──────────────────────────────────────────────────
    "finance_business": [
        "financial analysis", "accounting", "budgeting", "forecasting",
        "business development", "marketing", "sales", "business analysis",
        "market research", "supply chain", "procurement",
        "private sector development",
    ],

    # ── Education & Training ────────────────────────────────────────────────
    "education_training": [
        "curriculum development", "training", "workshop facilitation",
        "e-learning", "instructional design", "teaching", "mentoring",
        "training materials", "module development", "capacity building",
    ],

    # ── International Development / NGO ────────────────────────────────────
    "development_ngo": [
        "community development", "humanitarian", "international development",
        "ngo", "donor relations", "grant writing", "advocacy",
        "policy development", "coordination", "partnership",
        "provincial development", "government relations",
    ],

    # ── Healthcare & Clinical ───────────────────────────────────────────────
    "healthcare_clinical": [
        "clinical research", "clinical trials", "patient care", "oncology",
        "pediatric", "medical records", "regulatory compliance",
        "quality assurance", "informed consent", "adverse events",
        "serious adverse events", "irb", "institutional review board",
        "case report form", "protocol compliance", "site initiation",
        "investigational drug", "chart review", "chart audit",
        "corrective action", "corrective action plans", "pharmacovigilance",
        "good clinical practice", "gcp", "hipaa", "healthcare compliance",
        "patient safety", "clinical data management", "biostatistics",
        "epidemiology", "public health", "nursing", "pharmacy",
        "medical coding", "icd coding", "cpt coding", "ehr",
        "electronic health records", "telemedicine", "clinical documentation",
        "patient scheduling", "outpatient", "inpatient",
    ],

    # ── Clinical Research Tools ─────────────────────────────────────────────
    "clinical_research_tools": [
        "qualtrics", "velos", "michart", "eresearch", "redcap",
        "medidata rave", "oracle clinical", "ctms", "clinicaltrials.gov",
        "epic", "cerner", "meditech",
    ],

    # ── Administrative & Operations ─────────────────────────────────────────
    "administrative": [
        "data management", "database management", "calendar management",
        "inventory management", "expense reports", "reconciliation",
        "pivot tables", "proofreading", "proof reading", "audit",
        "standard operating procedures", "sop", "liaison",
        "organizational skills", "scheduling", "record keeping",
        "document management", "filing", "correspondence",
        "minute taking", "report writing", "data entry",
    ],

    # ── Office Tools Extended ───────────────────────────────────────────────
    "office_tools_extended": [
        "google drive", "ms teams", "microsoft teams",
        "zoom", "webex", "sharepoint", "onedrive", "dropbox",
    ],

    # ── IT Support & Infrastructure ─────────────────────────────────────────
    "it_support": [
        "technical support", "help desk", "it support", "service desk",
        "desktop support", "hardware installation", "software installation",
        "troubleshooting", "system administration", "network administration",
        "patch management", "active directory", "windows server",
        "virtualization", "vmware", "hyper-v", "vpn", "firewall",
        "tcp/ip", "dns", "dhcp", "itil", "remote support",
        "incident management", "asset management", "it infrastructure",
        "network monitoring", "backup and recovery", "endpoint management",
    ],

    # ── Telecom & RF Engineering ─────────────────────────────────────────────
    "telecom_rf": [
        "rf optimization", "lte", "cdma", "wireless networks", "telecom",
        "network testing", "drive testing", "field testing", "base station",
        "spectrum analysis", "rf engineering", "network performance",
        "5g", "4g", "3g", "wimax", "ofdm", "voip", "network planning",
        "signal analysis", "radio frequency", "cell planning",
        "network optimization", "mapinfo", "root cause analysis",
        "technical advocacy", "customer advocacy", "first office application",
    ],

    # ── Design & Creative ───────────────────────────────────────────────────
    "design_creative": [
        "graphic design", "adobe illustrator", "adobe photoshop",
        "adobe indesign", "adobe creative suite", "logo design",
        "branding", "typography", "color theory", "print design",
        "digital design", "ui design", "ux design", "web design",
        "mockup", "wireframe", "vector graphics", "layout design",
        "motion graphics", "video editing", "animation", "figma",
        "sketch", "canva", "brand identity", "visual design",
        "content creation", "social media design", "photography",
        "illustration", "art direction", "packaging design",
    ],

    # ── Hospitality & Customer Service ──────────────────────────────────────
    "hospitality_service": [
        "customer service", "guest relations", "front of house",
        "event planning", "reservations management", "food handling",
        "table management", "fine dining", "hospitality",
        "waitstaff", "bartending", "catering", "conflict resolution",
        "complaint handling", "cash handling", "pos systems",
        "upselling", "customer retention", "customer satisfaction",
        "service recovery", "banquet", "food service",
    ],

    # ── Education & Pedagogy ─────────────────────────────────────────────────
    "education_pedagogy": [
        "lesson planning", "curriculum design", "classroom management",
        "student assessment", "differentiated instruction",
        "literacy instruction", "mathematics instruction",
        "special education", "iep", "formative assessment",
        "summative assessment", "common core", "educational technology",
        "learning management system", "behavior management",
        "parent communication", "staff development",
        "educational leadership", "professional development",
        "volunteer management", "public relations",
        "balanced literacy", "guided reading", "phonics instruction",
        "stem education", "blended learning", "project based learning",
        "standards based grading", "instructional coaching",
        "teacher evaluation", "classroom technology",
    ],

    # ── BPO & Call Center ───────────────────────────────────────────────────
    "bpo_call_center": [
        "call center", "bpo", "inbound calls", "outbound calls",
        "customer support", "ticket handling", "escalation management",
        "crm software", "live chat support", "kpi tracking",
        "quality monitoring", "workforce management", "aht",
        "first call resolution", "order processing", "voice process",
        "non-voice process", "email support", "technical troubleshooting",
        "service level agreement", "sla", "call handling", "ivr",
        "helpdesk", "support ticketing", "zendesk", "freshdesk",
        "customer experience",
    ],

    # ── Consulting & Strategy ───────────────────────────────────────────────
    "consulting_strategy": [
        "management consulting", "business consulting", "strategy consulting",
        "process improvement", "gap analysis", "stakeholder engagement",
        "business transformation", "operational excellence", "kpi development",
        "roi analysis", "proposal writing", "client management",
        "executive presentation", "swot analysis", "six sigma",
        "lean methodology", "business process reengineering",
        "workshop facilitation", "benchmarking", "root cause analysis",
        "feasibility study", "due diligence", "business case development",
        "agile consulting",
    ],

    # ── Sales & Business Development Extended ───────────────────────────────
    "sales_extended": [
        "crm", "lead generation", "revenue growth", "p&l management",
        "profit and loss", "digital marketing", "seo", "sem",
        "google adwords", "online marketing", "account management",
        "sales strategy", "brand management", "competitive analysis",
        "market analysis", "pricing strategy",
        "customer relationship management", "sales forecasting",
        "territory management", "dealersocket", "vinsolutions",
        "cold calling", "lead qualification", "sales pipeline",
        "door to door sales", "field sales", "outside sales",
        "inside sales", "b2b sales", "b2c sales", "retail sales",
        "cross-selling", "deal closing",
    ],

    # ── Fitness & Wellness ──────────────────────────────────────────────────
    "fitness_wellness": [
        "personal training", "group fitness", "fitness instruction",
        "nutrition counseling", "wellness coaching", "exercise programming",
        "cpr certified", "aed", "fitness assessment", "class choreography",
        "member retention", "health coaching", "weight management",
        "injury prevention", "fitness programming", "zumba", "yoga",
        "pilates", "strength training", "cardiovascular training",
        "wellness program management", "group exercise",
        "fitness certification", "afaa", "certified personal trainer",
        "cpt", "health promotion", "physical education",
        "sports coaching", "athletic training",
    ],

    # ── Retail, Warehouse & Logistics ───────────────────────────────────────
    "retail_logistics": [
        "retail sales", "merchandising", "stock management",
        "product knowledge", "inventory control", "order fulfillment",
        "logistics", "forklift operation", "warehouse operations",
        "supply chain", "osha compliance", "health and safety",
        "shipping and receiving", "packing", "picking",
        "quality control", "lean manufacturing", "warehouse management",
        "freight handling", "delivery management", "fleet management",
        "route planning", "goods receiving", "stock replenishment",
        "point of sale", "pos", "cash register",
    ],

    # ── Automotive & Dealership ─────────────────────────────────────────────
    "automotive": [
        "automotive sales", "vehicle sales", "test drive", "car dealership",
        "auto financing", "f&i", "finance and insurance", "vehicle inventory",
        "trade-in appraisal", "lease negotiation", "car sales",
        "automotive crm", "dealership management", "lot management",
        "vehicle inspection", "auto repair", "service advisor",
        "parts management", "automotive technology", "cdp",
        "reynolds and reynolds", "dealer management system", "dms",
        "desking", "automotive marketing", "automotive financing",
        "vehicle appraisal", "new car sales", "used car sales",
        "fleet sales", "aftermarket sales", "service department",
    ],

    # ── Culinary & Food Service ─────────────────────────────────────────────
    "culinary": [
        "cooking", "food preparation", "menu planning", "menu development",
        "catering", "kitchen management", "food cost control",
        "recipe development", "culinary arts", "sous vide", "baking",
        "pastry", "butchery", "food presentation", "kitchen operations",
        "meal prep", "portion control", "food ordering",
        "kitchen staff management", "haccp", "allergen awareness",
        "food hygiene", "international cuisine", "line cooking",
        "prep cook", "banquet cooking", "buffet management",
        "sous chef", "head chef", "executive chef",
        "food costing", "kitchen scheduling", "catering management",
        "food and beverage", "barista", "sommelier",
    ],

    # ── Digital Media & Ad Tech ─────────────────────────────────────────────
    "digital_media": [
        "programmatic advertising", "demand side platform", "dsp",
        "media buying", "media planning", "digital advertising",
        "display advertising", "online video advertising",
        "mobile advertising", "social media advertising",
        "remarketing", "retargeting", "audience targeting",
        "data driven marketing", "ad tech", "campaign optimization",
        "kpi reporting", "media strategy", "rfp", "cpa marketing",
        "roi tracking", "ad trafficking", "double click",
        "google dv360", "the trade desk", "comscore", "nielsen",
        "media analytics", "media mix modeling", "attribution modeling",
        "paid social", "paid search", "display campaigns",
        "video campaigns", "media budget management",
        "supply side platform", "ssp", "ad exchange",
        "behavioral targeting", "contextual targeting",
        "scarborough", "mri", "media research",
    ],
}

# Flat map: skill_string -> category (for fast reverse lookup)
SKILL_TO_CATEGORY = {}
for category, skills in SKILLS_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill.lower()] = category
