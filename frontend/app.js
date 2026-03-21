/**
 * app.js — AURORA Frontend
 * Handles: file uploads, drag & drop, API calls, radar chart, pathway rendering
 */

const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:8000"
  : "/api";  // production: nginx proxy

// ============================================================
// State
// ============================================================
let resumeFile = null;
let jdFile = null;
let radarChart = null;
let analysisResult = null;

// ============================================================
// DOM refs
// ============================================================
const resumeInput     = document.getElementById("resume-input");
const jdInput         = document.getElementById("jd-input");
const resumeDropZone  = document.getElementById("resume-drop-zone");
const jdDropZone      = document.getElementById("jd-drop-zone");
const resumeName      = document.getElementById("resume-name");
const jdName          = document.getElementById("jd-name");
const analyzeBtn      = document.getElementById("analyze-btn");
const loadingOverlay  = document.getElementById("loading-overlay");
const resultsSection  = document.getElementById("results-section");
const uploadSection   = document.getElementById("upload-section");
const resetBtn        = document.getElementById("reset-btn");
const errorToast      = document.getElementById("error-toast");
const toastMsg        = document.getElementById("toast-msg");
const toastClose      = document.getElementById("toast-close");

// ============================================================
// File handling
// ============================================================
function handleFileSelect(file, type) {
  if (!file) return;
  const MAX_MB = 10;
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(`File too large: ${file.name} (max ${MAX_MB}MB)`);
    return;
  }
  const allowed = [".pdf", ".docx", ".doc", ".txt", ".md"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError(`File type not supported: ${ext}. Use PDF, DOCX, or TXT.`);
    return;
  }

  if (type === "resume") {
    resumeFile = file;
    resumeName.textContent = file.name;
    resumeDropZone.classList.add("has-file");
  } else {
    jdFile = file;
    jdName.textContent = file.name;
    jdDropZone.classList.add("has-file");
  }
  checkAnalyzeReady();
}

function checkAnalyzeReady() {
  analyzeBtn.disabled = !resumeFile;
}

// Input change events
resumeInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0], "resume"));
jdInput.addEventListener("change",     (e) => handleFileSelect(e.target.files[0], "jd"));

// Drag & drop
function setupDragDrop(zone, type) {
  zone.addEventListener("dragover",  (e) => { e.preventDefault(); zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", ()  => { zone.classList.remove("dragging"); });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragging");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileSelect(file, type);
  });
  // Clicking the zone (not the button) also triggers file picker
  zone.addEventListener("click", (e) => {
    if (e.target.classList.contains("upload-btn") || e.target.tagName === "INPUT") return;
    type === "resume" ? resumeInput.click() : jdInput.click();
  });
}
setupDragDrop(resumeDropZone, "resume");
setupDragDrop(jdDropZone, "jd");

// ============================================================
// Loading step animation
// ============================================================
const STEPS = ["step-parse", "step-extract", "step-gap", "step-path"];
let stepInterval = null;

function startLoadingAnimation() {
  let idx = 0;
  STEPS.forEach((id) => {
    const el = document.getElementById(id);
    el.classList.remove("active", "done");
  });
  document.getElementById(STEPS[0]).classList.add("active");
  stepInterval = setInterval(() => {
    document.getElementById(STEPS[idx]).classList.replace("active", "done");
    idx++;
    if (idx < STEPS.length) {
      document.getElementById(STEPS[idx]).classList.add("active");
    } else {
      clearInterval(stepInterval);
    }
  }, 3500);
}

function stopLoadingAnimation() {
  clearInterval(stepInterval);
  STEPS.forEach((id) => {
    document.getElementById(id).classList.remove("active", "done");
  });
}

// ============================================================
// API call
// ============================================================
analyzeBtn.addEventListener("click", async () => {
  if (!resumeFile) return;

  // Show loading
  loadingOverlay.hidden = false;
  startLoadingAnimation();
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("resume", resumeFile);
    if (jdFile) {
      formData.append("job_description", jdFile);
    }

    // Add default category required by the new visual endpoint
    formData.append("category", "ENGINEER");

    // Fetch from BOTH endpoints simultaneously
    const [standardResp, visualResp] = await Promise.all([
      fetch(`${API_BASE}/analyze`, { method: "POST", body: formData }),
      fetch(`${API_BASE}/analyze/visual`, { method: "POST", body: formData })
    ]);

    if (!standardResp.ok) {
      const errData = await standardResp.json().catch(() => ({}));
      throw new Error(errData.detail || `Standard API error ${standardResp.status}`);
    }
    if (!visualResp.ok) {
      const errData = await visualResp.json().catch(() => ({}));
      throw new Error(errData.detail || `Visual API error ${visualResp.status}`);
    }

    analysisResult = await standardResp.json();
    const visualResult = await visualResp.json();

    // 1. Render original UI
    renderResults(analysisResult);

    // 2. Render new Matplotlib Dashboard
    const visualContainer = document.getElementById("visual-results-container");
    const dashboardImg = document.getElementById("dashboard-image");
    
    if (visualContainer && dashboardImg && visualResult.dashboard_base64) {
      visualContainer.style.display = "block";
      dashboardImg.src = "data:image/png;base64," + visualResult.dashboard_base64;
    }

  } catch (err) {
    showError(err.message || "Network error — is the backend running?");
    analyzeBtn.disabled = false;
  } finally {
    loadingOverlay.hidden = true;
    stopLoadingAnimation();
  }
});

// ============================================================
// Render results
// ============================================================
function renderResults(data) {
  // Scroll to results
  resultsSection.hidden = false;
  uploadSection.style.display = "none";
  resultsSection.scrollIntoView({ behavior: "smooth" });

  const { candidate_profile, target_role, gap_analysis, learning_pathway } = data;

  // Summary bar
  document.getElementById("sum-role").textContent     = target_role?.title || "—";
  document.getElementById("sum-coverage").textContent = `${gap_analysis?.gap_summary?.coverage_pct ?? 0}%`;
  document.getElementById("sum-gaps").textContent     = gap_analysis?.gap_summary?.total_gaps ?? 0;
  document.getElementById("sum-weeks").textContent    = learning_pathway?.estimated_weeks ?? 0;
  document.getElementById("sum-hours").textContent    = `${learning_pathway?.estimated_total_hours ?? 0}h`;

  // Candidate profile
  renderCandidateProfile(candidate_profile);

  // Radar chart
  renderRadarChart(candidate_profile?.skills || [], target_role?.required_skills || []);

  // Reasoning
  document.getElementById("gap-reasoning").textContent     = gap_analysis?.reasoning || "";
  document.getElementById("pathway-reasoning").textContent = learning_pathway?.reasoning_summary || "";

  // Learning pathway phases
  renderPathway(learning_pathway?.phases || []);
}

// ---- Candidate profile ----
function renderCandidateProfile(profile) {
  const metaEl = document.getElementById("candidate-meta");
  metaEl.innerHTML = `
    <div class="meta-name">${profile?.name || "Candidate"}</div>
    <div class="meta-detail">
      ${profile?.current_role ? `<strong>${profile.current_role}</strong> · ` : ""}
      ${profile?.years_experience != null ? `${profile.years_experience} yrs experience` : ""}
    </div>
  `;

  const listEl = document.getElementById("resume-skills-list");
  listEl.innerHTML = "";
  (profile?.skills || []).forEach((skill) => {
    const chip = document.createElement("span");
    chip.className = "skill-chip skill-chip--neutral";
    const levelBar = "█".repeat(skill.proficiency_level || 0) + "░".repeat(5 - (skill.proficiency_level || 0));
    chip.innerHTML = `${skill.skill_name} <span class="skill-level" title="Level ${skill.proficiency_level}/5">${levelBar}</span>`;
    chip.title = `Level ${skill.proficiency_level}/5 — ${skill.evidence || ""}`;
    listEl.appendChild(chip);
  });
}

// ---- Radar chart ----
const RADAR_MAX_SKILLS = 10;

function renderRadarChart(resumeSkills, requiredSkills) {
  if (radarChart) { radarChart.destroy(); radarChart = null; }

  // Pick top N required skills for radar
  const topRequired = requiredSkills.slice(0, RADAR_MAX_SKILLS);
  const labels = topRequired.map((s) => s.skill_name);

  // Build resume lookup
  const resumeMap = {};
  resumeSkills.forEach((s) => { resumeMap[s.skill_name.toLowerCase()] = s.proficiency_level || 0; });

  const currentData = topRequired.map((s) => resumeMap[s.skill_name.toLowerCase()] || 0);
  const requiredData = topRequired.map((s) => s.required_level || 3);

  const ctx = document.getElementById("gap-radar-chart");
  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Required",
          data: requiredData,
          borderColor: "rgba(240, 165, 0, 0.8)",
          backgroundColor: "rgba(240, 165, 0, 0.1)",
          pointBackgroundColor: "rgba(240, 165, 0, 1)",
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: "Current",
          data: currentData,
          borderColor: "rgba(45, 212, 160, 0.8)",
          backgroundColor: "rgba(45, 212, 160, 0.1)",
          pointBackgroundColor: "rgba(45, 212, 160, 1)",
          borderWidth: 2,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        r: {
          min: 0, max: 5,
          ticks: { stepSize: 1, color: "#52596e", font: { size: 10 }, backdropColor: "transparent" },
          grid: { color: "rgba(255,255,255,0.06)" },
          angleLines: { color: "rgba(255,255,255,0.06)" },
          pointLabels: { color: "#8892aa", font: { size: 10, family: "'DM Sans', sans-serif" } },
        },
      },
      plugins: {
        legend: {
          labels: { color: "#8892aa", font: { size: 11 }, boxWidth: 12, padding: 16 },
        },
        tooltip: {
          backgroundColor: "#12151c",
          borderColor: "#242838",
          borderWidth: 1,
          titleColor: "#f0f2f8",
          bodyColor: "#8892aa",
        },
      },
    },
  });
}

// ---- Learning pathway ----
const LEVEL_COLORS = { beginner: "beginner", intermediate: "intermediate", advanced: "advanced" };

function renderPathway(phases) {
  const container = document.getElementById("pathway-container");
  container.innerHTML = "";

  if (!phases || phases.length === 0) {
    container.innerHTML = `<div class="panel" style="padding:2rem;text-align:center;color:var(--text-muted);">
      No gaps found — you already meet the requirements for this role! 🎉
    </div>`;
    return;
  }

  phases.forEach((phase, phaseIdx) => {
    const block = document.createElement("div");
    block.className = "phase-block";
    block.style.animationDelay = `${phaseIdx * 0.1}s`;

    block.innerHTML = `
      <div class="phase-header">
        <div class="phase-badge">${phase.phase}</div>
        <div class="phase-title">${phase.title}</div>
        <div class="phase-hours">~${phase.total_hours}h · ${Math.ceil(phase.total_hours / 4)}–${Math.ceil(phase.total_hours / 2)} days</div>
      </div>
      <div class="courses-grid" id="phase-${phase.phase}-courses"></div>
    `;

    container.appendChild(block);

    const coursesContainer = document.getElementById(`phase-${phase.phase}-courses`);
    (phase.courses || []).forEach((course) => {
      coursesContainer.appendChild(buildCourseCard(course));
    });
  });
}

function buildCourseCard(course) {
  const card = document.createElement("div");
  card.className = `course-card${course.is_prerequisite_only ? " course-card--prereq" : ""}`;

  const levelClass = LEVEL_COLORS[course.level] || "intermediate";
  const gapTags = (course.addresses_gaps || []).slice(0, 4)
    .map((g) => `<span class="gap-tag">${g}</span>`).join("");
  const prereqTag = course.is_prerequisite_only
    ? `<span class="gap-tag gap-tag--prereq">prerequisite</span>` : "";

  const modulesHtml = (course.modules || [])
    .map((m) => `<span class="module-chip">${m}</span>`).join("");

  card.innerHTML = `
    <div class="course-card-top">
      <span class="course-id">${course.id}</span>
      <span class="course-level-badge course-level-badge--${levelClass}">${course.level}</span>
    </div>
    <div class="course-name">${course.name}</div>
    <div class="course-desc">${course.description}</div>
    <div class="course-meta">
      <span class="course-hours">⏱ ${course.duration_hours}h</span>
      <span class="course-category">${formatCategory(course.category)}</span>
    </div>
    ${gapTags || prereqTag ? `
      <div class="course-gaps">
        <div class="course-gaps-label">Addresses</div>
        <div class="course-gap-tags">${gapTags}${prereqTag}</div>
      </div>
    ` : ""}
    <div class="course-reasoning">${course.reasoning || ""}</div>
    ${modulesHtml ? `
      <div class="course-modules">
        <div class="course-modules-toggle" onclick="toggleModules(this)">
          ▸ Modules (${course.modules.length})
        </div>
        <div class="modules-list">${modulesHtml}</div>
      </div>
    ` : ""}
  `;
  return card;
}

function formatCategory(cat) {
  const map = {
    "programming":    "Programming",
    "data-science":   "Data Science",
    "machine-learning": "ML/AI",
    "cloud":          "Cloud",
    "devops":         "DevOps",
    "web":            "Web Dev",
    "data":           "Data Eng",
    "soft-skills":    "Soft Skills",
    "security":       "Security",
    "business":       "Business",
  };
  return map[cat] || cat;
}

window.toggleModules = function(el) {
  const list = el.nextElementSibling;
  const open = list.classList.toggle("open");
  el.textContent = (open ? "▾" : "▸") + el.textContent.slice(1);
};

// ============================================================
// Reset
// ============================================================
resetBtn.addEventListener("click", () => {
  resumeFile = null;
  jdFile = null;
  analysisResult = null;

  resumeName.textContent = "No file selected";
  jdName.textContent = "No file selected";
  resumeInput.value = "";
  jdInput.value = "";
  resumeDropZone.classList.remove("has-file");
  jdDropZone.classList.remove("has-file");

  resultsSection.hidden = true;
  uploadSection.style.display = "";
  analyzeBtn.disabled = true;

  if (radarChart) { radarChart.destroy(); radarChart = null; }

  uploadSection.scrollIntoView({ behavior: "smooth" });
});

// ============================================================
// Error toast
// ============================================================
function showError(msg) {
  toastMsg.textContent = msg;
  errorToast.hidden = false;
  setTimeout(() => { errorToast.hidden = true; }, 6000);
}
toastClose.addEventListener("click", () => { errorToast.hidden = true; });
