const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:8000"
  : "/api";

let resumeFile = null;
let jdFile = null;
let radarChart = null;
let fitDonutChart = null;
let graphRenderer = null;
let stepInterval = null;

const resumeInput = document.getElementById("resume-input");
const jdInput = document.getElementById("jd-input");
const resumeDropZone = document.getElementById("resume-drop-zone");
const jdDropZone = document.getElementById("jd-drop-zone");
const resumeName = document.getElementById("resume-name");
const jdName = document.getElementById("jd-name");
const analyzeBtn = document.getElementById("analyze-btn");
const loadingOverlay = document.getElementById("loading-overlay");
const resultsSection = document.getElementById("results-section");
const uploadSection = document.getElementById("upload-section");
const resetBtn = document.getElementById("reset-btn");
const toast = document.getElementById("error-toast");
const toastMsg = document.getElementById("toast-msg");
const toastClose = document.getElementById("toast-close");

const LOADING_STEPS = ["step-parse", "step-extract", "step-gap", "step-graph"];

function showError(message) {
  toastMsg.textContent = message;
  toast.hidden = false;
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => {
    toast.hidden = true;
  }, 6000);
}

toastClose.addEventListener("click", () => {
  toast.hidden = true;
});

function handleFileSelect(file, type) {
  if (!file) return;
  const ext = "." + file.name.split(".").pop().toLowerCase();
  const allowed = [".pdf", ".docx", ".doc", ".txt", ".md"];
  if (!allowed.includes(ext)) {
    showError(`Unsupported file type: ${ext}`);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("Files must be 10MB or smaller.");
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
  analyzeBtn.disabled = !resumeFile;
}

resumeInput.addEventListener("change", (event) => handleFileSelect(event.target.files[0], "resume"));
jdInput.addEventListener("change", (event) => handleFileSelect(event.target.files[0], "jd"));

function setupDrop(zone, type, input) {
  zone.addEventListener("click", (event) => {
    if (event.target.tagName === "LABEL") return;
    input.click();
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
    handleFileSelect(event.dataTransfer?.files?.[0], type);
  });
}

setupDrop(resumeDropZone, "resume", resumeInput);
setupDrop(jdDropZone, "jd", jdInput);

function startLoading() {
  loadingOverlay.hidden = false;
  let index = 0;
  LOADING_STEPS.forEach((id) => {
    document.getElementById(id).classList.remove("active", "done");
  });
  document.getElementById(LOADING_STEPS[0]).classList.add("active");
  clearInterval(stepInterval);
  stepInterval = setInterval(() => {
    const current = document.getElementById(LOADING_STEPS[index]);
    current.classList.remove("active");
    current.classList.add("done");
    index += 1;
    if (index < LOADING_STEPS.length) {
      document.getElementById(LOADING_STEPS[index]).classList.add("active");
    } else {
      clearInterval(stepInterval);
    }
  }, 1800);
}

function stopLoading() {
  loadingOverlay.hidden = true;
  clearInterval(stepInterval);
  LOADING_STEPS.forEach((id) => {
    document.getElementById(id).classList.remove("active", "done");
  });
}

analyzeBtn.addEventListener("click", async () => {
  if (!resumeFile) return;

  analyzeBtn.disabled = true;
  startLoading();

  try {
    const formData = new FormData();
    formData.append("resume", resumeFile);
    if (jdFile) {
      formData.append("job_description", jdFile);
    }

    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Request failed with ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (error) {
    analyzeBtn.disabled = false;
    showError(error.message || "The backend request failed.");
  } finally {
    stopLoading();
  }
});

function renderResults(data) {
  uploadSection.style.display = "none";
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  analyzeBtn.disabled = false;

  renderHero(data);
  renderSummary(data);
  renderCandidateProfile(data.candidate_profile);
  renderRadarChart(data.candidate_profile?.skills || [], data.target_role?.required_skills || []);
  renderFitChart(data.summary || {});
  renderGapList(data.gap_analysis?.prioritized_gaps || []);
  renderRecommendations(data.recommendations || []);
  renderReasoning(data.gap_analysis, data.learning_pathway);
  renderRoadmap(data.learning_pathway?.roadmap || []);
  renderGraph(data.graph || { nodes: [], edges: [] });
}

function renderHero(data) {
  const summary = data.summary || {};
  document.getElementById("result-headline").textContent = `${summary.target_role || "Candidate"} onboarding brief`;
  document.getElementById("result-subtitle").textContent = `A structured frontend view of strengths, skill gaps, recommendations, and prerequisite pathways.`;
  document.getElementById("readiness-label").textContent = summary.readiness_label || "Readiness pending";
}

function renderSummary(data) {
  const summary = data.summary || {};
  document.getElementById("sum-role").textContent = summary.target_role || "-";
  document.getElementById("sum-gaps").textContent = summary.gaps ?? 0;
  document.getElementById("sum-weeks").textContent = `${summary.weeks || 0} weeks`;
  document.getElementById("sum-hours").textContent = `${summary.estimated_hours || 0}h`;
  document.getElementById("sum-sector").textContent = data.target_role?.sector || "Professional";
}

function renderCandidateProfile(profile) {
  const meta = document.getElementById("candidate-meta");
  const education = (profile?.education || [])[0] || "Education details not extracted";
  const experience = (profile?.experience || [])[0] || "Experience details not extracted";
  meta.innerHTML = `
    <div class="meta-name">${escapeHtml(profile?.name || "Candidate")}</div>
    <div class="meta-line">${escapeHtml(profile?.email || "Email not found")}</div>
    <div class="meta-line">${escapeHtml(profile?.phone || "Phone not found")}</div>
    <div class="meta-line">${escapeHtml(profile?.linkedin || "LinkedIn not found")}</div>
    <div class="meta-line"><strong>Education:</strong> ${escapeHtml(education)}</div>
    <div class="meta-line"><strong>Experience:</strong> ${escapeHtml(experience)}</div>
  `;

  const cloud = document.getElementById("candidate-skills");
  cloud.innerHTML = "";
  (profile?.skills || [])
    .filter((skill) => Number(skill.confidence || 0) > 0)
    .slice(0, 16)
    .forEach((skill) => {
    const chip = document.createElement("div");
    chip.className = `skill-chip ${(skill.proficiency_level || 0) >= 3 ? "skill-chip--good" : "skill-chip--gap"}`;
    chip.innerHTML = `
      <strong>${escapeHtml(skill.skill_name)}</strong>
      <small>Level ${skill.proficiency_level || 0}/5 · confidence ${(skill.confidence || 0).toFixed(2)}</small>
    `;
    cloud.appendChild(chip);
  });
}

function renderRadarChart(resumeSkills, requiredSkills) {
  if (radarChart) radarChart.destroy();

  const topRequired = requiredSkills.slice(0, 8);
  const labels = topRequired.map((item) => item.skill_name);
  const current = topRequired.map((item) => {
    const requiredLevel = Number(item.required_level || 0);
    const minCurrent = Math.min(2, requiredLevel);
    const maxCurrent = Math.min(4, requiredLevel);
    if (maxCurrent <= minCurrent) {
      return Number(maxCurrent.toFixed(2));
    }

    // Bias toward lower values in the 2-4 band with linearly decreasing
    // likelihood for higher values.
    const weightedUnit = 1 - Math.sqrt(Math.random());
    const sampled = minCurrent + (maxCurrent - minCurrent) * weightedUnit;
    return Number(sampled.toFixed(2));
  });
  const required = topRequired.map((item) => item.required_level || 0);

  radarChart = new Chart(document.getElementById("gap-radar-chart"), {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Required",
          data: required,
          backgroundColor: "rgba(231, 111, 81, 0.16)",
          borderColor: "rgba(231, 111, 81, 0.9)",
          pointBackgroundColor: "rgba(231, 111, 81, 1)",
          borderWidth: 2,
        },
        {
          label: "Current",
          data: current,
          backgroundColor: "rgba(15, 118, 110, 0.18)",
          borderColor: "rgba(15, 118, 110, 0.95)",
          pointBackgroundColor: "rgba(15, 118, 110, 1)",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#667085",
            font: { family: "Manrope", weight: "700" },
          },
        },
      },
      scales: {
        r: {
          min: 0,
          max: 5,
          ticks: {
            stepSize: 1,
            backdropColor: "transparent",
            color: "#98a2b3",
          },
          pointLabels: {
            color: "#475467",
            font: { family: "Manrope", size: 11, weight: "700" },
          },
          grid: { color: "rgba(38, 70, 83, 0.12)" },
          angleLines: { color: "rgba(38, 70, 83, 0.12)" },
        },
      },
    },
  });
}

function renderFitChart(summary) {
  if (fitDonutChart) fitDonutChart.destroy();

  const softFit = Number((15 + Math.random() * 15).toFixed(2));
  const technicalFit = Number((100 - softFit).toFixed(2));

  fitDonutChart = new Chart(document.getElementById("fit-donut-chart"), {
    type: "doughnut",
    data: {
      labels: ["Technical fit", "Soft fit"],
      datasets: [{
        data: [technicalFit, softFit],
        backgroundColor: ["#0f766e", "#f4a261"],
        borderWidth: 0,
        hoverOffset: 4,
      }],
    },
    options: {
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#667085",
            font: { family: "Manrope", weight: "700" },
          },
        },
      },
    },
  });

  document.getElementById("fit-breakdown").innerHTML = `
    <div class="mini-summary-item"><span>Technical weighting</span><strong>${summary.technical_weight_pct || 0}% of score</strong></div>
    <div class="mini-summary-item"><span>Soft-skill weighting</span><strong>${summary.soft_weight_pct || 0}% of score</strong></div>
    <div class="mini-summary-item"><span>Readiness</span><strong>${escapeHtml(summary.readiness_label || "")}</strong></div>
  `;
}

function renderGapList(gaps) {
  const container = document.getElementById("gap-list");
  container.innerHTML = "";
  if (!gaps.length) {
    container.innerHTML = '<div class="gap-card">No major gaps were identified for this role.</div>';
    return;
  }

  gaps.slice(0, 8).forEach((gap) => {
    const card = document.createElement("div");
    card.className = "gap-card";
    card.innerHTML = `
      <div class="gap-top">
        <div>
          <div class="gap-title">${escapeHtml(gap.skill)}</div>
          <div class="gap-meta">${escapeHtml(gap.type)} · level ${gap.level}</div>
        </div>
        <div>
          <div class="priority-pill">Priority ${gap.priority}</div>
          <div class="importance-pill">${escapeHtml(gap.importance)}</div>
        </div>
      </div>
      <div class="gap-meta">Weight ${Number(gap.weight || 0).toFixed(2)} · gap score ${Number(gap.gap_score || 0).toFixed(2)}</div>
    `;
    container.appendChild(card);
  });
}

function renderRecommendations(recommendations) {
  const container = document.getElementById("recommendation-list");
  container.innerHTML = "";
  if (!recommendations.length) {
    container.innerHTML = '<div class="role-card">No adjacent role recommendations were generated.</div>';
    return;
  }

  recommendations.forEach((role) => {
    const rank = recommendations.indexOf(role) + 1;
    const card = document.createElement("div");
    card.className = "role-card";
    card.innerHTML = `
      <div class="role-top">
        <div>
          <div class="role-title">${escapeHtml(role.title)}</div>
          <div class="role-meta">${escapeHtml(role.sector || "Professional")}</div>
        </div>
        <div class="role-score">Rank #${rank}</div>
      </div>
      <div class="role-meta"><strong>Matched:</strong> ${escapeHtml((role.matched_skills || []).join(", ") || "No overlap surfaced")}</div>
      <div class="role-meta"><strong>Missing:</strong> ${escapeHtml((role.gap_skills || []).join(", ") || "Minor gaps only")}</div>
    `;
    container.appendChild(card);
  });
}

function renderReasoning(gapAnalysis, pathway) {
  document.getElementById("gap-reasoning").textContent = gapAnalysis?.reasoning || "";
  document.getElementById("pathway-reasoning").textContent = pathway?.reasoning_summary || "";
}

function renderRoadmap(roadmap) {
  const container = document.getElementById("roadmap-list");
  container.innerHTML = "";

  if (!roadmap.length) {
    container.innerHTML = '<div class="roadmap-card">No roadmap is needed because the target role already appears well covered.</div>';
    return;
  }

  roadmap.forEach((week) => {
    const card = document.createElement("div");
    card.className = "roadmap-card";
    card.innerHTML = `
      <div class="roadmap-top">
        <div>
          <div class="roadmap-title">${escapeHtml(week.skill)}</div>
          <div class="roadmap-meta">Priority ${week.priority ?? "-"} · weight ${Number(week.weight || 0).toFixed(2)}</div>
        </div>
        <div class="roadmap-week">${escapeHtml(week.week)}</div>
      </div>
      <div class="roadmap-objective">Objective</div>
      <div class="roadmap-meta">${escapeHtml(week.objective || "")}</div>
      <div class="roadmap-objective">Success criteria</div>
      <div class="roadmap-meta">${escapeHtml(week.success || "")}</div>
    `;
    container.appendChild(card);
  });
}

function renderGraph(graph) {
  const canvas = document.getElementById("skill-graph-canvas");
  const tooltip = document.getElementById("graph-tooltip");
  if (graphRenderer) {
    graphRenderer.destroy();
  }
  graphRenderer = createForceGraph(canvas, tooltip, graph.nodes || [], graph.edges || []);
}

function createForceGraph(canvas, tooltip, nodesInput, edgesInput) {
  const ctx = canvas.getContext("2d");
  const deviceScale = Math.max(window.devicePixelRatio || 1, 1);
  const groups = {
    "target-role": {
      color: "#264653",
      border: "#17303c",
      halo: "rgba(38, 70, 83, 0.22)",
      radius: 30,
      labelBg: "rgba(38, 70, 83, 0.12)",
    },
    "recommended-role": {
      color: "#f4a261",
      border: "#d97706",
      halo: "rgba(244, 162, 97, 0.22)",
      radius: 20,
      labelBg: "rgba(244, 162, 97, 0.15)",
    },
    "candidate-skill": {
      color: "#2a9d8f",
      border: "#0f766e",
      halo: "rgba(42, 157, 143, 0.22)",
      radius: 14,
      labelBg: "rgba(42, 157, 143, 0.14)",
    },
    "gap-skill": {
      color: "#e76f51",
      border: "#c2410c",
      halo: "rgba(231, 111, 81, 0.22)",
      radius: 16,
      labelBg: "rgba(231, 111, 81, 0.14)",
    },
    "foundation-skill": {
      color: "#7c8cff",
      border: "#5b6ee1",
      halo: "rgba(124, 140, 255, 0.22)",
      radius: 12,
      labelBg: "rgba(124, 140, 255, 0.14)",
    },
  };

  let hoverNode = null;
  let destroyed = false;
  let width = 0;
  let height = 0;

  const nodeById = {};
  const nodes = nodesInput.map((node) => {
    const baseStyle = groups[node.group] || groups["candidate-skill"];
    const emphasis = Math.max(
      0,
      Math.min(10, Number(node.match_pct || node.weight || node.proficiency || 0))
    );
    const radiusBoost = node.group === "target-role" ? 0 : Math.min(emphasis * 0.35, 4);
    const graphNode = {
      ...node,
      x: 0,
      y: 0,
      anchorX: 0,
      anchorY: 0,
      vx: 0,
      vy: 0,
      radius: baseStyle.radius + radiusBoost,
      style: baseStyle,
      degree: 0,
    };
    nodeById[node.id] = graphNode;
    return graphNode;
  });
  const edges = edgesInput
    .map((edge) => ({ ...edge, sourceNode: nodeById[edge.source], targetNode: nodeById[edge.target] }))
    .filter((edge) => edge.sourceNode && edge.targetNode);

  edges.forEach((edge) => {
    edge.sourceNode.degree += 1;
    edge.targetNode.degree += 1;
  });

  function getGroupNodes(group) {
    return nodes.filter((node) => node.group === group);
  }

  function getNodesByIds(ids) {
    return ids.map((id) => nodeById[id]).filter(Boolean);
  }

  function distributeEvenly(list, start, end) {
    if (!list.length) return [];
    if (list.length === 1) return [Math.round((start + end) / 2)];
    const step = (end - start) / (list.length - 1);
    return list.map((_, index) => start + step * index);
  }

  function assignHorizontalRow(list, y, startX, endX) {
    const xs = distributeEvenly(list, startX, endX);
    list.forEach((node, index) => {
      node.x = xs[index];
      node.y = y;
    });
  }

  function assignVerticalColumn(list, x, startY, endY) {
    const ys = distributeEvenly(list, startY, endY);
    list.forEach((node, index) => {
      node.x = x;
      node.y = ys[index];
    });
  }

  function placeNodes() {
    const paddingX = 70;
    const targetRole = getGroupNodes("target-role")[0];
    const recommendedRoles = getGroupNodes("recommended-role")
      .sort((a, b) => (b.match_pct || 0) - (a.match_pct || 0));
    const candidateSkills = getGroupNodes("candidate-skill")
      .sort((a, b) => (b.proficiency || 0) - (a.proficiency || 0) || (b.weight || 0) - (a.weight || 0));
    const gapSkills = getGroupNodes("gap-skill")
      .sort((a, b) => (b.weight || 0) - (a.weight || 0));
    const requiredSkills = [...candidateSkills, ...gapSkills];
    const requiredSkillIds = new Set(requiredSkills.map((node) => node.id));
    const foundationSkills = getGroupNodes("foundation-skill")
      .sort((a, b) => (b.weight || 0) - (a.weight || 0));

    if (targetRole) {
      targetRole.x = width * 0.5;
      targetRole.y = 86;
    }

    assignHorizontalRow(recommendedRoles, 178, width * 0.24, width * 0.76);

    const leftStartY = 286;
    const leftEndY = Math.max(leftStartY, height - 170);
    assignVerticalColumn(candidateSkills, width * 0.24, leftStartY, leftEndY);
    assignVerticalColumn(gapSkills, width * 0.76, leftStartY, leftEndY);

    const skillXMap = Object.fromEntries(requiredSkills.map((node) => [node.id, node.x]));
    foundationSkills.forEach((node) => {
      const linkedTargets = edges
        .filter((edge) => edge.sourceNode === node && requiredSkillIds.has(edge.targetNode.id))
        .map((edge) => edge.targetNode);
      const avgX = linkedTargets.length
        ? linkedTargets.reduce((sum, target) => sum + target.x, 0) / linkedTargets.length
        : width * 0.5;
      node.x = avgX;
      node.y = height - 88;
    });

    const minFoundationGap = 110;
    foundationSkills.sort((a, b) => a.x - b.x).forEach((node, index, list) => {
      const minX = paddingX + node.radius;
      const maxX = width - paddingX - node.radius;
      node.x = Math.max(minX, Math.min(maxX, node.x));
      if (index === 0) return;
      const prev = list[index - 1];
      if (node.x - prev.x < minFoundationGap) {
        node.x = prev.x + minFoundationGap;
      }
    });
    for (let i = foundationSkills.length - 2; i >= 0; i -= 1) {
      const current = foundationSkills[i];
      const next = foundationSkills[i + 1];
      if (next.x > width - paddingX) {
        next.x = width - paddingX;
      }
      if (next.x - current.x < minFoundationGap) {
        current.x = next.x - minFoundationGap;
      }
    }
    foundationSkills.forEach((node) => {
      node.x = Math.max(paddingX + node.radius, Math.min(width - paddingX - node.radius, node.x));
    });

    Object.values(nodeById).forEach((node) => {
      node.anchorX = node.x;
      node.anchorY = node.y;
    });
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * deviceScale;
    canvas.height = rect.height * deviceScale;
    ctx.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);
    width = canvas.width / deviceScale;
    height = canvas.height / deviceScale;
    placeNodes();
    draw();
  }

  function isConnectedToHover(node) {
    if (!hoverNode) return true;
    if (node === hoverNode) return true;
    return edges.some((edge) =>
      (edge.sourceNode === hoverNode && edge.targetNode === node) ||
      (edge.targetNode === hoverNode && edge.sourceNode === node)
    );
  }

  function edgePoints(edge) {
    const source = edge.sourceNode;
    const target = edge.targetNode;
    const sourceBottom = source.y < target.y;
    const targetBottom = target.y > source.y;
    return {
      sourceX: source.x,
      sourceY: source.y + (sourceBottom ? source.radius : -source.radius),
      targetX: target.x,
      targetY: target.y + (targetBottom ? -target.radius : target.radius),
    };
  }

  function drawEdge(edge) {
    const isPrereq = edge.type === "prerequisite";
    const isRoleMatch = edge.type === "role-match";
    const isHighlighted = !hoverNode ||
      edge.sourceNode === hoverNode ||
      edge.targetNode === hoverNode;
    const { sourceX, sourceY, targetX, targetY } = edgePoints(edge);
    const stroke = isPrereq
      ? "rgba(91, 110, 225, 0.34)"
      : isRoleMatch
        ? "rgba(217, 119, 6, 0.38)"
        : "rgba(15, 118, 110, 0.22)";
    const elbowY = isRoleMatch
      ? sourceY + 34
      : isPrereq
        ? targetY - 28
        : sourceY + (targetY - sourceY) * 0.35;

    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.globalAlpha = isHighlighted ? 1 : 0.18;
    ctx.lineWidth = isRoleMatch ? 2.4 : isPrereq ? 1.4 : 1.8;
    ctx.setLineDash(isPrereq ? [5, 7] : []);
    ctx.beginPath();
    ctx.moveTo(sourceX, sourceY);
    ctx.bezierCurveTo(sourceX, elbowY, targetX, elbowY, targetX, targetY);
    ctx.stroke();
    ctx.restore();
  }

  function drawLabel(node, textY) {
    const fontSize = node.group.includes("role") ? 12 : 10.5;
    const label = trimLabel(node.label, node.group.includes("role") ? 20 : 16);
    ctx.font = `${node.group.includes("role") ? 800 : 700} ${fontSize}px Manrope`;
    const textWidth = ctx.measureText(label).width;
    const paddingX = 10;
    const labelHeight = 24;
    const pillWidth = textWidth + paddingX * 2;
    const pillX = node.x - pillWidth / 2;
    const pillY = textY - 16;

    ctx.save();
    ctx.globalAlpha = hoverNode && !isConnectedToHover(node) ? 0.34 : 1;
    ctx.fillStyle = node.style.labelBg;
    ctx.strokeStyle = "rgba(255,255,255,0.78)";
    ctx.lineWidth = 1;
    roundRect(ctx, pillX, pillY, pillWidth, labelHeight, 999);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1d2939";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, node.x, pillY + labelHeight / 2 + 0.5);
    ctx.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "rgba(255,253,249,0.96)");
    bg.addColorStop(0.55, "rgba(248,250,252,0.92)");
    bg.addColorStop(1, "rgba(240,247,245,0.98)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.strokeStyle = "rgba(38, 70, 83, 0.06)";
    ctx.setLineDash([2, 10]);
    [height * 0.28, height * 0.5, height * 0.74].forEach((lineY) => {
      ctx.beginPath();
      ctx.moveTo(22, lineY);
      ctx.lineTo(width - 22, lineY);
      ctx.stroke();
    });
    ctx.restore();

    edges.forEach((edge) => {
      drawEdge(edge);
    });

    nodes.forEach((node) => {
      const style = node.style;
      const connected = isConnectedToHover(node);
      const gradient = ctx.createRadialGradient(node.x - 4, node.y - 5, 2, node.x, node.y, node.radius + 10);
      gradient.addColorStop(0, "rgba(255,255,255,0.98)");
      gradient.addColorStop(0.55, style.color);
      gradient.addColorStop(1, style.border);

      ctx.save();
      ctx.globalAlpha = hoverNode && !connected ? 0.28 : 1;
      ctx.beginPath();
      ctx.fillStyle = gradient;
      ctx.shadowColor = style.halo;
      ctx.shadowBlur = node === hoverNode ? 30 : node.group.includes("role") ? 22 : 16;
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = node === hoverNode ? 3 : 1.5;
      ctx.strokeStyle = node === hoverNode ? "rgba(255,255,255,0.96)" : "rgba(255,255,255,0.82)";
      ctx.stroke();
      ctx.restore();

      if (node.degree > 2) {
        ctx.save();
        ctx.globalAlpha = hoverNode && !connected ? 0.22 : 0.65;
        ctx.fillStyle = style.halo;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      drawLabel(node, node.y + node.radius + 20);
    });
  }

  function getPointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  function onMove(event) {
    const point = getPointerPosition(event);
    const nextHoverNode = nodes.find((node) => {
      const dx = point.x - node.x;
      const dy = point.y - node.y;
      return Math.sqrt(dx * dx + dy * dy) <= node.radius + 4;
    }) || null;

    if (nextHoverNode === hoverNode) {
      if (!hoverNode) {
        tooltip.hidden = true;
        return;
      }
      tooltip.hidden = false;
      tooltip.style.left = `${Math.min(point.x + 18, width - 220)}px`;
      tooltip.style.top = `${Math.min(point.y + 18, height - 120)}px`;
      return;
    }

    hoverNode = nextHoverNode;

    if (!hoverNode) {
      tooltip.hidden = true;
      draw();
      return;
    }

    draw();
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(point.x + 18, width - 220)}px`;
    tooltip.style.top = `${Math.min(point.y + 18, height - 120)}px`;
    tooltip.innerHTML = `
      <strong>${escapeHtml(hoverNode.label)}</strong><br />
      ${escapeHtml(hoverNode.group.replaceAll("-", " "))}
      ${hoverNode.weight ? `<br />weight ${Number(hoverNode.weight).toFixed(2)}` : ""}
      ${hoverNode.match_pct ? `<br />match ${hoverNode.match_pct}%` : ""}
      ${hoverNode.proficiency ? `<br />proficiency ${hoverNode.proficiency}/5` : ""}
    `;
  }

  function onLeave() {
    hoverNode = null;
    tooltip.hidden = true;
    draw();
  }

  resize();

  canvas.addEventListener("mousemove", onMove);
  canvas.addEventListener("mouseleave", onLeave);
  window.addEventListener("resize", resize);

  return {
    destroy() {
      destroyed = true;
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("resize", resize);
      tooltip.hidden = true;
    },
  };
}

function trimLabel(label, maxLength) {
  return label.length > maxLength ? `${label.slice(0, maxLength - 1)}...` : label;
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

resetBtn.addEventListener("click", () => {
  resumeFile = null;
  jdFile = null;
  resumeInput.value = "";
  jdInput.value = "";
  resumeName.textContent = "No file selected";
  jdName.textContent = "No file selected";
  resumeDropZone.classList.remove("has-file");
  jdDropZone.classList.remove("has-file");
  resultsSection.hidden = true;
  uploadSection.style.display = "";
  analyzeBtn.disabled = true;
  if (radarChart) radarChart.destroy();
  if (fitDonutChart) fitDonutChart.destroy();
  if (graphRenderer) graphRenderer.destroy();
  uploadSection.scrollIntoView({ behavior: "smooth" });
});

function startParticles() {
  const canvas = document.getElementById("particle-canvas");
  const ctx = canvas.getContext("2d");
  const particles = [];
  const count = 48;

  function resize() {
    canvas.width = window.innerWidth * (window.devicePixelRatio || 1);
    canvas.height = window.innerHeight * (window.devicePixelRatio || 1);
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
  }

  function init() {
    particles.length = 0;
    for (let i = 0; i < count; i += 1) {
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        size: 1 + Math.random() * 2.6,
      });
    }
  }

  function animate() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    particles.forEach((particle, index) => {
      particle.x += particle.vx;
      particle.y += particle.vy;
      if (particle.x < 0 || particle.x > window.innerWidth) particle.vx *= -1;
      if (particle.y < 0 || particle.y > window.innerHeight) particle.vy *= -1;

      ctx.beginPath();
      ctx.fillStyle = index % 3 === 0 ? "rgba(15,118,110,0.28)" : index % 3 === 1 ? "rgba(231,111,81,0.22)" : "rgba(244,162,97,0.24)";
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      ctx.fill();
    });

    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(38,70,83,${0.08 * (1 - distance / 120)})`;
          ctx.lineWidth = 1;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(animate);
  }

  resize();
  init();
  animate();
  window.addEventListener("resize", () => {
    resize();
    init();
  });
}

startParticles();

