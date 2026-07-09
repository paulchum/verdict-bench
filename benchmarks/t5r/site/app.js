(() => {
  "use strict";

  const data = window.__VERDICT_TRACE_DATA__ || {
    summary: {}, manifest: {}, calls: [], decisions: [], events: [], simulations: []
  };
  const state = { selected: null, tab: "decision", filtered: [] };
  const elements = {
    arm: document.querySelector("#arm-filter"),
    domain: document.querySelector("#domain-filter"),
    task: document.querySelector("#task-filter"),
    search: document.querySelector("#trace-search"),
    clear: document.querySelector("#clear-filters"),
    list: document.querySelector("#decision-list"),
    resultCount: document.querySelector("#result-count"),
    detailTitle: document.querySelector("#detail-title"),
    detailContext: document.querySelector("#detail-context"),
    detail: document.querySelector("#detail-content")
  };

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const number = (value) => new Intl.NumberFormat("en-US").format(Number(value || 0));
  const money = (value) => new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 4
  }).format(Number(value || 0));
  const fixed = (value, digits = 3) => value == null ? "n/a" : Number(value).toFixed(digits);
  const decisionId = (item) => [item.arm, item.domain, item.task_id, item.trial, item.decision_idx].join("|");
  const sameDecision = (item, selected) => item.arm === selected.arm
    && item.domain === selected.domain
    && String(item.task_id) === String(selected.task_id)
    && Number(item.trial || 0) === Number(selected.trial || 0)
    && Number(item.decision_idx) === Number(selected.decision_idx);

  function setup() {
    populateSelect(elements.arm, unique(data.decisions.map((item) => item.arm)));
    populateSelect(elements.domain, unique(data.decisions.map((item) => item.domain)));
    populateSelect(elements.task, unique(data.decisions.map((item) => String(item.task_id))), "Task ");
    [elements.arm, elements.domain, elements.task].forEach((element) => element.addEventListener("change", renderList));
    elements.search.addEventListener("input", renderList);
    elements.clear.addEventListener("click", () => {
      elements.arm.value = "";
      elements.domain.value = "";
      elements.task.value = "";
      elements.search.value = "";
      renderList();
    });
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        state.tab = button.dataset.tab;
        document.querySelectorAll("[data-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
        renderDetail();
      });
    });
    elements.list.addEventListener("keydown", handleListKeys);
    renderSummary();
    renderList();
  }

  function unique(values) {
    return [...new Set(values.filter((value) => value != null))].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
  }

  function populateSelect(select, values, prefix = "") {
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = `${prefix}${value}`;
      select.append(option);
    });
  }

  function renderSummary() {
    const summary = data.summary || {};
    const arms = summary.arms || [];
    const totalCost = arms.reduce((sum, arm) => sum + Number(arm.gross_price_sheet_cost_usd || 0), 0);
    const totalTokens = arms.reduce((sum, arm) => sum + Number(arm.total_tokens || 0), 0);
    document.querySelector("#metric-decisions").textContent = number(data.decisions.length);
    document.querySelector("#metric-cost").textContent = money(totalCost);
    document.querySelector("#metric-tokens").textContent = number(totalTokens);
    document.querySelector("#metric-refusals").textContent = number(summary.t8?.refused_keys || 0);
    document.querySelector("#metric-meter").textContent = summary.meter_reconciliation?.status || "n/a";
    document.querySelector("#run-status").textContent = summary.status || data.manifest?.status || "unknown";
    const synthetic = String(summary.status || "").startsWith("[SIM]");
    document.querySelector("#evidence-banner").textContent = synthetic
      ? "[SIM] Deterministic synthetic evidence for interface and pipeline validation. These are not measured Tau outcomes and must not be cited as benchmark performance."
      : "Pilot evidence only. Inspect individual calls and certificates here; only a separately frozen adjudication can issue K-R1.";
    const source = data.manifest?.source || {};
    document.querySelector("#provenance-line").textContent = source.commit
      ? `Tau ${source.ref || ""} · ${String(source.commit).slice(0, 12)} · patch ${String(source.patch_sha256 || "").slice(0, 12)}`
      : `Artifact schema ${data.schema_version || "unknown"}`;
  }

  function renderList() {
    const query = elements.search.value.trim().toLowerCase();
    state.filtered = data.decisions.filter((item) => {
      const haystack = [item.arm, item.domain, item.task_id, item.outcome, item.selected_component, item.context_key].join(" ").toLowerCase();
      return (!elements.arm.value || item.arm === elements.arm.value)
        && (!elements.domain.value || item.domain === elements.domain.value)
        && (!elements.task.value || String(item.task_id) === elements.task.value)
        && (!query || haystack.includes(query));
    });
    elements.resultCount.textContent = `${number(state.filtered.length)} records`;
    elements.list.innerHTML = state.filtered.length ? state.filtered.map((item) => {
      const id = decisionId(item);
      const selected = state.selected && decisionId(state.selected) === id;
      const outcomeClass = /accept/i.test(item.outcome) ? "accept" : /notseparated|refus|censor|coarse|inhomogeneous/i.test(item.outcome) ? "refuse" : "";
      return `<button class="decision-row" type="button" role="option" data-id="${escapeHtml(id)}" aria-selected="${selected}">
        <span class="decision-primary"><strong>${escapeHtml(item.arm)} · ${escapeHtml(item.domain)}:${escapeHtml(item.task_id)}</strong><span>${escapeHtml(item.selected_component)} · decision ${escapeHtml(item.decision_idx)}</span></span>
        <span class="outcome-chip ${outcomeClass}">${escapeHtml(item.outcome)}</span>
        <span class="decision-secondary"><span>Trial ${escapeHtml(item.trial || 0)} · ${escapeHtml(item.context_key)}</span></span>
      </button>`;
    }).join("") : '<p class="empty-state">No decisions match these filters.</p>';
    elements.list.querySelectorAll("[data-id]").forEach((button) => button.addEventListener("click", () => selectById(button.dataset.id)));
    if (!state.selected || !state.filtered.some((item) => decisionId(item) === decisionId(state.selected))) {
      state.selected = state.filtered[0] || null;
      renderListSelection();
    }
    renderDetail();
  }

  function selectById(id) {
    state.selected = state.filtered.find((item) => decisionId(item) === id) || null;
    renderListSelection();
    renderDetail();
  }

  function renderListSelection() {
    elements.list.querySelectorAll("[data-id]").forEach((button) => {
      button.setAttribute("aria-selected", String(state.selected && button.dataset.id === decisionId(state.selected)));
    });
  }

  function handleListKeys(event) {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) || !state.filtered.length) return;
    event.preventDefault();
    const current = Math.max(0, state.filtered.findIndex((item) => state.selected && decisionId(item) === decisionId(state.selected)));
    const next = event.key === "Home" ? 0 : event.key === "End" ? state.filtered.length - 1
      : event.key === "ArrowDown" ? Math.min(current + 1, state.filtered.length - 1) : Math.max(current - 1, 0);
    state.selected = state.filtered[next];
    renderListSelection();
    renderDetail();
    elements.list.querySelector(`[data-id="${CSS.escape(decisionId(state.selected))}"]`)?.scrollIntoView({ block: "nearest" });
  }

  function renderDetail() {
    const selected = state.selected;
    if (!selected) {
      elements.detailContext.textContent = "Select a trace";
      elements.detailTitle.textContent = "Decision evidence";
      elements.detail.innerHTML = '<p class="empty-state">Choose a decision to inspect generation, scoring, spend, and certificate evidence.</p>';
      return;
    }
    elements.detailContext.textContent = `${selected.arm} · ${selected.domain}:${selected.task_id} · trial ${selected.trial || 0}`;
    elements.detailTitle.textContent = `${selected.outcome} at decision ${selected.decision_idx}`;
    if (state.tab === "calls") renderCalls(selected);
    else if (state.tab === "certificate") renderCertificate(selected);
    else renderDecision(selected);
  }

  function selectedCalls(selected) {
    return data.calls.filter((item) => sameDecision(item, selected));
  }

  function selectedEvents(selected) {
    return data.events.filter((item) => sameDecision(item, selected));
  }

  function taskResult(selected) {
    return data.simulations.find((item) => item.arm === selected.arm && item.domain === selected.domain
      && String(item.task_id) === String(selected.task_id) && Number(item.trial || 0) === Number(selected.trial || 0));
  }

  function renderDecision(selected) {
    const result = taskResult(selected);
    const candidates = Object.entries(selected.candidate_scores || {});
    const calls = selectedCalls(selected);
    const spend = calls.reduce((sum, call) => sum + Number(call.billed_usd || 0), 0);
    elements.detail.innerHTML = `<div class="detail-grid">
      ${fact("Selected", selected.selected_component)}
      ${fact("Incumbent", selected.incumbent_component)}
      ${fact("Decision spend", money(spend))}
      ${fact("Task reward", result?.reward_info?.reward ?? "n/a")}
    </div>
    <section class="evidence-section"><h3>Candidate outcomes</h3>
      <div class="table-wrap"><table><thead><tr><th>Component</th><th>Proxy score</th><th>Audit outcome</th><th>Disposition</th></tr></thead>
      <tbody>${candidates.length ? candidates.map(([component, score]) => `<tr><td>${escapeHtml(component)}</td><td class="numeric">${fixed(score)}</td><td class="numeric">${escapeHtml(selected.bernoulli_outcomes?.[component] ?? "n/a")}</td><td>${component === selected.selected_component ? "selected" : "not selected"}</td></tr>`).join("") : `<tr><td>${escapeHtml(selected.selected_component)}</td><td class="numeric">n/a</td><td class="numeric">n/a</td><td>committed directly</td></tr>`}</tbody></table></div>
    </section>
    <section class="evidence-section"><h3>Context key</h3><p class="mono">${escapeHtml(selected.context_key)}</p></section>`;
  }

  function renderCalls(selected) {
    const calls = selectedCalls(selected);
    elements.detail.innerHTML = calls.length ? `<div class="timeline">${calls.map((call) => `<div class="timeline-item">
      <strong>${escapeHtml(call.purpose)}</strong>
      <span>${escapeHtml(call.component || "user simulator")}<br><span class="mono">${escapeHtml(call.resolved_model || call.requested_model)}</span></span>
      <span>${number(call.total_tokens)} tokens<br>${money(call.billed_usd)}</span>
    </div>`).join("")}</div>
    <section class="evidence-section"><h3>Physical call ledger</h3><div class="table-wrap"><table><thead><tr><th>Request</th><th>Status</th><th>Input</th><th>Cached</th><th>Output</th><th>Latency</th></tr></thead><tbody>
      ${calls.map((call) => `<tr><td class="mono">${escapeHtml(call.request_id || call.physical_call_id)}</td><td>${escapeHtml(call.status)}</td><td class="numeric">${number(call.input_tokens)}</td><td class="numeric">${number(call.cached_input_tokens)}</td><td class="numeric">${number(call.output_tokens)}</td><td class="numeric">${fixed(call.latency_seconds, 2)}s</td></tr>`).join("")}
    </tbody></table></div></section>` : '<p class="empty-state">No provider calls were recorded for this decision.</p>';
  }

  function renderCertificate(selected) {
    const events = selectedEvents(selected);
    const threshold = selected.threshold;
    elements.detail.innerHTML = `<div class="detail-grid">
      ${fact("Accept log e", fixed(selected.accept_log_e))}
      ${fact("Refute log e", fixed(selected.refute_log_e))}
      ${fact("Threshold", fixed(threshold))}
      ${fact("Outcome", selected.outcome)}
    </div>
    <section class="evidence-section"><h3>Certificate and refusal events</h3>
      ${events.length ? `<div class="table-wrap"><table><thead><tr><th>Event</th><th>Component</th><th>Detail</th><th>log e / threshold</th></tr></thead><tbody>${events.map((event) => `<tr><td>${escapeHtml(event.event)}</td><td>${escapeHtml(event.component || "n/a")}</td><td>${escapeHtml(event.detail)}</td><td class="numeric">${fixed(event.log_e)} / ${fixed(event.threshold)}</td></tr>`).join("")}</tbody></table></div>` : '<p class="empty-state">No certificate events were emitted for this decision.</p>'}
    </section>`;
  }

  function fact(label, value) {
    return `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
  }

  setup();
})();
