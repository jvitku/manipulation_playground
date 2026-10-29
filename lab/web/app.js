/* Lab frontend: hash routing (#/ ladder, #/task/<id>), run controls, polling. Plain ES, no build step. */
(function () {
  "use strict";
  const TERMINAL = ["done", "failed", "stopped"];
  const isTerminal = (s) => TERMINAL.includes(s);
  const appEl = document.getElementById("app");

  // ---------- helpers ----------
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids) if (kid != null) e.append(kid);
    return e;
  }

  async function api(method, url, body) {
    const opt = { method, headers: {} };
    if (body !== undefined) {
      opt.headers["Content-Type"] = "application/json";
      opt.body = JSON.stringify(body);
    }
    const r = await fetch(url, opt);
    let data = null;
    if (r.status !== 204) {
      try { data = await r.json(); } catch (e) { data = null; }
    }
    if (!r.ok) {
      const d = data && data.detail;
      const msg = typeof d === "string" ? d : d ? JSON.stringify(d) : `${r.status} ${r.statusText}`;
      const err = new Error(msg);
      err.status = r.status;
      throw err;
    }
    return data;
  }

  function fmtWall(s) {
    s = Math.max(0, Math.round(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    const mm = String(m).padStart(2, "0"), ss = String(sec).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }
  const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US").replace(/,/g, " "));

  function estClass(seconds) {
    return seconds < 45 * 60 ? "good" : seconds < 90 * 60 ? "warn" : "bad";
  }

  // ---------- param forms ----------
  async function fillCkptSelect(select, taskId) {
    let cks = [];
    try { cks = ((await api("GET", `/api/tasks/${encodeURIComponent(taskId)}`)).checkpoints) || []; } catch (e) { /* empty list */ }
    select.replaceChildren();
    for (const c of cks) {
      const m = c.metrics || {};
      const key = Object.keys(m).find((k) => typeof m[k] === "number");
      const head = key ? ` · ${key} ${Number(m[key]).toPrecision(3)}` : "";
      select.append(el("option", { value: c.id, text: `step ${fmtInt(c.step)}${c.label ? " · " + c.label : ""}${head} · run ${c.run_id}${c.created ? " · " + c.created : ""}` }));
    }
    if (!select.options.length) select.append(el("option", { value: "", text: `no ${taskId} checkpoints yet — train one first` }));
  }

  function buildForm(params, container) {
    container.replaceChildren();
    const inputs = {};
    for (const p of params) {
      let input;
      if (p.kind === "choice") {
        input = el("select");
        (p.choices || []).forEach((c, i) => {
          input.append(el("option", { value: String(i), text: String(c) }));
        });
        const di = (p.choices || []).findIndex((c) => c === p.default);
        input.value = String(di < 0 ? 0 : di);
      } else if (p.kind === "bool") {
        input = el("input", { type: "checkbox" });
        input.checked = !!p.default;
      } else if (p.kind === "ckpt") {
        // a checkpoint of another task (p.ckpt_task): options come from that task's detail endpoint
        input = el("select");
        input.append(el("option", { value: "", text: "loading checkpoints…" }));
        fillCkptSelect(input, p.ckpt_task);
      } else {
        input = el("input", { type: "number", step: p.step != null ? p.step : p.kind === "int" ? 1 : "any" });
        if (p.min != null) input.min = p.min;
        if (p.max != null) input.max = p.max;
        input.value = p.default;
      }
      const wrap = el("div", { class: "param" + (p.kind === "bool" ? "" : "") }, el("label", { text: p.label || p.name }), input);
      if (p.help) wrap.title = p.help;
      if (p.long) wrap.append(el("div", { class: "hint", text: "makes runs long" }));
      container.append(wrap);
      inputs[p.name] = { p, input };
    }
    return {
      values() {
        const out = {};
        for (const { p, input } of Object.values(inputs)) {
          if (p.kind === "choice") out[p.name] = p.choices[Number(input.value)];
          else if (p.kind === "bool") out[p.name] = input.checked;
          else if (p.kind === "ckpt") out[p.name] = input.value;
          else if (input.value === "") out[p.name] = p.default;
          else out[p.name] = p.kind === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
        }
        return out;
      },
      set(values) {
        for (const { p, input } of Object.values(inputs)) {
          const v = values && p.name in values ? values[p.name] : p.default;
          if (p.kind === "choice") {
            const i = p.choices.findIndex((c) => c === v);
            input.value = String(i < 0 ? 0 : i);
          } else if (p.kind === "bool") input.checked = !!v;
          else if (p.kind === "ckpt") {
            // a started run stores the resolved file path; match the option whose id ends it
            const o = Array.from(input.options).find((x) => x.value && typeof v === "string" && (v === x.value || v.endsWith("/runs/" + p.ckpt_task + "/" + x.value.replace("/", "/ckpt/"))));
            if (o) input.value = o.value;
          } else input.value = v;
        }
      },
      reset() { this.set(null); },
      onChange(fn) {
        for (const { input } of Object.values(inputs)) {
          input.addEventListener("input", fn);
          input.addEventListener("change", fn);
        }
      },
    };
  }

  // ---------- global: GPU badge ----------
  async function pollGpu() {
    try {
      const g = await api("GET", "/api/gpu");
      const box = document.getElementById("gpu");
      if (!g.total_mb) box.replaceChildren("GPU ", el("b", { text: "n/a" }));
      else box.replaceChildren("GPU ", el("b", { text: (g.used_mb / 1024).toFixed(1) }), ` / ${(g.total_mb / 1024).toFixed(1)} GB`);
    } catch (e) { /* keep last value */ }
  }
  pollGpu();
  setInterval(pollGpu, 10000);

  // ---------- routing ----------
  let teardown = null; // stops pollers of the current page
  function route() {
    if (teardown) { teardown(); teardown = null; }
    const m = location.hash.match(/^#\/task\/([^/?]+)/);
    if (m) teardown = taskPage(decodeURIComponent(m[1]));
    else teardown = ladderPage();
  }
  window.addEventListener("hashchange", route);

  function taskCard(t, active) {
    const a = el("a", { class: "tcard" + (active ? " active" : ""), href: `#/task/${encodeURIComponent(t.id)}` },
      el("span", { class: "n", text: t.order }),
      el("span", { class: "t", text: t.title }),
      el("span", { class: "min", text: t.est_label || "", title: "nominal time at the default parameters" }),
      el("span", { class: "s", text: t.subtitle || "" }));
    if (t.running) a.append(el("span", { class: "run", text: "running" }));
    else if (t.last_state) a.append(el("span", { class: "last " + t.last_state, text: "last: " + t.last_state }));
    return a;
  }

  // ---------- ladder page ----------
  function ladderPage() {
    let alive = true, timer = null;
    appEl.replaceChildren(el("div", { class: "ladder full", id: "ladder" }, el("p", { class: "note", text: "loading…" })));
    async function load() {
      try {
        const tasks = await api("GET", "/api/tasks");
        if (!alive) return;
        tasks.sort((a, b) => a.order - b.order);
        document.getElementById("ladder").replaceChildren(...tasks.map((t) => taskCard(t, false)));
      } catch (e) {
        if (alive) document.getElementById("ladder").replaceChildren(el("div", { class: "err", text: `Cannot load tasks: ${e.message}` }));
      }
      if (alive) timer = setTimeout(load, 5000);
    }
    load();
    return () => { alive = false; clearTimeout(timer); };
  }

  // ---------- task page ----------
  function taskPage(taskId) {
    let alive = true;
    const timers = new Set();
    const later = (fn, ms) => { const t = setTimeout(() => { timers.delete(t); if (alive) fn(); }, ms); timers.add(t); return t; };

    // run attach state
    let runToken = 0;       // bumps on every attach/detach; stale polls compare
    let currentRun = null;  // run_id
    let currentState = null;
    let detail = null;
    let form = null, evalForm = null, charts = null;

    const nav = el("nav", { class: "ladder" });
    const main = el("main");
    appEl.replaceChildren(el("div", { class: "layout" }, nav, main));
    main.append(el("p", { class: "note", text: "loading…" }));

    async function loadNav() {
      try {
        const tasks = await api("GET", "/api/tasks");
        if (!alive) return;
        tasks.sort((a, b) => a.order - b.order);
        nav.replaceChildren(...tasks.map((t) => taskCard(t, t.id === taskId)));
        const mine = tasks.find((t) => t.id === taskId), h = main.querySelector(".lesson h2");
        if (mine && h) h.replaceChildren(`Task ${mine.order} `, el("span", { class: "k", text: "· " + mine.title }));
      } catch (e) { /* nav is a convenience */ }
      if (alive) later(loadNav, 5000);
    }
    loadNav();

    (async function init() {
      try {
        detail = await api("GET", `/api/tasks/${encodeURIComponent(taskId)}`);
      } catch (e) {
        main.replaceChildren(el("div", { class: "err", text: `Cannot load task ${taskId}: ${e.message}` }));
        return;
      }
      if (alive) build();
    })();

    function build() {
      const spec = detail.spec;
      main.replaceChildren();

      // --- per-task vector-grid toggle (re-renders the last result without refetching) ---
      const vgBox = el("input", { type: "checkbox" });
      vgBox.checked = window.LabVecGrid.enabled(taskId);
      vgBox.addEventListener("change", () => {
        window.LabVecGrid.setEnabled(taskId, vgBox.checked);
        if (liveRow) { liveEl.replaceChildren(); liveEl._live = null; drawLive(); }
        if (lastShown) showResult(lastShown.result, lastShown.evalId);
      });
      if (spec.vecgrids) main.append(el("div", { class: "ctl" },
        el("label", { class: "vgtoggle", title: "show vector values as colour grids in the result viewer (turn off if rendering is slow)" },
          vgBox, "vector grids")));
      let lastShown = null;

      // --- lesson card ---
      const lessonBody = el("div", { class: "lesson-body" });
      const nav_t = nav.querySelector(".active .t");
      main.append(el("section", { class: "card lesson" }, el("h2", { text: nav_t ? nav_t.textContent : taskId }), lessonBody));
      loadLesson(spec.lesson_url, lessonBody);

      // --- training card ---
      const paramsEl = el("div", { class: "params" });
      const startBtn = el("button", { class: "primary", text: "Start", onclick: onStart });
      const pauseBtn = el("button", { text: "Pause", onclick: onPause });
      const stopBtn = el("button", { text: "Stop", onclick: onStop });
      const resetBtn = el("button", { text: "Reset", onclick: onReset });
      const estEl = el("span", { class: "est", text: "estimate …" });
      const runSel = el("select", { class: "runsel", title: "past runs", onchange: () => attachRun(runSel.value || null) });
      const errEl = el("div", { class: "err" });
      const statusEl = el("div", { class: "status" });
      const chartsEl = el("div", { class: "charts" });
      const liveEl = el("div", { class: "live-panel", hidden: true }); // spec.live renderer container
      main.append(el("section", { class: "card" },
        el("h2", {}, "Training ", el("span", { class: "k", text: "· run, watch, perturb" })),
        paramsEl,
        el("div", { class: "ctl" }, startBtn, pauseBtn, stopBtn, resetBtn, runSel, estEl),
        errEl, statusEl, chartsEl, liveEl));

      // live panel: the newest metrics row with an `attn` key goes to the spec.live viewer, which
      // updates its own DOM in place (cached on liveEl._live, tagged with the viewer name); the last snapshot stays after the run ends.
      let liveRow = null;
      function liveCtx() {
        return { task: taskId, taskId, evalId: null, vecgrids: !!spec.vecgrids && window.LabVecGrid.enabled(taskId) };
      }
      function drawLive() {
        const fn = spec.live && window.LabViewers.get(spec.live);
        if (!fn || !liveRow) return;
        liveEl.hidden = false;
        try { fn(liveRow, liveEl, liveCtx()); } catch (e) { liveEl._live = null; window.LabViewers.jsonFallback(liveRow, liveEl); }
      }
      function pushLive(rows) {
        if (!spec.live) return;
        for (let i = rows.length - 1; i >= 0; i--) {
          if (rows[i] && rows[i].attn !== undefined) { liveRow = rows[i]; drawLive(); return; }
        }
      }
      function resetLive() { liveRow = null; liveEl.hidden = true; liveEl.replaceChildren(); liveEl._live = null; }

      form = buildForm(spec.params, paramsEl);
      charts = window.makeCharts(chartsEl, spec.charts);

      // estimate (debounced)
      let estTimer = null, estSeq = 0;
      function scheduleEstimate() {
        clearTimeout(estTimer);
        estTimer = later(async () => {
          const seq = ++estSeq;
          try {
            const r = await api("POST", `/api/tasks/${encodeURIComponent(taskId)}/estimate`, { params: form.values() });
            if (!alive || seq !== estSeq) return;
            estEl.className = `est ${estClass(r.seconds)}`;
            estEl.textContent = `estimate ${r.label}`;
          } catch (e) {
            if (!alive || seq !== estSeq) return;
            estEl.className = "est";
            estEl.textContent = e.status === 422 ? `invalid: ${e.message}` : "estimate unavailable";
          }
        }, 400);
      }
      form.onChange(scheduleEstimate);
      scheduleEstimate();

      function renderRunSel() {
        runSel.replaceChildren(el("option", { value: "", text: "— runs —" }));
        for (const r of detail.runs) {
          const step = r.step != null ? ` · ${fmtInt(r.step)}${r.total_steps ? "/" + fmtInt(r.total_steps) : ""}` : "";
          runSel.append(el("option", { value: r.run_id, text: `${r.run_id} · ${r.state}${step}` }));
        }
        runSel.value = currentRun || "";
      }

      function renderButtons() {
        const active = currentRun && !isTerminal(currentState);
        startBtn.disabled = false;
        pauseBtn.disabled = !active;
        stopBtn.disabled = !active;
        pauseBtn.textContent = currentState === "paused" ? "Resume" : "Pause";
      }

      function renderStatus(st) {
        const state = st.state || "?";
        const kids = [el("span", {}, el("span", { class: `dot ${state}` }), el("b", { text: state }))];
        if (st.step != null) kids.push(el("span", {}, "step ", el("b", { text: fmtInt(st.step) }), st.total_steps ? ` / ${fmtInt(st.total_steps)}` : ""));
        if (st.wall_s != null) kids.push(el("span", {}, "wall ", el("b", { text: fmtWall(st.wall_s) })));
        if (currentRun) kids.push(el("span", {}, "run ", el("b", { text: currentRun })));
        if (st.msg || st.message) kids.push(el("span", { text: st.msg || st.message }));
        statusEl.replaceChildren(...kids);
      }

      async function refreshDetail() {
        try {
          detail = await api("GET", `/api/tasks/${encodeURIComponent(taskId)}`);
          if (!alive) return;
          renderRunSel();
          renderCkpts();
        } catch (e) { /* ignore */ }
      }

      // attach charts to a run: backfill with since=0, then live-poll only while non-terminal
      async function attachRun(runId) {
        const token = ++runToken;
        currentRun = runId || null;
        currentState = null;
        charts.reset();
        resetLive();
        statusEl.replaceChildren();
        renderButtons();
        runSel.value = currentRun || "";
        if (!runId) return;
        let next = 0;
        async function poll() {
          if (!alive || token !== runToken) return;
          let r;
          try {
            r = await api("GET", `/api/runs/${encodeURIComponent(taskId)}/${encodeURIComponent(runId)}?since=${next}`);
          } catch (e) {
            if (alive && token === runToken) {
              errEl.textContent = `Cannot poll run: ${e.message}`;
              later(poll, 3000);
            }
            return;
          }
          if (!alive || token !== runToken) return;
          errEl.textContent = "";
          next = r.next;
          charts.push(r.metrics);
          pushLive(r.metrics || []);
          const st = r.status || {};
          const was = currentState;
          currentState = st.state;
          renderStatus(st);
          renderButtons();
          if (isTerminal(st.state)) {
            if (!was || !isTerminal(was)) refreshDetail();
            return;
          }
          later(poll, 1000);
        }
        poll();
      }

      async function onStart() {
        errEl.textContent = "";
        startBtn.disabled = true;
        try {
          const r = await api("POST", `/api/tasks/${encodeURIComponent(taskId)}/runs`, { params: form.values() });
          await refreshDetail();
          attachRun(r.run_id);
        } catch (e) {
          errEl.textContent = e.message;
        }
        startBtn.disabled = false;
      }
      async function control(cmd) {
        if (!currentRun) return;
        try {
          await api("POST", `/api/runs/${encodeURIComponent(taskId)}/${encodeURIComponent(currentRun)}/control`, { cmd });
          errEl.textContent = "";
        } catch (e) { errEl.textContent = e.message; }
      }
      function onPause() { control(currentState === "paused" ? "run" : "pause"); }
      function onStop() { control("stop"); }
      function onReset() {
        form.reset();
        errEl.textContent = "";
        attachRun(null);
        scheduleEstimate();
      }

      renderRunSel();
      renderButtons();
      buildEvalCard();
      renderCkpts();

      // initial attach: newest run, if any
      if (detail.runs.length) attachRun(detail.runs[0].run_id);

      // ---------- evaluation card ----------
      var ckptSel, evalErr, evalStatus, outcomeEl, viewerEl, evalBtn;
      function buildEvalCard() {
        const evalParamsEl = el("div", { class: "params" });
        ckptSel = el("select");
        const ckptWrap = el("div", { class: "param wide" }, el("label", { text: "checkpoint" }), ckptSel);
        evalBtn = el("button", { class: "primary", text: "Run evaluation", onclick: onEval });
        evalErr = el("div", { class: "err" });
        evalStatus = el("div", { class: "status" });
        outcomeEl = el("div");
        viewerEl = el("div", { class: "chart" });
        viewerEl.hidden = true;
        main.append(el("section", { class: "card" },
          el("h2", {}, "Evaluation ", el("span", { class: "k", text: "· load a checkpoint, change the world, watch it act" })),
          el("div", { class: "params" }, ckptWrap),
          evalParamsEl,
          el("div", { class: "ctl" }, evalBtn),
          evalErr, evalStatus,
          el("div", { class: "evalrow" }, outcomeEl, viewerEl)));
        evalForm = buildForm(spec.eval_params, evalParamsEl);
      }

      function renderCkpts() {
        if (!ckptSel) return;
        const prev = ckptSel.value;
        ckptSel.replaceChildren();
        for (const c of detail.checkpoints || []) {
          if (c.reference) { ckptSel.append(el("option", { value: c.id, text: c.label })); continue; }
          const m = c.metrics || {};
          const key = Object.keys(m).find((k) => typeof m[k] === "number");
          const head = key ? ` · ${key} ${Number(m[key]).toPrecision(3)}` : "";
          ckptSel.append(el("option", { value: c.id, text: `step ${fmtInt(c.step)}${c.label ? " · " + c.label : ""}${head} · run ${c.run_id}${c.created ? " · " + c.created : ""}` }));
        }
        if (!ckptSel.options.length) ckptSel.append(el("option", { value: "", text: "no checkpoints yet — train first" }));
        if (prev) ckptSel.value = prev;
        evalBtn.disabled = !ckptSel.value;
      }

      let evalToken = 0;
      async function onEval() {
        evalErr.textContent = "";
        if (!ckptSel.value) return;
        const token = ++evalToken;
        evalBtn.disabled = true;
        outcomeEl.replaceChildren();
        viewerEl.hidden = true;
        let evalId;
        try {
          const r = await api("POST", `/api/tasks/${encodeURIComponent(taskId)}/evals`, { ckpt: ckptSel.value, params: evalForm.values() });
          evalId = r.eval_id;
        } catch (e) {
          evalErr.textContent = e.message;
          evalBtn.disabled = !ckptSel.value;
          return;
        }
        async function poll() {
          if (!alive || token !== evalToken) return;
          let r;
          try {
            r = await api("GET", `/api/evals/${encodeURIComponent(taskId)}/${encodeURIComponent(evalId)}`);
          } catch (e) {
            evalErr.textContent = `Cannot poll evaluation: ${e.message}`;
            evalBtn.disabled = false;
            return;
          }
          if (!alive || token !== evalToken) return;
          const st = r.status || {};
          // build the list conditionally: native replaceChildren stringifies a null argument
          // into the literal text "null" (unlike el(), which skips null kids)
          const kids = [el("span", {}, el("span", { class: `dot ${st.state}` }), el("b", { text: st.state || "?" }))];
          if (st.wall_s != null) kids.push(el("span", {}, "wall ", el("b", { text: fmtWall(st.wall_s) })));
          if (st.msg || st.message) kids.push(el("span", { text: st.msg || st.message }));
          evalStatus.replaceChildren(...kids);
          if (!isTerminal(st.state)) { later(poll, 1000); return; }
          evalBtn.disabled = false;
          if (st.state === "done" && r.result) showResult(r.result, evalId);
          else if (st.state !== "done") evalErr.textContent = `Evaluation ${st.state}${st.msg ? ": " + st.msg : ""}`;
        }
        poll();
      }

      function showResult(result, evalId) {
        lastShown = { result, evalId };
        outcomeEl.replaceChildren(outcomeTable(result));
        viewerEl.hidden = false;
        viewerEl.replaceChildren(el("h3", { text: "result" }), el("div", { class: "viewer-body" }));
        const ctx = {
          task: taskId,
          evalId,
          vecgrids: !!spec.vecgrids && window.LabVecGrid.enabled(taskId),
          fileUrl: (name) => `/api/evals/${encodeURIComponent(taskId)}/${encodeURIComponent(evalId)}/file/${encodeURIComponent(name)}`,
        };
        window.LabViewers.render(spec.viewer, result, viewerEl.querySelector(".viewer-body"), ctx);
      }
    }

    return () => {
      alive = false;
      runToken++;
      timers.forEach(clearTimeout);
      timers.clear();
    };
  }

  // Outcome table. Two result shapes:
  //  - sectioned (tasks 1-2): top-level objects (t1/t2) of scalar fractions, one row per number;
  //    a "wilson95" [lo, hi] on the section is the CI of its "success" rate.
  //  - flat (tasks 3-9): top-level {n_episodes, success, wilson95, counts: {outcome: k}, ...} (plus
  //    base_success for the residual task): headline success % with its CI, then each count as
  //    "k of n" (counts are integers, never percentages). Other top-level numbers are settings.
  function outcomeTable(result) {
    const rows = [];
    const pct = (v) => (v >= 0 && v <= 1 ? `${(v * 100).toFixed(1)} %` : Number(v).toPrecision(3));
    const ci = (w) => (Array.isArray(w) && w.length === 2 ? `${(w[0] * 100).toFixed(0)}–${(w[1] * 100).toFixed(0)}` : "—");
    const bad = (k) => /collision|error|abort|wrong|timeout/.test(k);
    const row = (label, text, cls, c) =>
      rows.push(el("tr", {}, el("td", { text: label }), el("td", {}, el("span", { class: cls, text })), el("td", { text: c })));
    const r = result || {};
    const n = typeof r.n_episodes === "number" ? r.n_episodes : null;
    if (typeof r.success === "number") {
      row(n != null ? `success (${n} episodes)` : "success", pct(r.success), "chip", ci(r.wilson95));
      if (r.counts && typeof r.counts === "object") {
        for (const [k, v] of Object.entries(r.counts)) {
          if (k === "success" || typeof v !== "number") continue;
          row(k, n != null ? `${v} of ${n}` : String(v), bad(k) ? "chip bad" : "chip", "—");
        }
      }
      if (typeof r.base_success === "number") row("base policy success", pct(r.base_success), "chip", "—");
    }
    for (const [sec, obj] of Object.entries(r)) {
      if (!obj || typeof obj !== "object" || Array.isArray(obj) || sec === "counts") continue;
      for (const [k, v] of Object.entries(obj)) {
        if (typeof v !== "number") continue;
        row(`${sec} ${k}`, pct(v), bad(k) ? "chip bad" : "chip", k === "success" ? ci(obj.wilson95) : "—");
      }
    }
    if (!rows.length) return el("p", { class: "note", text: "No scalar outcome metrics in this result." });
    return el("table", { class: "out" }, el("tr", {}, el("th"), el("th", { text: "value" }), el("th", { text: "95 % CI" })), ...rows);
  }

  // Typeset $$..$$ / \(..\) inside the lesson only. KaTeX is `defer`red, so it may not have run yet:
  // retry once on window load. If it never loads, the raw LaTeX stays visible.
  function typesetMath(container) {
    const run = () => {
      try {
        if (window.renderMathInElement)
          window.renderMathInElement(container, {
            delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }],
            throwOnError: false,
          });
      } catch (e) { /* leave the raw LaTeX visible */ }
    };
    if (window.renderMathInElement) run();
    else window.addEventListener("load", run, { once: true });
  }

  // Syntax-highlight the lesson's code snippets (once per element: highlightElement marks `data-highlighted`).
  function highlightSnippets(container) {
    if (!window.hljs) return;
    container.querySelectorAll("pre code").forEach((c) => {
      if (c.dataset.highlighted) return;
      try { window.hljs.highlightElement(c); } catch (e) { /* leave the plain text */ }
    });
  }

  // "The whole code": one collapsible after the last snippet holding every source file the
  // lesson's snippets reference (code[data-src]), fetched lazily from /api/source.
  function appendWholeCode(container) {
    const srcs = [...new Set([...container.querySelectorAll("code[data-src]")].map((c) => c.dataset.src))];
    if (!srcs.length) return;
    const body = el("div");
    const d = el("details", { class: "code allcode" },
      el("summary", { text: `The whole code \u2014 ${srcs.length} file${srcs.length > 1 ? "s" : ""}` }), body);
    d.addEventListener("toggle", async () => {
      if (!d.open || d.dataset.loaded) return;
      d.dataset.loaded = "1";
      for (const path of srcs) {
        const code = el("code", { class: "language-python" });
        body.append(el("h4", { class: "file", text: path }), el("pre", {}, code));
        try {
          const r = await fetch("/api/source/" + path);
          if (!r.ok) throw new Error(String(r.status));
          code.textContent = await r.text();
          if (window.hljs) window.hljs.highlightElement(code);
        } catch (e) {
          code.textContent = `could not load ${path} (${e.message})`;
        }
      }
    });
    const snips = container.querySelectorAll("details.code");
    const last = snips[snips.length - 1];
    if (last) last.after(d); else container.append(d);
  }

  // Schematic block details: an svg element with data-i="key" gets a small "i" badge in its
  // top-right corner; the key looks up {t, html} in the lesson's <script class="blkinfo"> JSON.
  function wireBlockInfo(container) {
    const info = {};
    container.querySelectorAll("script.blkinfo").forEach((sc) => {
      try { Object.assign(info, JSON.parse(sc.textContent)); } catch (e) { /* bad JSON: no badges */ }
    });
    container.querySelectorAll("svg [data-i]").forEach((node) => {
      const d = info[node.dataset.i];
      const svg = node.ownerSVGElement;
      if (!d || !svg) return;
      let b;
      try { b = node.getBBox(); } catch (e) { return; }
      if (!b || !b.width) return;
      const NS = "http://www.w3.org/2000/svg";
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "blk-i");
      g.setAttribute("transform", `translate(${b.x + b.width - 3},${b.y + 3})`);
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("r", "7");
      const t = document.createElementNS(NS, "text");
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("y", "3.2");
      t.textContent = "i";
      g.append(c, t);
      g.addEventListener("click", (ev) => { ev.stopPropagation(); showBlockInfo(d); });
      svg.append(g);
    });
  }

  function showBlockInfo(d) {
    const body = el("div", { class: "body" });
    body.innerHTML = d.html || ""; // trusted: ships inside our own lesson files
    const pop = el("div", { class: "blk-pop" },
      el("button", { class: "x", type: "button", text: "\u00d7", onclick: () => ov.remove() }),
      el("h4", { text: d.t || "details" }), body);
    const ov = el("div", { class: "blk-pop-ov", onclick: (e) => { if (e.target === ov) ov.remove(); } }, pop);
    const esc = (e) => { if (e.key === "Escape") { ov.remove(); document.removeEventListener("keydown", esc); } };
    document.addEventListener("keydown", esc);
    document.body.append(ov);
    typesetMath(body);
  }

  async function loadLesson(url, container) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(String(r.status));
      const text = await r.text();
      if (text.trimStart().startsWith("<")) container.innerHTML = text; // trusted: served from our own /lessons
      else container.replaceChildren(...text.split(/\n{2,}/).map((p) => el("p", { text: p })));
      typesetMath(container);
      highlightSnippets(container);
      appendWholeCode(container);
      wireBlockInfo(container);
    } catch (e) {
      container.replaceChildren(el("p", { class: "note", text: "lesson coming soon" }));
    }
  }

  route();
})();
