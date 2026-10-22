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
          else input.value = v;
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
      el("span", { class: "min", text: t.minutes_label || "" }),
      el("span", { class: "s", text: t.subtitle || "" }));
    if (t.running) a.append(el("span", { class: "run", text: "running" }));
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
        if (lastShown) showResult(lastShown.result, lastShown.evalId);
      });
      main.append(el("div", { class: "ctl" },
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
      main.append(el("section", { class: "card" },
        el("h2", {}, "Training ", el("span", { class: "k", text: "· run, watch, perturb" })),
        paramsEl,
        el("div", { class: "ctl" }, startBtn, pauseBtn, stopBtn, resetBtn, runSel, estEl),
        errEl, statusEl, chartsEl));

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
        if (st.msg) kids.push(el("span", { text: st.msg }));
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
          const m = c.metrics || {};
          const key = Object.keys(m).find((k) => typeof m[k] === "number");
          const head = key ? ` · ${key} ${Number(m[key]).toPrecision(3)}` : "";
          ckptSel.append(el("option", { value: c.id, text: `step ${fmtInt(c.step)}${c.label ? " · " + c.label : ""}${head} · run ${c.run_id}` }));
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
          evalStatus.replaceChildren(
            el("span", {}, el("span", { class: `dot ${st.state}` }), el("b", { text: st.state || "?" })),
            st.wall_s != null ? el("span", {}, "wall ", el("b", { text: fmtWall(st.wall_s) })) : null,
            st.msg ? el("span", { text: st.msg }) : null);
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
          vecgrids: window.LabVecGrid.enabled(taskId),
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

  // Outcome table: for every top-level section of the result with scalar numbers, one row per number;
  // a "wilson95" [lo, hi] on the section is shown as the CI of its "success" rate.
  function outcomeTable(result) {
    const rows = [];
    const pct = (v) => (v >= 0 && v <= 1 ? `${(v * 100).toFixed(1)} %` : Number(v).toPrecision(3));
    for (const [sec, obj] of Object.entries(result || {})) {
      if (!obj || typeof obj !== "object" || Array.isArray(obj)) continue;
      for (const [k, v] of Object.entries(obj)) {
        if (typeof v !== "number") continue;
        let ci = "—";
        if (Array.isArray(obj.wilson95) && k === "success") ci = `${(obj.wilson95[0] * 100).toFixed(0)}–${(obj.wilson95[1] * 100).toFixed(0)}`;
        const bad = /collision|error/.test(k);
        rows.push(el("tr", {}, el("td", { text: `${sec} ${k}` }), el("td", {}, el("span", { class: bad ? "chip bad" : "chip", text: pct(v) })), el("td", { text: ci })));
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

  async function loadLesson(url, container) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(String(r.status));
      const text = await r.text();
      if (text.trimStart().startsWith("<")) container.innerHTML = text; // trusted: served from our own /lessons
      else container.replaceChildren(...text.split(/\n{2,}/).map((p) => el("p", { text: p })));
      typesetMath(container);
    } catch (e) {
      container.replaceChildren(el("p", { class: "note", text: "lesson coming soon" }));
    }
  }

  route();
})();
