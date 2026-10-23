/* Result viewers. Contract for real viewers (added by later tasks):
 *   LabViewers.register(name, fn)  where fn(resultObj, container, ctx) renders into container.
 *   ctx = {task, evalId, vecgrids (bool: per-task vector-grid toggle; see vecgrid.js), fileUrl(name) -> "/api/evals/<task>/<eval>/file/<name>"}.
 *   LabViewers.get(name) -> the registered fn (live panels: spec.live, called as fn(metricsRow, container, ctx) per new row).
 *   LabViewers.render(viewerName, resultObj, container, ctx) dispatches by spec.viewer;
 *   unknown/unregistered names fall back to pretty-printed JSON. */
(function () {
  const registry = {};

  function jsonFallback(result, container) {
    const pre = document.createElement("pre");
    pre.className = "json";
    pre.textContent = JSON.stringify(result, null, 2);
    container.replaceChildren(pre);
  }

  function render(viewer, result, container, ctx) {
    const fn = registry[viewer] || jsonFallback;
    try {
      fn(result, container, ctx || {});
    } catch (e) {
      jsonFallback(result, container);
    }
  }

  // ---- shared helpers -------------------------------------------------------------------
  const NS = "http://www.w3.org/2000/svg";
  const T1_OBSTACLE = 0.25, T1_HALF_GAP = 0.5; // toy.py: T1_OBSTACLE, T1_HALF_GAP

  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") e.textContent = v; else e.setAttribute(k, v);
    }
    for (const c of kids) if (c) e.append(c);
    return e;
  }
  function s(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }
  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  const num = (x) => (typeof x === "number" && isFinite(x) ? x : null);
  const pct = (x) => (num(x) == null ? "n/a" : (100 * x).toFixed(1) + "%");
  function chip(label, value) {
    return h("span", { class: "chip", style: "background:var(--chip);border-radius:999px;padding:2px 10px;font:12px var(--mono);margin-right:8px;display:inline-block" },
      document.createTextNode(label + " "), h("b", { text: value }));
  }

  // ---- t1scatter: observation c vs sampled a_x ------------------------------------------
  function t1scatter(result, container) {
    const t1 = (result && result.t1) || {};
    const pts = Array.isArray(t1.points) ? t1.points.filter((p) => Array.isArray(p) && num(p[0]) != null && num(p[1]) != null) : [];
    const wrap = h("div", { class: "viewer-t1" });
    const stats = h("div", { style: "margin-bottom:6px" });
    stats.append(chip("collision", pct(t1.collision)), chip("left fraction", pct(t1.left_frac)),
      chip("mode error", num(t1.mode_err) == null ? "n/a" : t1.mode_err.toFixed(3)));
    wrap.append(stats);

    const W = 420, H = 300, L = 40, R = 10, T = 10, B = 32;
    const cx0 = -0.4, cx1 = 0.4, ay0 = -1, ay1 = 1;
    const X = (c) => L + ((c - cx0) / (cx1 - cx0)) * (W - L - R);
    const Y = (a) => T + (1 - (a - ay0) / (ay1 - ay0)) * (H - T - B);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;max-width:520px;height:auto;font:10px var(--mono)", role: "img", "aria-label": "T1 samples: obstacle position vs sampled action" });
    svg.append(s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    // obstacle band |a_x - c| < 0.25, as a polygon along the diagonal a_x = c
    const band = [[cx0, cx0 - T1_OBSTACLE], [cx1, cx1 - T1_OBSTACLE], [cx1, cx1 + T1_OBSTACLE], [cx0, cx0 + T1_OBSTACLE]];
    svg.append(s("polygon", { points: band.map(([c, a]) => `${X(c)},${Y(a)}`).join(" "), fill: "var(--bad-soft)", stroke: "var(--bad)", "stroke-dasharray": "3 3", "stroke-opacity": "0.6" }));
    for (const sign of [-1, 1]) {
      svg.append(s("line", { x1: X(cx0), y1: Y(cx0 + sign * T1_HALF_GAP), x2: X(cx1), y2: Y(cx1 + sign * T1_HALF_GAP), stroke: "var(--good)", "stroke-width": 1.5, "stroke-dasharray": "5 3" }));
    }
    for (const a of [-1, -0.5, 0, 0.5, 1]) {
      svg.append(s("text", { x: L - 5, y: Y(a) + 3, "text-anchor": "end", fill: "var(--muted)" }, String(a)));
    }
    for (const c of [-0.4, -0.2, 0, 0.2, 0.4]) {
      svg.append(s("text", { x: X(c), y: H - B + 13, "text-anchor": "middle", fill: "var(--muted)" }, String(c)));
    }
    svg.append(s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: "var(--muted)" }, "obstacle position c"));
    svg.append(s("text", { x: 11, y: (T + H - B) / 2, "text-anchor": "middle", fill: "var(--muted)", transform: `rotate(-90 11 ${(T + H - B) / 2})` }, "sampled a_x"));

    const tip = h("div", { style: "font:12px var(--mono);color:var(--muted);min-height:1.4em;margin-top:4px", text: pts.length ? "hover a point for its values" : "no samples in this result" });
    for (const p of pts) {
      const hit = Math.abs(p[1] - p[0]) < T1_OBSTACLE;
      const dot = s("circle", { cx: X(p[0]), cy: Y(Math.max(ay0, Math.min(ay1, p[1]))), r: 3, fill: hit ? "var(--bad)" : "var(--good)", "fill-opacity": 0.75 });
      const show = () => { tip.textContent = `c=${p[0].toFixed(3)}  a_x=${p[1].toFixed(3)}  a_y=${num(p[2]) == null ? "n/a" : p[2].toFixed(3)}  ${hit ? "collision" : "clear"}`; };
      dot.addEventListener("mouseenter", show);
      dot.addEventListener("mousemove", show);
      dot.append(s("title", {}, `c=${p[0].toFixed(3)}, a_x=${p[1].toFixed(3)}${hit ? " (collision)" : ""}`));
      svg.append(dot);
    }
    wrap.append(svg, tip);
    wrap.append(h("div", { class: "leg", style: "font-size:11px;color:var(--muted)", text: "shaded band = obstacle (|a_x - c| < 0.25), dashed green = the two expert modes c ± 0.5; red points collide" }));
    container.replaceChildren(wrap);
  }

  // ---- traj2d: animated rollouts --------------------------------------------------------
  function traj2d(result, container) {
    const t2 = (result && result.t2) || {};
    const eps = Array.isArray(t2.episodes) ? t2.episodes.filter((e) => e && Array.isArray(e.p) && e.p.length) : [];
    const wrap = h("div", { class: "viewer-traj2d" });
    const stats = h("div", { style: "margin-bottom:6px" });
    stats.append(chip("success", pct(t2.success)), chip("collision", pct(t2.collision)), chip("timeout", pct(t2.timeout)));
    if (Array.isArray(t2.wilson95) && t2.wilson95.length === 2) {
      stats.append(chip("95% CI", `${pct(t2.wilson95[0])} – ${pct(t2.wilson95[1])}`));
    }
    wrap.append(stats);
    if (!eps.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No episodes to animate (0 episodes in this result)." }));
      container.replaceChildren(wrap);
      return;
    }
    const maxLen = Math.max(...eps.map((e) => e.p.length));
    const R = num(t2.obstacle_radius) || 0.3;
    const SIZE = 360;
    const canvas = h("canvas", { width: SIZE, height: SIZE, style: "width:100%;max-width:360px;height:auto;border:1px solid var(--line);border-radius:8px;display:block" });
    const label = h("span", { style: "font:12px var(--mono);color:var(--muted)" });
    const btn = h("button", { type: "button", text: "Pause" });
    const slider = h("input", { type: "range", min: 0, max: maxLen - 1, value: 0, step: 1, style: "flex:1;min-width:120px" });
    const row = h("div", { style: "display:flex;gap:10px;align-items:center;margin-top:8px;max-width:520px" }, btn, slider);
    wrap.append(canvas, row, label);
    container.replaceChildren(wrap);

    const ctx2 = canvas.getContext("2d");
    // world [-1.2, 1.2]^2 -> canvas, y up
    const WORLD = 1.2;
    const px = (x) => ((x + WORLD) / (2 * WORLD)) * SIZE;
    const py = (y) => SIZE - ((y + WORLD) / (2 * WORLD)) * SIZE;
    const sc = (r) => (r / (2 * WORLD)) * SIZE;
    const color = (o) => (o === "success" ? css("--good", "#1a7f4b") : o === "collision" ? css("--bad", "#b3261e") : css("--muted", "#6b7280"));
    let step = 0, playing = true, timer = null;

    function draw() {
      const bg = css("--panel", "#fff");
      ctx2.fillStyle = bg;
      ctx2.fillRect(0, 0, SIZE, SIZE);
      ctx2.strokeStyle = css("--line", "#e2e4e9");
      ctx2.strokeRect(0.5, 0.5, SIZE - 1, SIZE - 1);
      // obstacles (one disc per distinct centre) and goals
      const seen = new Set();
      for (const e of eps) {
        const o = Array.isArray(e.o) ? e.o : [0, 0];
        const k = o.join(",");
        if (seen.has(k)) continue;
        seen.add(k);
        ctx2.beginPath();
        ctx2.arc(px(o[0]), py(o[1]), sc(R), 0, 2 * Math.PI);
        ctx2.fillStyle = css("--bad-soft", "#fbe9e7");
        ctx2.globalAlpha = 0.5;
        ctx2.fill();
        ctx2.globalAlpha = 1;
        ctx2.strokeStyle = css("--bad", "#b3261e");
        ctx2.setLineDash([3, 3]);
        ctx2.stroke();
        ctx2.setLineDash([]);
      }
      for (const e of eps) {
        if (Array.isArray(e.g)) {
          ctx2.fillStyle = css("--muted", "#6b7280");
          ctx2.fillRect(px(e.g[0]) - 3, py(e.g[1]) - 3, 6, 6);
        }
      }
      for (const e of eps) {
        const n = Math.min(step + 1, e.p.length);
        ctx2.beginPath();
        for (let i = 0; i < n; i++) {
          const [x, y] = e.p[i];
          if (i === 0) ctx2.moveTo(px(x), py(y)); else ctx2.lineTo(px(x), py(y));
        }
        ctx2.strokeStyle = color(e.outcome);
        ctx2.globalAlpha = e.outcome === "timeout" ? 0.45 : 0.7;
        ctx2.lineWidth = 1.5;
        ctx2.stroke();
        ctx2.globalAlpha = 1;
        const last = e.p[n - 1];
        ctx2.beginPath();
        ctx2.arc(px(last[0]), py(last[1]), 2.5, 0, 2 * Math.PI);
        ctx2.fillStyle = color(e.outcome);
        ctx2.fill();
      }
      const cnt = { success: 0, collision: 0, timeout: 0 };
      for (const e of eps) if (cnt[e.outcome] != null) cnt[e.outcome]++;
      label.textContent = `${eps.length} episodes shown (${cnt.success} success, ${cnt.collision} collision, ${cnt.timeout} timeout) · step ${step}/${maxLen - 1}`;
      slider.value = step;
    }
    function tick() {
      if (!playing) return;
      if (step >= maxLen - 1) { playing = false; btn.textContent = "Replay"; return; }
      step += 1;
      draw();
    }
    // stop the timer when the viewer leaves the DOM
    timer = setInterval(() => { if (!canvas.isConnected) { clearInterval(timer); return; } tick(); }, 40);
    btn.addEventListener("click", () => {
      if (!playing && step >= maxLen - 1) step = 0;
      playing = !playing;
      btn.textContent = playing ? "Pause" : "Play";
      draw();
    });
    slider.addEventListener("input", () => {
      step = Number(slider.value);
      playing = false;
      btn.textContent = "Play";
      draw();
    });
    draw();
  }

  // ---- t1: composite for the task-1 spec (scatter + rollouts side by side) --------------
  function t1(result, container, ctx) {
    const a = h("div", { class: "chart", style: "min-width:0" }, h("h3", { text: "T1: sampled actions" }), h("div"));
    const b = h("div", { class: "chart", style: "min-width:0" }, h("h3", { text: "T2: rollouts" }), h("div"));
    const grid = h("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px" }, a, b);
    container.replaceChildren(grid);
    t1scatter(result, a.lastChild, ctx);
    traj2d(result, b.lastChild, ctx);
  }

  // ---- flowtransport: noise -> actions, animated over the K sampling steps ---------------
  function flowtransport(result, container) {
    const tr = (result && result.trace) || {};
    const xs = Array.isArray(tr.xs) ? tr.xs : [];
    const cs = Array.isArray(tr.c) ? tr.c : [];
    const wrap = h("div", { class: "viewer-flowtransport" });
    if (!xs.length || !cs.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No sampling trace in this result." }));
      container.replaceChildren(wrap);
      return;
    }
    const K = xs.length - 1;
    const W = 420, H = 300, L = 40, R = 10, T = 10, B = 32;
    const cx0 = -0.4, cx1 = 0.4, ay0 = -2.5, ay1 = 2.5;
    const X = (c) => L + ((c - cx0) / (cx1 - cx0)) * (W - L - R);
    const Y = (a) => T + (1 - (Math.max(ay0, Math.min(ay1, a)) - ay0) / (ay1 - ay0)) * (H - T - B);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;max-width:520px;height:auto;font:10px var(--mono)", role: "img", "aria-label": "Sampling trace: noise to actions" });
    svg.append(s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    const band = [[cx0, cx0 - T1_OBSTACLE], [cx1, cx1 - T1_OBSTACLE], [cx1, cx1 + T1_OBSTACLE], [cx0, cx0 + T1_OBSTACLE]];
    svg.append(s("polygon", { points: band.map(([c, a]) => `${X(c)},${Y(a)}`).join(" "), fill: "var(--bad-soft)", stroke: "var(--bad)", "stroke-dasharray": "3 3", "stroke-opacity": "0.6" }));
    for (const sign of [-1, 1]) {
      svg.append(s("line", { x1: X(cx0), y1: Y(cx0 + sign * T1_HALF_GAP), x2: X(cx1), y2: Y(cx1 + sign * T1_HALF_GAP), stroke: "var(--good)", "stroke-width": 1.5, "stroke-dasharray": "5 3" }));
    }
    for (const a of [-2, -1, 0, 1, 2]) svg.append(s("text", { x: L - 5, y: Y(a) + 3, "text-anchor": "end", fill: "var(--muted)" }, String(a)));
    for (const c of [-0.4, -0.2, 0, 0.2, 0.4]) svg.append(s("text", { x: X(c), y: H - B + 13, "text-anchor": "middle", fill: "var(--muted)" }, String(c)));
    svg.append(s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: "var(--muted)" }, "obstacle position c"));
    svg.append(s("text", { x: 11, y: (T + H - B) / 2, "text-anchor": "middle", fill: "var(--muted)", transform: `rotate(-90 11 ${(T + H - B) / 2})` }, "sample a_x"));
    const dots = cs.map(() => { const d = s("circle", { r: 3, fill: "var(--accent)", "fill-opacity": 0.75 }); svg.append(d); return d; });
    const label = h("span", { style: "font:12px var(--mono);color:var(--muted)" });
    const btn = h("button", { type: "button", text: "Pause" });
    const slider = h("input", { type: "range", min: 0, max: K, value: 0, step: 1, style: "flex:1;min-width:120px" });
    const row = h("div", { style: "display:flex;gap:10px;align-items:center;margin-top:8px;max-width:520px" }, btn, slider);
    wrap.append(svg, row, label);
    container.replaceChildren(wrap);
    let step = 0, playing = true;
    function draw() {
      xs[step].forEach((p, i) => {
        const hit = Math.abs(p[0] - cs[i]) < T1_OBSTACLE;
        dots[i].setAttribute("cx", X(cs[i]));
        dots[i].setAttribute("cy", Y(p[0]));
        dots[i].setAttribute("fill", step === K ? (hit ? "var(--bad)" : "var(--good)") : "var(--accent)");
      });
      label.textContent = step === 0 ? `step 0/${K}: pure noise` : `step ${step}/${K}${step === K ? ": final actions" : ""}`;
      slider.value = step;
    }
    const timer = setInterval(() => {
      if (!svg.isConnected) { clearInterval(timer); return; }
      if (!playing) return;
      if (step >= K) { playing = false; btn.textContent = "Replay"; return; }
      step += 1; draw();
    }, 450);
    btn.addEventListener("click", () => {
      if (!playing && step >= K) step = 0;
      playing = !playing; btn.textContent = playing ? "Pause" : "Play"; draw();
    });
    slider.addEventListener("input", () => { step = Number(slider.value); playing = false; btn.textContent = "Play"; draw(); });
    draw();
  }

  // ---- stepsweep: T1 collision vs number of sampling steps (eval-only data) --------------
  function stepsweep(result, container) {
    const sw = (result && result.sweep) || {};
    const data = sw.euler || {};
    const ks = Object.keys(data).map(Number).filter((k) => num(data[String(k)]) != null).sort((a, b) => a - b);
    const wrap = h("div", { class: "viewer-stepsweep" });
    if (!ks.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No step sweep in this result." }));
      container.replaceChildren(wrap);
      return;
    }
    const W = 420, H = 240, L = 44, R = 12, T = 12, B = 34;
    const X = (i) => L + (i / Math.max(1, ks.length - 1)) * (W - L - R);
    const Y = (v) => T + (1 - v) * (H - T - B);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;max-width:520px;height:auto;font:10px var(--mono)", role: "img", "aria-label": "T1 collision rate vs sampling steps" });
    svg.append(s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(s("text", { x: L - 5, y: Y(v) + 3, "text-anchor": "end", fill: "var(--muted)" }, pct(v)));
      svg.append(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--line)", "stroke-opacity": 0.5 }));
    }
    const cur = num(result.sample_steps);
    ks.forEach((k, i) => svg.append(s("text", { x: X(i), y: H - B + 14, "text-anchor": "middle", fill: k === cur ? "var(--ink)" : "var(--muted)", "font-weight": k === cur ? "700" : "400" }, String(k))));
    svg.append(s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: "var(--muted)" }, `sampling steps K (${sw.kind || "euler"})`));
    svg.append(s("polyline", { points: ks.map((k, i) => `${X(i)},${Y(data[String(k)])}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
    ks.forEach((k, i) => {
      const v = data[String(k)];
      const d = s("circle", { cx: X(i), cy: Y(v), r: k === cur ? 5 : 3.5, fill: "var(--accent)" });
      d.append(s("title", {}, `K=${k}: T1 collision ${pct(v)}`));
      svg.append(d);
      svg.append(s("text", { x: X(i), y: Y(v) - 8, "text-anchor": "middle", fill: "var(--ink)" }, pct(v)));
    });
    wrap.append(svg, h("div", { class: "leg", style: "font-size:11px;color:var(--muted)", text: "T1 collision rate of the same trained net, sampled with K steps (bold = the K used for the scatter and rollouts)." }));
    container.replaceChildren(wrap);
  }

  // ---- flow: composite for the task-2 spec -----------------------------------------------
  function flow(result, container, ctx) {
    const card = (title) => h("div", { class: "chart", style: "min-width:0" }, h("h3", { text: title }), h("div"));
    const head = result && result.head ? `${result.head}, K=${result.sample_steps}` : "";
    const a = card(`T1: noise to actions (${head})`), b = card("T1: sampled actions"),
      c = card("T1 collision vs sampling steps"), d = card("T2: rollouts");
    const grid = h("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px" }, a, b, c, d);
    container.replaceChildren(grid);
    flowtransport(result, a.lastChild, ctx);
    t1scatter(result, b.lastChild, ctx);
    stepsweep(result, c.lastChild, ctx);
    traj2d(result, d.lastChild, ctx);
  }

  // ---- contact2d: peg-search side view with force arrow and Fx/Fz strip chart --------------
  // Geometry mirrors src/fvb/ta_tutorial/contact2d.py: WH=0.006 (hole half-width), WC=0.012
  // (chamfer width / depth), DEPTH=0.06 (hole depth below the chamfer bottom).
  const C2_WH = 0.006, C2_WC = 0.012, C2_DEPTH = 0.06, C2_FABORT = 15.0;

  function contact2d(result, container, ctx) {
    const eps = Array.isArray(result && result.episodes)
      ? result.episodes.filter((e) => e && Array.isArray(e.pos) && e.pos.length && Array.isArray(e.force)) : [];
    const wrap = h("div", { class: "viewer-contact2d" });
    const stats = h("div", { style: "margin-bottom:6px" });
    stats.append(chip("success", pct(result && result.success)));
    const counts = (result && result.counts) || {};
    for (const o of ["force_abort", "wrong_hole", "timeout"]) {
      if (num(counts[o]) != null && num(result.n_episodes)) stats.append(chip(o.replace("_", " "), pct(counts[o] / result.n_episodes)));
    }
    if (Array.isArray(result && result.wilson95) && result.wilson95.length === 2) {
      stats.append(chip("95% CI", `${pct(result.wilson95[0])} – ${pct(result.wilson95[1])}`));
    }
    if (result && result.force_zeroed) {
      const twf = result.trained_with_force; // absent in results written before the field existed
      stats.append(chip("force input", twf === false ? "zeroed (trained without force)" : twf === true ? "blinded at eval (trained with force)" : "zeroed"));
    }
    wrap.append(stats);
    if (!eps.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No episodes to animate in this result." }));
      container.replaceChildren(wrap);
      return;
    }
    const W = 480, HC = 300, HS = 100;
    const X0 = -0.16, X1 = 0.16, Z0 = -0.12, Z1 = 0.15;
    const px = (x) => ((x - X0) / (X1 - X0)) * W;
    const pz = (z) => HC - ((z - Z0) / (Z1 - Z0)) * HC;
    const outColor = (o) => (o === "success" ? css("--good", "#1a7f4b") : o === "force_abort" ? css("--bad", "#b3261e")
      : o === "wrong_hole" ? css("--warn", "#b7791f") : css("--muted", "#6b7280"));
    const canvas = h("canvas", { width: W, height: HC, style: "width:100%;max-width:480px;height:auto;display:block;border-radius:8px;border:3px solid var(--line)" });
    const strip = h("canvas", { width: W, height: HS, style: "width:100%;max-width:480px;height:auto;display:block;margin-top:6px;border:1px solid var(--line);border-radius:6px" });
    const sel = h("select", {});
    eps.forEach((e, i) => sel.append(h("option", { value: i, text: `episode ${i + 1}: ${e.task ? e.task + " - " : ""}${e.outcome}` })));
    const btn = h("button", { type: "button", text: "Pause" });
    const slider = h("input", { type: "range", min: 0, max: 1, value: 0, step: 1, style: "flex:1;min-width:120px" });
    const label = h("span", { style: "font:12px var(--mono);color:var(--muted)" });
    const row = h("div", { style: "display:flex;gap:10px;align-items:center;margin:8px 0;max-width:480px" }, sel, btn, slider);
    wrap.append(row, canvas, strip, label);
    // vector grids (only what the payload holds): the 10x2 force window the net reads as its torque
    // input (zeros when blinded) and the last 10 applied actions (the current one is the last row).
    const vg = window.LabVecGrid && window.LabVecGrid.panel(ctx);
    let vgBox = null;
    if (vg) {
      vgBox = h("div", { class: "vecgrids" });
      wrap.append(vgBox, h("p", { class: "note", text:
        "Grids: rows are time (oldest to newest), columns are (x, z). Force window = the last 10 force readings, "
        + "the net's torque-history input (all zero when the force input is blinded). Actions = the previous 9 actions plus the "
        + "one chosen at this step (newest last; grey = no action, the episode has ended). Red negative, black zero, green positive; hover for values." }));
    }
    container.replaceChildren(wrap);
    let gridEp = null, forceG = null, actG = null, encG = null, baseG = null, deltaG = null;
    function drawGrids(e, st) {
      if (!vgBox) return;
      const HW = 10, win = [];
      for (let i = st - HW + 1; i <= st; i++) win.push(i >= 0 ? e.force[Math.min(i, e.force.length - 1)] : [0, 0]);
      const blind = !!(result && result.force_zeroed);
      const hasAct = Array.isArray(e.act) && e.act.length > 0;
      if (gridEp !== e) { // (re)build only when the episode changes; per frame we update in place
        vgBox.replaceChildren();
        forceG = window.LabVecGrid.render(vgBox, win, { title: blind ? "force window (blinded: zeros)" : "force window 10x2 (N)", vmax: C2_FABORT });
        actG = hasAct ? window.LabVecGrid.render(vgBox, [[0, 0]], { title: "actions 10x2 (ax, az): previous 9 + chosen now", vmax: 1 }) : null;
        // residual-rl task: the frozen base policy's action and the residual that was added to it
        const hasRes = hasAct && Array.isArray(e.base_act) && e.base_act.length === e.act.length && Array.isArray(e.delta);
        baseG = hasRes ? window.LabVecGrid.render(vgBox, [[0, 0]], { title: "base policy action 10x2 (frozen flow net)", vmax: 1 }) : null;
        deltaG = hasRes ? window.LabVecGrid.render(vgBox, [[0, 0]], { title: "residual added 10x2 (scale x delta)", vmax: 0.5 }) : null;
        // embeddings task: the command vector the net consumed for this episode (raw; text = pooled learned embedding)
        encG = Array.isArray(e.enc) && e.enc.length
          ? window.LabVecGrid.render(vgBox, e.enc, { title: `command encoding (${(result && result.encoding) || "enc"}, ${e.enc.length}-D)`, cols: Math.min(e.enc.length, 8) }) : null;
        gridEp = e;
      }
      forceG.update(blind ? win.map(() => [0, 0]) : win);
      if (actG) {
        const acts = [];
        // act[i] is chosen at step i (moves pos[i] -> pos[i+1]); beyond the last step it does not exist -> n/a
        for (let i = st - HW + 1; i <= st; i++) acts.push(i < 0 ? [NaN, NaN] : i < e.act.length ? e.act[i] : [NaN, NaN]);
        actG.update(acts);
        for (const [g2, key] of [[baseG, "base_act"], [deltaG, "delta"]]) {
          if (!g2) continue;
          const rows = [];
          for (let i = st - HW + 1; i <= st; i++) rows.push(i < 0 ? [NaN, NaN] : i < e[key].length ? e[key][i] : [NaN, NaN]);
          g2.update(rows);
        }
      }
    }
    const g = canvas.getContext("2d"), gs = strip.getContext("2d");
    let ep = eps[0], step = 0, playing = true;

    function surface(e) {
      const z0 = e.z0, zf = z0 - C2_WC, zb = z0 - C2_WC - C2_DEPTH;
      const pts = [[X0 - 0.05, z0]];
      for (const hx of [...e.holes].sort((a, b) => a - b)) {
        pts.push([hx - C2_WH - C2_WC, z0], [hx - C2_WH, zf], [hx - C2_WH, zb], [hx + C2_WH, zb], [hx + C2_WH, zf], [hx + C2_WH + C2_WC, z0]);
      }
      pts.push([X1 + 0.05, z0]);
      return pts;
    }
    function draw() {
      const last = ep.pos.length - 1;
      step = Math.max(0, Math.min(step, last));
      canvas.style.borderColor = outColor(ep.outcome);
      g.fillStyle = css("--panel", "#fff");
      g.fillRect(0, 0, W, HC);
      const sp = surface(ep);
      g.beginPath();
      sp.forEach(([x, z], i) => (i ? g.lineTo(px(x), pz(z)) : g.moveTo(px(x), pz(z))));
      g.lineTo(W, HC); g.lineTo(0, HC); g.closePath();
      g.fillStyle = css("--chip", "#eceef2"); g.fill();
      g.strokeStyle = css("--ink", "#1f2430"); g.lineWidth = 1.5; g.stroke();
      // the policy's hole estimate (dashed); two-hole scenes (embeddings task): perceived xs per hole + red/blue hole markers
      g.setLineDash([4, 4]); g.strokeStyle = css("--muted", "#6b7280"); g.lineWidth = 1;
      g.fillStyle = css("--muted", "#6b7280"); g.font = "10px monospace";
      const ests = Array.isArray(ep.est) ? ep.est : ep.est == null ? [] : [ep.est];
      for (const ex of ests) { g.beginPath(); g.moveTo(px(ex), 0); g.lineTo(px(ex), HC * 0.25); g.stroke(); }
      g.setLineDash([]);
      if (ests.length) g.fillText(ests.length > 1 ? "perceived" : "estimate", px(ests[0]) + 3, 12);
      if (ep.holes.length === 2) {
        ep.holes.forEach((hx, k) => {
          const col = k === 0 ? "#e5483d" : "#3b6ff0", isT = ep.target === k;
          g.fillStyle = col;
          g.beginPath(); g.moveTo(px(hx) - 6, pz(ep.z0) - 16); g.lineTo(px(hx) + 6, pz(ep.z0) - 16); g.lineTo(px(hx), pz(ep.z0) - 5); g.closePath(); g.fill();
          if (isT) { g.strokeStyle = col; g.lineWidth = 2; g.strokeRect(px(hx) - 9, pz(ep.z0) - 20, 18, 18); }
        });
      }
      // trail, peg, force arrow
      g.beginPath();
      for (let i = 0; i <= step; i++) { const [x, z] = ep.pos[i]; i ? g.lineTo(px(x), pz(z)) : g.moveTo(px(x), pz(z)); }
      g.strokeStyle = outColor(ep.outcome); g.globalAlpha = 0.5; g.lineWidth = 1.5; g.stroke(); g.globalAlpha = 1;
      const [x, z] = ep.pos[step], [fx, fz] = ep.force[Math.min(step, ep.force.length - 1)];
      const f = Math.hypot(fx, fz);
      if (f > 0.3) {
        const L = Math.min(70, 4.5 * f), ax = px(x) + (fx / f) * L, az = pz(z) - (fz / f) * L;
        g.strokeStyle = css("--accent", "#2f6fed"); g.lineWidth = 2;
        g.beginPath(); g.moveTo(px(x), pz(z)); g.lineTo(ax, az); g.stroke();
        const a = Math.atan2(az - pz(z), ax - px(x));
        g.beginPath(); g.moveTo(ax, az); g.lineTo(ax - 7 * Math.cos(a - 0.4), az - 7 * Math.sin(a - 0.4));
        g.lineTo(ax - 7 * Math.cos(a + 0.4), az - 7 * Math.sin(a + 0.4)); g.closePath();
        g.fillStyle = css("--accent", "#2f6fed"); g.fill();
      }
      g.beginPath(); g.arc(px(x), pz(z), 4.5, 0, 2 * Math.PI);
      g.fillStyle = css("--ink", "#1f2430"); g.fill();
      // strip chart: Fx, Fz over time, cursor at the current step
      gs.fillStyle = css("--panel", "#fff"); gs.fillRect(0, 0, W, HS);
      const fm = Math.max(C2_FABORT, ...ep.force.map((v) => Math.max(Math.abs(v[0]), Math.abs(v[1]))));
      const tx = (i) => (i / Math.max(1, last)) * (W - 2) + 1, fy = (v) => HS / 2 - (v / fm) * (HS / 2 - 6);
      gs.strokeStyle = css("--line", "#e2e4e9"); gs.lineWidth = 1;
      gs.beginPath(); gs.moveTo(0, fy(0)); gs.lineTo(W, fy(0)); gs.stroke();
      gs.setLineDash([3, 3]); gs.strokeStyle = css("--bad", "#b3261e");
      for (const v of [C2_FABORT, -C2_FABORT]) { gs.beginPath(); gs.moveTo(0, fy(v)); gs.lineTo(W, fy(v)); gs.stroke(); }
      gs.setLineDash([]);
      [[0, css("--warn", "#b7791f"), "Fx"], [1, css("--accent", "#2f6fed"), "Fz"]].forEach(([k, col, name], j) => {
        gs.beginPath();
        ep.force.forEach((v, i) => (i ? gs.lineTo(tx(i), fy(v[k])) : gs.moveTo(tx(i), fy(v[k]))));
        gs.strokeStyle = col; gs.lineWidth = 1.3; gs.stroke();
        gs.fillStyle = col; gs.font = "10px monospace"; gs.fillText(name, 4 + 24 * j, 11);
      });
      gs.fillStyle = css("--muted", "#6b7280"); gs.fillText(`±${C2_FABORT} N abort`, W - 84, 11);
      gs.strokeStyle = css("--ink", "#1f2430");
      gs.beginPath(); gs.moveTo(tx(step), 0); gs.lineTo(tx(step), HS); gs.stroke();
      drawGrids(ep, step);
      slider.max = last; slider.value = step;
      label.textContent = `${ep.task ? ep.task + " · " : ""}${ep.outcome} · step ${step}/${last} · F = (${fx.toFixed(1)}, ${fz.toFixed(1)}) N`;
    }
    const timer = setInterval(() => {
      if (!canvas.isConnected) { clearInterval(timer); return; }
      if (!playing) return;
      if (step >= ep.pos.length - 1) { playing = false; btn.textContent = "Replay"; return; }
      step += 1; draw();
    }, 40);
    sel.addEventListener("change", () => { ep = eps[Number(sel.value)]; step = 0; playing = true; btn.textContent = "Pause"; draw(); });
    btn.addEventListener("click", () => {
      if (!playing && step >= ep.pos.length - 1) step = 0;
      playing = !playing; btn.textContent = playing ? "Pause" : "Play"; draw();
    });
    slider.addEventListener("input", () => { step = Number(slider.value); playing = false; btn.textContent = "Play"; draw(); });
    draw();
  }

  // per-command outcome table shared by the embeddings and tavla viewers
  function commandTable(wrap, result, what) {
    const cmds = Array.isArray(result && result.commands) ? result.commands : [];
    if (!cmds.length) return;
    const tbl = h("table", { class: "out" });
    tbl.append(h("tr", {}, ...["command", "n", "success", "95% CI", "force abort", "wrong hole", "timeout"].map((t) => h("th", { text: t }))));
    for (const c of cmds) {
      const k = c.counts || {}, ci = Array.isArray(c.wilson95) ? `${pct(c.wilson95[0])} – ${pct(c.wilson95[1])}` : "n/a";
      const cell = (o) => (c.n ? pct((k[o] || 0) / c.n) : "–");
      tbl.append(h("tr", {}, h("td", { text: c.name }), h("td", { text: String(c.n) }), h("td", { text: c.n ? pct(c.success) : "–" }),
        h("td", { text: c.n ? ci : "–" }), h("td", { text: cell("force_abort") }), h("td", { text: cell("wrong_hole") }), h("td", { text: cell("timeout") })));
    }
    wrap.append(h("h4", { text: `Per-command outcomes - ${what}${result.tasks && result.tasks !== "all" ? ` (${result.tasks} commands only)` : ""}` }), tbl);
  }

  // ---- tavla: per-command table + contact2d rollouts (the image-conditioned TA-VLA policy)
  function tavla(result, container, ctx) {
    const wrap = h("div", { class: "viewer-tavla" });
    commandTable(wrap, result, `torque token: ${(result && result.torque_pos) || "?"}`);
    const rolls = h("div", {});
    wrap.append(h("h4", { text: "Closed-loop rollouts" }), rolls);
    container.replaceChildren(wrap);
    contact2d(result, rolls, ctx);
  }

  // ---- embeddings: outcome table per command + vecgrid matrix (commands x encodings) + contact2d rollouts
  function embeddings(result, container, ctx) {
    const wrap = h("div", { class: "viewer-embeddings" });
    commandTable(wrap, result, `${result.encoding} encoding`);
    const enc = result && result.encodings, vg = window.LabVecGrid && window.LabVecGrid.panel(ctx);
    if (enc && Array.isArray(enc.arms) && vg) {
      wrap.append(h("h4", { text: "What each encoding hands the network (raw conditioning vectors)" }));
      const grid = h("div", { style: `display:grid;grid-template-columns:150px repeat(${enc.arms.length},auto);gap:10px 18px;align-items:start;margin:6px 0` });
      grid.append(h("div", {}));
      enc.arms.forEach((a) => grid.append(h("div", { style: "font:600 12px var(--mono)", text: `${a} (${(enc.vectors[a][0] || []).length}-D)` })));
      const vmax = {};
      enc.arms.forEach((a) => { vmax[a] = Math.max(1e-6, ...enc.vectors[a].flat().map(Math.abs)); });
      (enc.commands || []).forEach((name, r) => {
        grid.append(h("div", { style: "font:12px var(--mono);color:var(--muted)", text: name }));
        enc.arms.forEach((a) => {
          const cell = h("div", {});
          const v = enc.vectors[a][r];
          window.LabVecGrid.render(cell, v, { labels: enc.labels && enc.labels[a], vmax: vmax[a], cols: Math.min(v.length, 8), cell: 16 });
          grid.append(cell);
        });
      });
      wrap.append(grid, h("p", { class: "note", text:
        "Rows are the four commands, columns the trained encodings; each cell is the vector the network receives, drawn with one colour scale per column "
        + "(red negative, black zero, green positive; hover a cell for its name and value). onehot and binary carry the same information in different geometry; "
        + "text is the learned embedding after training (the pooled word vectors); cont is in physical units - its largest entry (newtons) sets the scale, hover for the hole x. "
        + (enc.note || "") }));
    }
    const rolls = h("div", {});
    wrap.append(h("h4", { text: "Closed-loop rollouts" }), rolls);
    container.replaceChildren(wrap);
    contact2d(result, rolls, ctx);
  }


  // ---- video: gantry rollout mp4 + outcome table + per-episode force/action strip charts ----------
  // result.json from lab/tasks/task4_force_token.py. Frame layout of rollout.mp4 per rendered episode:
  // VID_CARD title-card frames, then 1 initial frame + one per step, then VID_HOLD hold frames.
  const VID_CARD = 15, VID_HOLD = 20, VID_FABORT = 60.0;

  function video(result, container, ctx) {
    const r = result || {};
    const eps = Array.isArray(r.episodes) ? r.episodes.filter((e) => e && Array.isArray(e.force) && e.force.length) : [];
    const wrap = h("div", { class: "viewer-video" });
    const stats = h("div", { style: "margin-bottom:6px" });
    stats.append(chip("success", pct(r.success)));
    if (Array.isArray(r.wilson95) && r.wilson95.length === 2) stats.append(chip("95% CI", `${pct(r.wilson95[0])} – ${pct(r.wilson95[1])}`));
    if (r.model) stats.append(chip("policy", `${r.model} H=${r.H} K=${r.K}`));
    if (r.force_zeroed) {
      const twf = r.trained_with_force;
      stats.append(chip("force input", twf === false ? "zeroed (trained without force)" : twf === true ? "blinded at eval (trained with force)" : "zeroed"));
    }
    wrap.append(stats);
    // outcome table
    const counts = r.counts || {}, n = num(r.n_episodes) || 0;
    const tbl = h("table", { class: "out" });
    tbl.append(h("tr", {}, ...["outcome", "episodes", "share"].map((t) => h("th", { text: t }))));
    for (const o of ["success", "force_abort", "timeout"]) {
      if (num(counts[o]) != null) tbl.append(h("tr", {}, h("td", { text: o.replace("_", " ") }), h("td", { text: String(counts[o]) }), h("td", { text: n ? pct(counts[o] / n) : "n/a" })));
    }
    wrap.append(tbl);
    const bk = Array.isArray(r.buckets) ? r.buckets.filter((b) => b && b.n > 0) : [];
    if (bk.length) {
      const bt = h("table", { class: "out", style: "margin-top:8px" });
      bt.append(h("tr", {}, ...["|hole offset| mm", "episodes", "success"].map((t) => h("th", { text: t }))));
      for (const b of bk) bt.append(h("tr", {}, h("td", { text: b.hi == null ? `${b.lo}+` : `${b.lo} – ${b.hi}` }), h("td", { text: String(b.n) }), h("td", { text: pct(b.success) })));
      wrap.append(bt, h("p", { class: "note", text: `Clearance ${r.clearance_mm} mm: offsets under about half of it go in without any search; the force policy earns its keep on the larger ones.` }));
    }
    const vid = h("video", { controls: "", preload: "metadata", style: "width:100%;max-width:640px;display:block;margin:10px 0;border-radius:8px;background:#000" });
    const hasVideo = !!(r.video && ctx && typeof ctx.fileUrl === "function");
    if (hasVideo) { vid.setAttribute("src", ctx.fileUrl(r.video)); wrap.append(vid); }
    else wrap.append(h("p", { class: "note", text: "No video in this result (rendered episodes = 0)." }));
    if (!eps.length) { container.replaceChildren(wrap); return; }
    const W = 640, HS = 110;
    const sel = h("select", {});
    eps.forEach((e, i) => sel.append(h("option", { value: i, text: `episode ${i + 1}: ${e.outcome} (offset ${e.offset_mm[0].toFixed(1)}, ${e.offset_mm[1].toFixed(1)} mm)` })));
    const label = h("span", { style: "font:12px var(--mono);color:var(--muted)" });
    const fc = h("canvas", { width: W, height: HS, style: "width:100%;max-width:640px;height:auto;display:block;margin-top:6px;border:1px solid var(--line);border-radius:6px" });
    const ac = h("canvas", { width: W, height: HS, style: "width:100%;max-width:640px;height:auto;display:block;margin-top:6px;border:1px solid var(--line);border-radius:6px" });
    wrap.append(h("div", { style: "margin:8px 0" }, sel), fc, ac, label);
    const vg = eps.some((e) => Array.isArray(e.obs_force)) && window.LabVecGrid && window.LabVecGrid.panel(ctx);
    let vgBox = null, obsG = null, gridEp = null;
    if (vg) {
      vgBox = h("div", { class: "vecgrids" });
      wrap.append(vgBox, h("p", { class: "note", text:
        "Grid: the force channels of the observation window the net reads at the video's current step (oldest row at the top; columns are Fx Fy Fz Tx Ty Tz, then max|F|, min Fz, max|T| over the control interval). "
        + "All zero when the force input is blinded. Red negative, black zero, green positive; hover for values." }));
    }
    container.replaceChildren(wrap);
    const fcg = fc.getContext("2d"), acg = ac.getContext("2d");
    let ep = eps[0], epIdx = 0, step = 0;
    const starts = [];
    { let acc = 0; for (const e of eps) { starts.push(acc); acc += VID_CARD + e.force.length + 1 + VID_HOLD; } }
    const fps = num(r.fps) || 20;
    function strip(g, series, names, cols, lim, title) {
      const last = ep.force.length - 1;
      g.fillStyle = css("--panel", "#fff"); g.fillRect(0, 0, W, HS);
      const tx = (i) => (i / Math.max(1, last)) * (W - 2) + 1, y = (v) => HS / 2 - (v / lim) * (HS / 2 - 8);
      g.strokeStyle = css("--line", "#e2e4e9"); g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, y(0)); g.lineTo(W, y(0)); g.stroke();
      series.forEach((k, j) => {
        g.beginPath();
        ep.force.forEach((_, i) => { const v = k(i); i ? g.lineTo(tx(i), y(Math.max(-lim, Math.min(lim, v)))) : g.moveTo(tx(i), y(Math.max(-lim, Math.min(lim, v)))); });
        g.strokeStyle = cols[j]; g.lineWidth = 1.3; g.stroke();
        g.fillStyle = cols[j]; g.font = "10px monospace"; g.fillText(names[j], 4 + 34 * j, 11);
      });
      g.fillStyle = css("--muted", "#6b7280"); g.fillText(`${title} (±${lim})`, W - 150, 11);
      g.strokeStyle = css("--ink", "#1f2430");
      g.beginPath(); g.moveTo(tx(step), 0); g.lineTo(tx(step), HS); g.stroke();
    }
    function draw() {
      step = Math.max(0, Math.min(step, ep.force.length - 1));
      const c = [css("--warn", "#b7791f"), css("--good", "#1a7f4b"), css("--accent", "#2f6fed"), css("--ink", "#1f2430")];
      const fl = ep.force.map((f) => Math.hypot(f[0], f[1], f[2]));
      const fl0 = Math.max(10, ...fl) * 1.1;
      strip(fcg, [(i) => ep.force[i][0], (i) => ep.force[i][1], (i) => ep.force[i][2], (i) => fl[i]], ["Fx", "Fy", "Fz", "|F|"], c, Math.min(VID_FABORT, Math.ceil(fl0)), "force N");
      strip(acg, [(i) => ep.act[i][0], (i) => ep.act[i][1], (i) => ep.act[i][2]], ["dx", "dy", "dz"], c, 5, "action mm");
      const f = ep.force[step];
      label.textContent = `${ep.outcome} · step ${step + 1}/${ep.force.length} · F = (${f[0].toFixed(1)}, ${f[1].toFixed(1)}, ${f[2].toFixed(1)}) N · peak ${ep.peak_F_N} N · recoveries ${ep.recoveries}`;
      if (vgBox && Array.isArray(ep.obs_force)) {
        const Hn = num(r.H) || 10, blind = !!r.force_zeroed, win = [];
        for (let i = step - Hn + 1; i <= step; i++) win.push(blind ? ep.obs_force[0].map(() => 0) : ep.obs_force[Math.max(0, i)]);
        if (gridEp !== ep) {
          vgBox.replaceChildren();
          obsG = window.LabVecGrid.render(vgBox, win, { title: blind ? "force obs window (blinded: zeros)" : `force obs window ${Hn}x9`, vmax: 10,
            labels: win.flatMap(() => ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "max|F|", "min Fz", "max|T|"]), cols: 9 });
          gridEp = ep;
        }
        obsG.update(win);
      }
    }
    if (hasVideo) {
      vid.addEventListener("timeupdate", () => {
        // follow the video: pick the episode whose frame range holds the current frame, then its step
        const f = Math.round(vid.currentTime * fps);
        let k = starts.length - 1;
        while (k > 0 && f < starts[k]) k--;
        if (k !== epIdx) { epIdx = k; ep = eps[k]; sel.value = String(k); }
        step = f - starts[k] - VID_CARD - 1;
        draw();
      });
    }
    sel.addEventListener("change", () => {
      epIdx = Number(sel.value); ep = eps[epIdx]; step = 0;
      if (hasVideo) vid.currentTime = (starts[epIdx] + VID_CARD) / fps;
      draw();
    });
    draw();
  }

  // ---- attention task: live heatmaps (training) and the inference stepper (evaluation) ------
  // Drawing helpers shared by attnlive, vitattn, fdlive and finddigit. paint(canvas, img, vals, grid): the
  // digit image (any size, 28x28 or find-digit's 28x56) in grey with the attention weights (vals,
  // grid[0]*grid[1] numbers, scaled to their own max) laid over it as translucent cells; vals null =
  // image only.
  function paintOverlay(cv, img, vals, grid) {
    const n = img.length, m = img[0].length, Sx = cv.width / m, Sy = cv.height / n, ctx2 = cv.getContext("2d");
    for (let y = 0; y < n; y++) for (let x = 0; x < m; x++) {
      const v = img[y][x];
      ctx2.fillStyle = `rgb(${v},${v},${v})`;
      ctx2.fillRect(x * Sx, y * Sy, Sx, Sy);
    }
    if (!vals) return;
    const [gh, gw] = grid, mx = Math.max(1e-9, ...vals), cw = cv.width / gw, ch = cv.height / gh;
    vals.forEach((a, i) => {
      ctx2.fillStyle = `rgba(255,90,20,${(0.8 * a / mx).toFixed(3)})`;
      ctx2.fillRect((i % gw) * cw, Math.floor(i / gw) * ch, cw, ch);
    });
    ctx2.strokeStyle = "rgba(255,255,255,0.25)";
    for (let i = 1; i < gw; i++) { ctx2.beginPath(); ctx2.moveTo(i * cw, 0); ctx2.lineTo(i * cw, cv.height); ctx2.stroke(); }
    for (let i = 1; i < gh; i++) { ctx2.beginPath(); ctx2.moveTo(0, i * ch); ctx2.lineTo(cv.width, i * ch); ctx2.stroke(); }
  }
  function overlayCanvas(size, height) {
    const cv = document.createElement("canvas"), hh = height || size;
    cv.width = size; cv.height = hh;
    cv.style.cssText = `width:${size}px;height:${hh}px;image-rendering:pixelated;border:1px solid var(--line);border-radius:4px`;
    return cv;
  }

  // attnlive(row, container, ctx): row.attn = [H][Tq][Tk] weights in 0..1 (one heatmap per head,
  //   vecgrid palette, vmax 1); row.qkv = {labels, heads: [{q, k, v}]} -> Q/K/V grids when
  //   ctx.vecgrids. When the row also has row.img ([28][28] uint8) and row.grid ([gh, gw]) and each
  //   map's first query row has 1 + gh*gw weights (CLS first), every map is drawn instead as the
  //   CLS row reshaped to the patch grid over the digit (row.names: per-map titles).
  //   The DOM is cached on container._live (tagged with this viewer's name) and updated in place.
  function attnlive(row, container, ctx) {
    const A = row && row.attn;
    if (!Array.isArray(A) || !Array.isArray(A[0]) || !Array.isArray(A[0][0]) || !window.LabVecGrid) {
      container._live = null;
      return jsonFallback(row, container);
    }
    const grid = Array.isArray(row.grid) && row.grid.length === 2 ? row.grid : null;
    const overlay = !!(grid && Array.isArray(row.img) && A[0][0].length === 1 + grid[0] * grid[1]);
    const qkv = row.qkv && Array.isArray(row.qkv.heads) ? row.qkv : null;
    const useQkv = !!(qkv && ctx && ctx.vecgrids);
    const labels = qkv && qkv.labels;
    const key = [A.length, A[0].length, A[0][0].length, useQkv ? qkv.heads.length : -1, overlay ? "img" : "grid"].join("x");
    let st = container._live;
    if (!st || st.viewer !== "attnlive" || st.key !== key) {
      const title = h("h3", { text: "" });
      const attnBox = h("div", overlay ? { class: "vecgrids", style: "display:flex;flex-wrap:wrap;gap:10px" } : { class: "vecgrids" });
      const qkvBox = h("div", { class: "vecgrids" });
      const note = h("p", { class: "note", text: overlay
        ? "Each picture is where the CLS token looks (one head of one block) over the probe digit; orange = more attention weight, scaled per picture."
        : "Each head's row is the query token's attention over the tokens (columns, left to right as in the probe); bright = more weight. Hover a cell for the token." });
      container.replaceChildren(title, attnBox, note, qkvBox);
      const lab = labels && labels.length === A[0][0].length ? labels : null;
      st = container._live = { viewer: "attnlive", key, title, attn: [], qkv: [] };
      A.forEach((m, i) => {
        const name = (row.names && row.names[i]) || `head ${i}`;
        if (overlay) {
          const cv = overlayCanvas(112);
          attnBox.append(h("div", {}, h("div", { class: "vg-title", text: name }), cv));
          st.attn.push({ update: (mm) => paintOverlay(cv, st.row.img, mm[0].slice(1), st.row.grid) });
        } else {
          st.attn.push(window.LabVecGrid.render(attnBox, m, { title: name, vmax: 1, labels: lab, cell: 22 }));
        }
      });
      if (useQkv) {
        qkv.heads.forEach((hd, i) => {
          st.qkv.push(["q", "k", "v"].map((n) => window.LabVecGrid.render(qkvBox, hd[n],
            { title: `head ${i} ${n.toUpperCase()}`, labels: undefined, cell: 12 })));
        });
      }
    }
    st.row = row;
    st.title.textContent = `live attention - probe example` + (row.step != null ? `, step ${row.step}` : "");
    A.forEach((m, i) => st.attn[i] && st.attn[i].update(m, 1));
    if (useQkv) qkv.heads.forEach((hd, i) => st.qkv[i] && ["q", "k", "v"].forEach((n, j) => st.qkv[i][j].update(hd[n])));
  }

  // vitattn(result, container, ctx): pick a digit, a block and a head; the CLS attention over the
  //   patch grid is laid over the digit (see task_vit.py for the schema).
  function vitattn(result, container, ctx) {
    const ex = result && result.examples, grid = result && result.grid;
    if (!Array.isArray(ex) || !ex.length || !Array.isArray(grid) || !Array.isArray(ex[0].attn)) return jsonFallback(result, container);
    const sel = (opts, label) => {
      const s2 = h("select", {});
      opts.forEach((t, i) => s2.append(h("option", { value: String(i), text: t })));
      return [h("label", { style: "margin-right:12px" }, document.createTextNode(label + " "), s2), s2];
    };
    const [exL, exS] = sel(ex.map((e, i) => `#${i}: label ${e.label}, pred ${e.pred}${e.pred === e.label ? "" : " (wrong)"}`), "digit");
    const [lyL, lyS] = sel(Array.from({ length: result.depth }, (_, i) => `block ${i}`), "block");
    const [hdL, hdS] = sel(Array.from({ length: result.heads }, (_, i) => `head ${i}`), "head");
    const big = overlayCanvas(280), plain = overlayCanvas(280), info = h("p", { class: "note", text: "" });
    const vg = h("div", { class: "vecgrids" });
    const row = h("div", { style: "display:flex;gap:16px;flex-wrap:wrap;margin:8px 0" },
      h("div", {}, h("div", { class: "vg-title", text: "digit" }), plain),
      h("div", {}, h("div", { class: "vg-title", text: "CLS attention over the patch grid" }), big));
    container.replaceChildren(
      h("div", { style: "margin-bottom:6px" }, chip("test accuracy", pct(result.acc)), chip("patch", `${result.patch}px`),
        chip("grid", `${grid[0]}x${grid[1]}`), chip("blocks", String(result.depth)), chip("heads", String(result.heads))),
      h("div", { style: "margin:6px 0" }, exL, lyL, hdL), row, info, vg);
    function draw() {
      const e = ex[Number(exS.value)], l = Number(lyS.value), hh = Number(hdS.value);
      const m = e.attn[l][hh].flat();
      paintOverlay(plain, e.pixels, null, grid);
      paintOverlay(big, e.pixels, m, grid);
      info.textContent = `predicted ${e.pred} with confidence ${pct(e.conf)} (label ${e.label}). ` +
        `This head puts ${pct(e.cls_self[l][hh])} of its weight on the CLS token itself and ${pct(m.reduce((a, b) => a + b, 0))} on the patches; ` +
        `the strongest patch gets ${pct(Math.max(...m))}.`;
      vg.replaceChildren();
      if (ctx && ctx.vecgrids && window.LabVecGrid && e.vec) {
        window.LabVecGrid.render(vg, e.vec.cls, { title: `CLS token (1 x ${e.vec.cls[0].length})`, cell: 10 });
        window.LabVecGrid.render(vg, e.vec.patches, { title: `patch embeddings (${e.vec.patches.length} x ${e.vec.patches[0].length}), row = patch in reading order`, cell: 10 });
      }
    }
    [exS, lyS, hdS].forEach((x) => x.addEventListener("change", draw));
    draw();
  }

  // fdlive(row, container, ctx): the find-digit live panel. row.attn = [H][1][gh*gw] (the query token's
  //   cross-attention row of every head over the patches, no CLS slot), row.img ([28][56] uint8),
  //   row.grid ([gh, gw]), row.query (digit class), row.names. One overlay canvas per head, painted in
  //   place; anything else falls back to JSON. DOM cached on container._live tagged "fdlive".
  function fdlive(row, container, ctx) {
    const A = row && row.attn, grid = row && Array.isArray(row.grid) && row.grid.length === 2 ? row.grid : null;
    if (!Array.isArray(A) || !Array.isArray(A[0]) || !Array.isArray(A[0][0]) || !grid || !Array.isArray(row.img)
      || A[0][0].length !== grid[0] * grid[1]) {
      container._live = null;
      return jsonFallback(row, container);
    }
    const key = [A.length, grid[0], grid[1], row.img.length, row.img[0].length].join("x");
    let st = container._live;
    if (!st || st.viewer !== "fdlive" || st.key !== key) {
      const title = h("h3", { text: "" });
      const box = h("div", { class: "vecgrids", style: "display:flex;flex-wrap:wrap;gap:10px" });
      const note = h("p", { class: "note", text: "Each picture is where the query token looks (one head) over the probe canvas; orange = more attention weight, scaled per picture. The query asks for one digit class; a trained head moves to the half that holds it." });
      container.replaceChildren(title, box, note);
      st = container._live = { viewer: "fdlive", key, title, cvs: [] };
      A.forEach((m, i) => {
        const cv = overlayCanvas(224, 112);
        box.append(h("div", {}, h("div", { class: "vg-title", text: (row.names && row.names[i]) || `head ${i}` }), cv));
        st.cvs.push(cv);
      });
    }
    st.title.textContent = `live attention - probe canvas, query digit ${row.query}` + (row.step != null ? `, step ${row.step}` : "");
    A.forEach((m, i) => st.cvs[i] && paintOverlay(st.cvs[i], row.img, m[0], grid));
  }

  // finddigit(result, container, ctx): pick a canvas and a head; the query's attention over the patch grid is laid
  //   over the canvas next to the plain canvas, with verdict chips (see task_finddigit.py for the schema).
  function finddigit(result, container, ctx) {
    const ex = result && result.examples, grid = result && result.grid;
    const ok = Array.isArray(ex) && ex.length && Array.isArray(grid) && grid.length === 2 &&
      ex.every((e) => Array.isArray(e.pixels) && Array.isArray(e.attn) && Array.isArray(e.attn[0]) && num(e.query) != null);
    if (!ok) return jsonFallback(result, container);
    const ANS = ["left", "right", "absent"];
    const sel = (opts, label) => {
      const s2 = h("select", {});
      opts.forEach((t, i) => s2.append(h("option", { value: String(i), text: t })));
      return [h("label", { style: "margin-right:12px" }, document.createTextNode(label + " "), s2), s2];
    };
    const [exL, exS] = sel(ex.map((e, i) => `#${i}: find ${e.query} -> ${ANS[e.pred]}${e.pred === e.truth ? "" : " (wrong)"}`), "canvas");
    const [hdL, hdS] = sel(ex[0].attn.map((_, i) => `head ${i}`), "head");
    const plain = overlayCanvas(336, 168), big = overlayCanvas(336, 168), info = h("p", { class: "note", text: "" });
    const chips = h("div", { style: "margin:6px 0" });
    const pics = h("div", { style: "display:flex;gap:16px;flex-wrap:wrap;margin:8px 0" },
      h("div", {}, h("div", { class: "vg-title", text: "canvas" }), plain),
      h("div", {}, h("div", { class: "vg-title", text: "where the query looks" }), big));
    const vg = h("div", { class: "vecgrids" });
    container.replaceChildren(
      h("div", { style: "margin-bottom:6px" }, chip("accuracy", pct(result.acc)), chip("absent cases", pct(result.absent_acc)),
        chip("noise", String(result.noise)), chip("heads", String(result.heads)), chip("blocks", String(result.depth))),
      h("div", { style: "margin:6px 0" }, exL, hdL), chips, pics, info, vg);
    function draw() {
      const e = ex[Number(exS.value)], hh = Number(hdS.value), m = e.attn[hh].flat();
      paintOverlay(plain, e.pixels, null, grid);
      paintOverlay(big, e.pixels, m, grid);
      const right = e.pred === e.truth;
      chips.replaceChildren(chip("query", String(e.query)), chip("truth", ANS[e.truth]), chip("predicted", ANS[e.pred]),
        chip("confidence", pct(e.conf)), chip(right ? "correct" : "wrong", right ? "yes" : "no"));
      const half = (c0, c1) => { let t = 0; m.forEach((v, i) => { const c = i % grid[1]; if (c >= c0 && c < c1) t += v; }); return t; };
      const hw = grid[1] / 2, L = half(0, hw), R = half(hw, grid[1]);
      info.textContent = `Digits on the canvas: ${e.classes ? e.classes.join(" and ") : "?"}. This head puts ${pct(L)} of its weight on the left half and ${pct(R)} on the right; ` +
        `the strongest patch gets ${pct(Math.max(...m))}.`;
      vg.replaceChildren();
      if (ctx && ctx.vecgrids && window.LabVecGrid && e.vec) {
        window.LabVecGrid.render(vg, e.vec.query, { title: `query token (1 x ${e.vec.query[0].length})`, cell: 10 });
        window.LabVecGrid.render(vg, e.vec.keys, { title: `K of the ${e.vec.keys.length} most attended patches (${e.vec.patches.join(", ")})`, cell: 10 });
      }
    }
    [exS, hdS].forEach((x) => x.addEventListener("change", draw));
    draw();
  }

  // attnstep(result, container, ctx): the stage slider over result.stages (see task_attention.py)
  function attnstep(result, container, ctx) {
    const stages = result && result.stages, probe = result && result.probe;
    if (!Array.isArray(stages) || !stages.length || !probe) return jsonFallback(result, container);
    const toks = probe.tokens.map((t) => t.label), T = toks.length;
    const TEXT = {
      embed: "Each pair token is its key embedding plus its value embedding; the query token is the key to look up plus a learned '?'.",
      qkv: "Three linear maps of the tokens: the query token asks (Q), every token advertises (K) and carries (V).",
      scores: "Score of token j for this query: q . k_j / sqrt(d_k) - how well the key matches the question.",
      softmax: "Softmax turns the scores into weights that sum to 1 per head.",
      mixture: "Weighted sum of the V vectors per head, concatenated, then mixed by the output projection.",
      readout: "A linear readout of the mixed vector gives a score for each of the possible values.",
    };
    const NAME = { embed: "1 embed", qkv: "2 Q / K / V", scores: "3 scores", softmax: "4 softmax", mixture: "5 mixture", readout: "6 readout" };
    const useVg = !!(ctx && ctx.vecgrids && window.LabVecGrid);
    const slider = h("input", { type: "range", min: "0", max: String(stages.length - 1), value: "0", step: "1", style: "width:260px" });
    const name = h("b", { text: "" });
    const body = h("div", { class: "vecgrids" });
    const tstrip = h("div", { style: "margin:6px 0" });
    toks.forEach((t, i) => tstrip.append(chip(i === T - 1 ? "query" : `token ${i}`, t)));
    const ok = probe.pred === probe.target;
    container.replaceChildren(
      h("div", { style: "margin-bottom:6px" }, chip("accuracy", pct(result.acc)), chip("pairs", String(result.n_pairs)),
        chip("heads", String(result.heads)), chip("d_k", String(result.d_k))),
      tstrip,
      h("div", { style: "display:flex;gap:12px;align-items:center;margin:6px 0" }, slider, name),
      h("p", { class: "note", text: "" }), body);
    const note = container.querySelector("p.note");
    const grid = (title, vals, opts) => { if (useVg) window.LabVecGrid.render(body, vals, Object.assign({ title }, opts || {})); };
    function bars(title, vals, vmax) { // one row per token: label, value, bar - works without vector grids
      const box = h("div", { class: "vecgrid" }, h("div", { class: "vg-title", text: title }));
      vals.forEach((v, i) => {
        const w = Math.max(0, Math.min(1, v / vmax)) * 100;
        box.append(h("div", { style: "display:flex;gap:6px;align-items:center;font:11px var(--mono)" },
          h("span", { style: "width:70px", text: toks[i] }),
          h("span", { style: `display:inline-block;height:8px;width:${w * 1.2}px;background:var(--accent)` }),
          h("span", { style: "color:var(--muted)", text: v.toFixed(3) })));
      });
      body.append(box);
    }
    function show(i) {
      const sg = stages[i];
      name.textContent = NAME[sg.name] || sg.name;
      note.textContent = TEXT[sg.name] || "";
      body.replaceChildren();
      if (sg.name === "embed") {
        grid(`tokens x d_model (${T} x ${sg.tokens[0].length})`, sg.tokens, { cell: 10, labels: undefined });
      } else if (sg.name === "qkv") {
        sg.q.forEach((q, hh) => {
          grid(`head ${hh} Q (1 x ${q[0].length})`, q, { cell: 12 });
          grid(`head ${hh} K (${T} x ${q[0].length})`, sg.k[hh], { cell: 12 });
          grid(`head ${hh} V (${T} x ${q[0].length})`, sg.v[hh], { cell: 12 });
        });
      } else if (sg.name === "scores") {
        const vm = Math.max(1e-6, ...sg.pre.flat().map(Math.abs));
        sg.pre.forEach((r, hh) => { grid(`head ${hh} scores`, [r], { cell: 22, labels: toks, vmax: vm }); bars(`head ${hh} score`, r, vm); });
      } else if (sg.name === "softmax") {
        sg.attn.forEach((r, hh) => { grid(`head ${hh} attention`, [r], { cell: 22, labels: toks, vmax: 1 }); bars(`head ${hh} weight`, r, 1); });
      } else if (sg.name === "mixture") {
        sg.per_head.forEach((r, hh) => grid(`head ${hh} sum(attn * V)`, [r], { cell: 12 }));
        grid(`concat (1 x ${sg.concat.length})`, [sg.concat], { cell: 8 });
        grid("after output projection", [sg.projected], { cell: 8 });
      } else if (sg.name === "readout") {
        grid("value probabilities", [sg.probs], { cell: 18, vmax: 1 });
        body.append(h("p", { text: `predicted v${sg.pred}, stored value v${sg.target} - ${ok ? "correct" : "wrong"}` }));
      } else {
        body.append(h("pre", { class: "json", text: JSON.stringify(sg, null, 2) }));
      }
      if (!useVg && ["embed", "qkv", "mixture"].includes(sg.name)) {
        body.append(h("p", { class: "note", text: "Turn on 'vector grids' to see the vectors of this stage." }));
      }
    }
    slider.addEventListener("input", () => show(Number(slider.value)));
    show(0);
  }

  // xattn(result, container, ctx): the cross-attention alignment of one translated date (see
  //   task_crossattn.py): output chars (rows) x source chars (columns), a head picker, and a decode
  //   stepper that outlines the row used at step t and shows what the decoder had generated so far.
  function xattn(result, container, ctx) {
    const ex = result && result.examples;
    const ok = Array.isArray(ex) && ex.length && ex.every((e) => Array.isArray(e.cross_attn) && Array.isArray(e.src_labels) && Array.isArray(e.out_labels) && Array.isArray(e.in_labels));
    if (!ok) return jsonFallback(result, container);
    const sel = (opts, label) => {
      const s2 = h("select", {});
      opts.forEach((t, i) => s2.append(h("option", { value: String(i), text: t })));
      return [h("label", { style: "margin-right:12px" }, document.createTextNode(label + " "), s2), s2];
    };
    const [exL, exS] = sel(ex.map((e, i) => `#${i}: ${e.source.trim()} -> ${e.decoded}${e.correct ? "" : " (wrong)"}`), "date");
    const [hdL, hdS] = sel(Array.from({ length: ex[0].cross_attn.length }, (_, i) => `head ${i}`), "head");
    const slider = h("input", { type: "range", min: "0", max: "0", value: "0", step: "1", style: "width:220px" });
    const stepLbl = h("b", { text: "" }), info = h("p", { class: "note", text: "" });
    const board = h("div", { style: "display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin:8px 0" });
    const vg = h("div", { class: "vecgrids" });
    // matrix(title, m, rowLabels, colLabels, hi): table of cells shaded with the vecgrid palette (weight 0..1);
    //   hi = index of the row to outline. Hover a cell for its weight.
    function matrix(title, m, rl, cl, hi) {
      const cell = "width:22px;height:22px;text-align:center;font:11px var(--mono);padding:0;";
      const t = h("table", { style: "border-collapse:collapse" });
      t.append(h("tr", {}, h("th", { style: cell }), ...cl.map((c) => h("th", { style: cell + "color:var(--muted);font-weight:400", text: c }))));
      m.forEach((r, i) => {
        const hl = i === hi;
        t.append(h("tr", {}, h("th", { style: cell + "color:" + (hl ? "var(--ink)" : "var(--muted)") + ";font-weight:" + (hl ? "700" : "400"), text: rl[i] }),
          ...r.map((v, j) => {
            const p = Math.max(0, Math.min(1, v)) * 100;
            return h("td", { title: `${rl[i]} -> ${cl[j]}: ${v.toFixed(3)}`, style: cell + `background:color-mix(in srgb, var(--vec-pos, #2a4) ${p.toFixed(1)}%, var(--vec-zero, #101114));` +
              (hl ? "border-top:2px solid var(--accent);border-bottom:2px solid var(--accent);" : "") });
          })));
      });
      return h("div", {}, h("div", { class: "vg-title", text: title }), t);
    }
    container.replaceChildren(
      h("div", { style: "margin-bottom:6px" }, chip("held-out exact-match", pct(result.exact)), chip("checked on", `${result.n_eval} dates`),
        chip("heads", String(result.heads)), chip("d_model", String(result.d_model))),
      h("div", { style: "margin:6px 0" }, exL, hdL),
      h("div", { style: "display:flex;gap:12px;align-items:center;margin:6px 0" }, h("span", { text: "decode step" }), slider, stepLbl),
      board, info, vg);
    function draw() {
      const e = ex[Number(exS.value)], hh = Number(hdS.value), L = e.out_labels.length;
      slider.max = String(L - 1);
      const t = Math.min(Number(slider.value), L - 1);
      slider.value = String(t);
      const row = e.cross_attn[hh][t], j = row.indexOf(Math.max(...row));
      stepLbl.textContent = `step ${t}: reads ${e.in_labels[t]} -> emits ${e.out_labels[t]}`;
      board.replaceChildren(
        matrix(`cross-attention, head ${hh}: output char (rows) x source char (columns)`, e.cross_attn[hh], e.out_labels, e.src_labels, t),
        matrix("decoder self-attention (causal)", e.dec_attn[hh], e.in_labels, e.in_labels, t),
        matrix("encoder self-attention", e.enc_attn[hh], e.src_labels, e.src_labels, -1));
      const so_far = e.out_labels.slice(0, t).join("");
      info.textContent = `source "${e.source}", target "${e.target}", decoded "${e.decoded}" (${e.correct ? "correct" : "wrong"}). ` +
        `Generated so far: "${so_far}". To emit ${e.out_labels[t]} this head puts ${pct(row[j])} of its weight on source char ${e.src_labels[j]} (column ${j}) ` +
        `and ${pct(row.reduce((a, b) => a + b, 0) - row[j])} on the rest.`;
      vg.replaceChildren();
      if (ctx && ctx.vecgrids && window.LabVecGrid && e.vec && e.vec.q && e.vec.q[hh]) {
        const dk = e.vec.q[hh][0].length;
        window.LabVecGrid.render(vg, e.vec.q[hh], { title: `head ${hh} Q: one row per output char (${L} x ${dk}), rows ${e.out_labels.join(" ")}`, cell: 12 });
        window.LabVecGrid.render(vg, e.vec.k[hh], { title: `head ${hh} K: one row per source char (${e.vec.k[hh].length} x ${dk}), rows ${e.src_labels.join(" ")}`, cell: 12 });
        window.LabVecGrid.render(vg, e.vec.v[hh], { title: `head ${hh} V: one row per source char (${e.vec.v[hh].length} x ${dk}), rows ${e.src_labels.join(" ")}`, cell: 12 });
      }
    }
    [exS, hdS].forEach((x) => x.addEventListener("change", () => { slider.value = "0"; draw(); }));
    slider.addEventListener("input", draw);
    draw();
  }

  window.LabViewers = { register: (name, fn) => { registry[name] = fn; }, get: (name) => registry[name], render, jsonFallback };
  window.LabViewers.register("t1scatter", t1scatter);
  window.LabViewers.register("traj2d", traj2d);
  window.LabViewers.register("t1", t1);
  window.LabViewers.register("flowtransport", flowtransport);
  window.LabViewers.register("stepsweep", stepsweep);
  window.LabViewers.register("flow", flow);
  window.LabViewers.register("contact2d", contact2d);
  window.LabViewers.register("embeddings", embeddings);
  window.LabViewers.register("tavla", tavla);
  window.LabViewers.register("video", video);
  window.LabViewers.register("attnlive", attnlive);
  window.LabViewers.register("attnstep", attnstep);
  window.LabViewers.register("vitattn", vitattn);
  window.LabViewers.register("xattn", xattn);
  window.LabViewers.register("fdlive", fdlive);
  window.LabViewers.register("finddigit", finddigit);
})();
