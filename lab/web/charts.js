/* makeCharts(container, chartsSpec) -> {push(rows), reset()}
 * chartsSpec: [{id, title, series: [{key, label, color?, dash?}], log_y?, ymin?, ymax?}, ...]
 * Rows are {"step": N, "<key>": v, ...}; rows sharing a step are merged; absent keys are null gaps. */
(function () {
  const COLORS = ["--chart1", "--chart2", "--chart3", "--loss"];

  function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function seriesColor(s, i) {
    if (s.color) return s.color.startsWith("--") ? cssColor(s.color, "#888") : s.color;
    return cssColor(COLORS[i % COLORS.length], "#888");
  }

  window.makeCharts = function makeCharts(container, chartsSpec) {
    container.replaceChildren();
    const charts = [];
    const steps = [];
    const byStep = new Map(); // step -> index in steps

    (chartsSpec || []).forEach((cs) => {
      const tile = document.createElement("div");
      tile.className = "chart";
      const h = document.createElement("h3");
      h.textContent = cs.title || "";
      const plot = document.createElement("div");
      plot.className = "plot";
      const leg = document.createElement("div");
      leg.className = "leg";
      tile.append(h, plot, leg);
      container.appendChild(tile);
      const series = cs.series || [];
      const cols = series.map(() => []);
      const colors = series.map(seriesColor);
      series.forEach((s, i) => {
        const span = document.createElement("span");
        const sw = document.createElement("i");
        sw.style.background = colors[i];
        span.append(sw, document.createTextNode(s.label || s.key));
        leg.appendChild(span);
      });
      const axisStroke = cssColor("--muted", "#888");
      const grid = { stroke: cssColor("--line", "#ddd"), width: 1 };
      const opts = {
        width: Math.max(plot.clientWidth || tile.clientWidth || 300, 200),
        height: 130,
        cursor: { show: true, drag: { x: false, y: false } },
        legend: { show: false },
        scales: { x: { time: false }, y: cs.log_y ? { distr: 3 } : {} },
        axes: [
          { stroke: axisStroke, grid, ticks: grid, size: 28, font: "10px JetBrains Mono, monospace" },
          { stroke: axisStroke, grid, ticks: grid, size: 46, font: "10px JetBrains Mono, monospace" },
        ],
        series: [{}].concat(
          series.map((s, i) => ({
            label: s.label || s.key,
            stroke: colors[i],
            width: 2,
            dash: s.dash,
            spanGaps: true,
            points: { show: false },
          }))
        ),
      };
      if (cs.ymin != null || cs.ymax != null) {
        opts.scales.y = Object.assign(opts.scales.y, { range: [cs.ymin ?? null, cs.ymax ?? null] });
      }
      const u = new uPlot(opts, [[]].concat(series.map(() => [])), plot);
      charts.push({ u, series, cols });
    });

    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => {
        charts.forEach((c) => {
          const w = c.u.root.parentElement.clientWidth;
          if (w > 0 && Math.abs(w - c.u.width) > 1) c.u.setSize({ width: w, height: 130 });
        });
      }).observe(container);
    }

    function redraw() {
      charts.forEach((c) => c.u.setData([steps].concat(c.cols)));
    }

    return {
      push(rows) {
        if (!rows || !rows.length) return;
        for (const r of rows) {
          if (r == null || typeof r.step !== "number") continue;
          let i = byStep.get(r.step);
          if (i === undefined) {
            i = steps.length;
            steps.push(r.step);
            byStep.set(r.step, i);
            charts.forEach((c) => c.cols.forEach((col) => col.push(null)));
          }
          charts.forEach((c) =>
            c.series.forEach((s, k) => {
              const v = r[s.key];
              if (typeof v === "number" && Number.isFinite(v)) c.cols[k][i] = v;
            })
          );
        }
        redraw();
      },
      reset() {
        steps.length = 0;
        byStep.clear();
        charts.forEach((c) => c.cols.forEach((col) => (col.length = 0)));
        redraw();
      },
    };
  };
})();
