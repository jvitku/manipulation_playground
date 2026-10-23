/* makeCharts(container, chartsSpec) -> {push(rows), reset()}
 * chartsSpec: [{id, title, series: [{key, label, color?, dash?}], log_y?, ymin?, ymax?}, ...]
 * Rows are {"step": N, "<key>": v, ...}; rows sharing a step are merged; absent keys are null gaps.
 * Optional per-chart `x_key`: the x-axis is that row field instead of "step" (rows without it are
 * skipped for that chart), so phases of one run can be overlaid on a per-phase step axis. Rows that
 * share an x value merge; they carry different series keys per phase, so nothing is overwritten. */
(function () {
  const COLORS = ["--chart1", "--chart2", "--chart3", "--loss"];

  function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  // compact tick label: 2-3 significant digits, k/M suffixes, no trailing zeros (0.5, 12, 2k, 0.012, 3e-4)
  function fmtTick(v) {
    if (v == null || !Number.isFinite(v)) return "";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
    if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
    if (a < 1e-3) return v.toExponential(0);
    return String(+v.toPrecision(a < 1 ? 2 : 3));
  }

  function seriesColor(s, i) {
    if (s.color) return s.color.startsWith("--") ? cssColor(s.color, "#888") : s.color;
    return cssColor(COLORS[i % COLORS.length], "#888");
  }

  window.makeCharts = function makeCharts(container, chartsSpec) {
    container.replaceChildren();
    const charts = [];

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
          {
            stroke: axisStroke, grid, ticks: grid, size: 42, space: 28, gap: 3,
            font: "10px JetBrains Mono, monospace", values: (u, vals) => vals.map(fmtTick),
          },
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
      charts.push({ u, series, cols, xKey: cs.x_key || "step", xs: [], byX: new Map() });
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
      charts.forEach((c) => c.u.setData([c.xs].concat(c.cols)));
    }

    return {
      push(rows) {
        if (!rows || !rows.length) return;
        for (const r of rows) {
          if (r == null) continue;
          for (const c of charts) {
            const x = r[c.xKey];
            if (typeof x !== "number") continue;
            let i = c.byX.get(x);
            if (i === undefined) {
              // keep the x-axis ascending: an overlaid phase restarts at a low x, so insert in order
              i = c.xs.length;
              while (i > 0 && c.xs[i - 1] > x) i--;
              c.xs.splice(i, 0, x);
              c.cols.forEach((col) => col.splice(i, 0, null));
              if (i < c.xs.length - 1) c.xs.forEach((xv, j) => c.byX.set(xv, j));
              else c.byX.set(x, i);
            }
            c.series.forEach((s, k) => {
              const v = r[s.key];
              if (typeof v === "number" && Number.isFinite(v)) c.cols[k][i] = v;
            });
          }
        }
        redraw();
      },
      reset() {
        charts.forEach((c) => {
          c.xs.length = 0;
          c.byX.clear();
          c.cols.forEach((col) => (col.length = 0));
        });
        redraw();
      },
    };
  };
})();
