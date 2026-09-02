/** MPI-120 CFA path diagram — shared by big_five_sem.html and big_five_tsne.html */
(function (global) {
  const TRAIT_COLOR = {
    neuroticism: "var(--n)",
    extraversion: "var(--e)",
    conscientiousness: "var(--c)",
    openness: "var(--o)",
    agreeableness: "var(--a)",
  };

  const TRAIT_NAME = {
    neuroticism: "Neuroticism",
    extraversion: "Extraversion",
    conscientiousness: "Conscientiousness",
    openness: "Openness",
    agreeableness: "Agreeableness",
  };

  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    return node;
  }

  function polar(cx, cy, r, deg) {
    const rad = (deg * Math.PI) / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  }

  function fmtLoading(v) {
    const s = v < 0 ? "−" : "";
    return s + Math.abs(v).toFixed(2).replace(/^0/, "");
  }

  function fmtCorr(v) {
    const s = v < 0 ? "−" : "";
    return s + Math.abs(v).toFixed(2).replace(/^0/, "");
  }

  function midLabel(x1, y1, x2, y2, text, cls, offset = 0) {
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const nx = (-dy / len) * offset;
    const ny = (dx / len) * offset;
    return el("text", {
      x: mx + nx,
      y: my + ny,
      class: cls,
      "text-anchor": "middle",
      "dominant-baseline": "middle",
    }, text);
  }

  function arrowMarker(id, color, refX = "6") {
    const m = el("marker", {
      id,
      markerWidth: "7",
      markerHeight: "7",
      refX,
      refY: "3.5",
      orient: "auto",
      markerUnits: "strokeWidth",
    });
    m.appendChild(el("path", {
      d: "M0,0 L7,3.5 L0,7 Z",
      fill: color,
    }));
    return m;
  }

  function buildLayout(data) {
    const factorById = Object.fromEntries(data.factors.map((f) => [f.id, f]));
    const layout = data.layout;
    const nodes = { factors: {}, items: {}, errors: {} };

    for (const f of data.factors) {
      nodes.factors[f.id] = { ...f, trait: f.trait };
    }

    for (const [fid, cluster] of Object.entries(data.clusters)) {
      const factor = factorById[fid];
      const n = cluster.items.length;
      cluster.items.forEach((item, i) => {
        const t = n === 1 ? 0 : (i / (n - 1) - 0.5);
        const angle = cluster.baseAngle + t * cluster.spread;
        const [ix, iy] = polar(factor.x, factor.y, layout.itemDist, angle);
        const [ex, ey] = polar(factor.x, factor.y, layout.errorDist, angle);
        const key = String(item.item);
        nodes.items[key] = {
          item: item.item,
          factor: fid,
          loading: item.loading,
          x: ix,
          y: iy,
        };
        nodes.errors[key] = {
          item: item.item,
          x: ex,
          y: ey,
        };
      });
    }
    return nodes;
  }

  function pearson(xs, ys) {
    const n = Math.min(xs.length, ys.length);
    if (n < 2) return null;
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < n; i += 1) {
      sx += xs[i];
      sy += ys[i];
    }
    sx /= n;
    sy /= n;
    let num = 0;
    let dx2 = 0;
    let dy2 = 0;
    for (let i = 0; i < n; i += 1) {
      const dx = xs[i] - sx;
      const dy = ys[i] - sy;
      num += dx * dy;
      dx2 += dx * dx;
      dy2 += dy * dy;
    }
    const den = Math.sqrt(dx2 * dy2);
    if (den < 1e-8) return 0;
    return num / den;
  }

  function computeEvCorrelations(points, factors) {
    const usable = points.filter((p) => p.ev_scores && factors.every((f) => p.ev_scores[f.trait] != null));
    if (usable.length < 2) return null;
    const pairs = [];
    for (let i = 0; i < factors.length; i += 1) {
      for (let j = i + 1; j < factors.length; j += 1) {
        const xs = usable.map((p) => p.ev_scores[factors[i].trait]);
        const ys = usable.map((p) => p.ev_scores[factors[j].trait]);
        pairs.push({
          a: factors[i].id,
          b: factors[j].id,
          r: pearson(xs, ys),
          n: usable.length,
        });
      }
    }
    return pairs;
  }

  function renderFactorCorrelations(gCorr, nodes, pairs, options = {}) {
    while (gCorr.firstChild) gCorr.removeChild(gCorr.firstChild);
    const observed = options.observed === true;
    for (const { a, b, r } of pairs) {
      if (r == null || Number.isNaN(r)) continue;
      const fa = nodes.factors[a];
      const fb = nodes.factors[b];
      const mx = (fa.x + fb.x) / 2;
      const my = (fa.y + fb.y) / 2;
      const cx = 500 + (mx - 500) * 0.35;
      const cy = 430 + (my - 430) * 0.35;
      const mag = Math.abs(r);
      gCorr.appendChild(el("path", {
        d: `M ${fa.x} ${fa.y} Q ${cx} ${cy} ${fb.x} ${fb.y}`,
        fill: "none",
        stroke: observed ? "#7a7268" : "#c8bfb3",
        "stroke-width": String(observed ? 0.85 + mag * 2.2 : 1),
        opacity: observed ? String(0.55 + mag * 0.45) : "1",
        "marker-start": "url(#arrow-corr-start)",
        "marker-end": "url(#arrow-corr-end)",
      }));
      gCorr.appendChild(midLabel(
        fa.x, fa.y, fb.x, fb.y,
        fmtCorr(r),
        observed ? "corr-coef corr-observed" : "corr-coef",
        10,
      ));
    }
  }

  function setFactorLabel(labelEl, f, score) {
    while (labelEl.firstChild) labelEl.removeChild(labelEl.firstChild);
    labelEl.setAttribute("x", f.x);
    labelEl.setAttribute("y", f.y);
    labelEl.setAttribute("text-anchor", "middle");
    labelEl.setAttribute("dominant-baseline", "middle");

    const letter = document.createElementNS(NS, "tspan");
    letter.setAttribute("x", f.x);
    letter.setAttribute("dy", score != null ? "-0.45em" : "0");
    letter.textContent = f.label;
    labelEl.appendChild(letter);

    if (score != null) {
      const val = document.createElementNS(NS, "tspan");
      val.setAttribute("x", f.x);
      val.setAttribute("dy", "1.05em");
      val.setAttribute("font-size", "10");
      val.setAttribute("font-weight", "700");
      val.textContent = score.toFixed(2);
      labelEl.appendChild(val);
    }
  }

  function renderDiagram(data) {
    const nodes = buildLayout(data);
    const [vbX, vbY, vbW, vbH] = data.layout.viewBox;
    const svg = el("svg", {
      viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
      preserveAspectRatio: "xMidYMid meet",
    });
    svg.__semData = data;
    svg.__semNodes = nodes;

    const defs = el("defs");
    defs.appendChild(arrowMarker("arrow-ink", "#1a1612"));
    defs.appendChild(arrowMarker("arrow-muted", "#7a7268"));
    defs.appendChild(arrowMarker("arrow-corr-start", "#c8bfb3", "1"));
    defs.appendChild(arrowMarker("arrow-corr-end", "#c8bfb3", "6"));
    svg.appendChild(defs);

    const gCorr = el("g", { id: "correlations" });
    const gLoad = el("g", { id: "loadings" });
    const gErrCorr = el("g", { id: "error-correlations" });
    const gNodes = el("g", { id: "nodes" });

    renderFactorCorrelations(gCorr, nodes, data.factor_correlations, { observed: false });

    for (const item of Object.values(nodes.items)) {
      const f = nodes.factors[item.factor];
      const e = nodes.errors[String(item.item)];
      const color = TRAIT_COLOR[f.trait] || "#1a1612";

      gLoad.appendChild(el("line", {
        "data-factor-id": item.factor,
        x1: f.x, y1: f.y, x2: item.x, y2: item.y,
        stroke: color,
        "stroke-width": "1.25",
        "marker-end": "url(#arrow-ink)",
        opacity: "0.85",
      }));
      gLoad.appendChild(midLabel(f.x, f.y, item.x, item.y, fmtLoading(item.loading), "coef", 9));

      gLoad.appendChild(el("line", {
        x1: e.x, y1: e.y, x2: item.x, y2: item.y,
        stroke: "#c8bfb3",
        "stroke-width": "1",
        "marker-end": "url(#arrow-muted)",
      }));
    }

    for (const { a, b, r } of data.error_correlations) {
      const ea = nodes.errors[String(a)];
      const eb = nodes.errors[String(b)];
      if (!ea || !eb) continue;
      const mx = (ea.x + eb.x) / 2;
      const my = (ea.y + eb.y) / 2;
      const path = el("path", {
        d: `M ${ea.x} ${ea.y} Q ${mx} ${my - 28} ${eb.x} ${eb.y}`,
        fill: "none",
        stroke: "#b5a99a",
        "stroke-width": "1",
        "stroke-dasharray": "3 2",
      });
      gErrCorr.appendChild(path);
      gErrCorr.appendChild(midLabel(ea.x, ea.y, eb.x, eb.y, fmtCorr(r), "corr-coef", 12));
    }

    for (const f of Object.values(nodes.factors)) {
      const color = TRAIT_COLOR[f.trait];
      gNodes.appendChild(el("circle", {
        "data-factor-id": f.id,
        cx: f.x, cy: f.y, r: data.layout.factorRadius,
        fill: "var(--panel)",
        stroke: color,
        "stroke-width": "2",
      }));
      gNodes.appendChild(el("text", {
        "data-factor-label": f.id,
        x: f.x, y: f.y,
        class: "factor-label",
        fill: color,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
      }, f.label));
    }

    for (const item of Object.values(nodes.items)) {
      const half = data.layout.itemSize / 2;
      gNodes.appendChild(el("rect", {
        x: item.x - half, y: item.y - half,
        width: data.layout.itemSize, height: data.layout.itemSize,
        fill: "var(--panel)",
        stroke: "#1a1612",
        "stroke-width": "1",
        rx: "2",
      }));
      gNodes.appendChild(el("text", {
        x: item.x, y: item.y,
        class: "item-label",
        fill: "#1a1612",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
      }, String(item.item)));
    }

    for (const e of Object.values(nodes.errors)) {
      gNodes.appendChild(el("circle", {
        cx: e.x, cy: e.y, r: data.layout.errorRadius,
        fill: "var(--bg)",
        stroke: "#7a7268",
        "stroke-width": "1",
      }));
      gNodes.appendChild(el("text", {
        x: e.x, y: e.y + 0.5,
        class: "error-label",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
      }, "e" + e.item));
    }

    svg.appendChild(gCorr);
    svg.appendChild(gErrCorr);
    svg.appendChild(gLoad);
    svg.appendChild(gNodes);
    return svg;
  }

  function scoreNorm(score) {
    return Math.max(0, Math.min(1, (score - 1) / 4));
  }

  function applySemSelection(svg, selection) {
    const data = svg.__semData;
    const nodes = svg.__semNodes;
    if (!data || !nodes) return;

    const scores = selection?.scores || null;
    const meta = selection?.meta || null;
    const correlations = selection?.correlations || data.factor_correlations;
    const observed = Boolean(selection?.correlations);
    const steered = meta?.trait || null;

    svg.querySelectorAll(".score-overlay").forEach((n) => n.remove());

    const gNodes = svg.querySelector("#nodes");
    const gCorr = svg.querySelector("#correlations");
    const gLoad = svg.querySelector("#loadings");
    const baseR = data.layout.factorRadius;

    renderFactorCorrelations(gCorr, nodes, correlations, { observed });

    for (const line of gLoad.querySelectorAll("line[data-factor-id]")) {
      const fid = line.getAttribute("data-factor-id");
      const factor = nodes.factors[fid];
      const emphasized = steered && factor?.trait === steered;
      line.setAttribute("opacity", emphasized ? "1" : observed ? "0.35" : "0.85");
      line.setAttribute("stroke-width", emphasized ? "1.75" : "1.25");
    }

    for (const f of Object.values(nodes.factors)) {
      const circle = svg.querySelector(`circle[data-factor-id="${f.id}"]`);
      const label = svg.querySelector(`text[data-factor-label="${f.id}"]`);
      if (!circle || !label) continue;

      const score = scores?.[f.trait];
      const t = score != null ? scoreNorm(score) : null;
      const emphasized = steered === f.trait;

      circle.setAttribute("r", String(baseR * (t != null ? 0.88 + 0.28 * t : 1)));
      circle.setAttribute("stroke-width", emphasized ? "3.5" : "2");
      circle.setAttribute("fill-opacity", t != null ? String(0.45 + 0.5 * t) : "1");
      setFactorLabel(label, f, score);

      if (t != null) {
        gNodes.appendChild(el("text", {
          class: "score-overlay trait-name",
          x: f.x,
          y: f.y + baseR + 18,
          fill: TRAIT_COLOR[f.trait] || "#1a1612",
          "text-anchor": "middle",
          "font-size": "9",
          "font-weight": "600",
          opacity: emphasized ? "1" : "0.72",
        }, TRAIT_NAME[f.trait] || f.trait));
      }

      if (emphasized && t != null) {
        gNodes.appendChild(el("circle", {
          class: "score-overlay steer-halo",
          cx: f.x,
          cy: f.y,
          r: baseR + 10,
          fill: "none",
          stroke: TRAIT_COLOR[f.trait] || "#1a1612",
          "stroke-width": "1.5",
          opacity: "0.55",
        }));
      }
    }
  }

  function applyFactorScores(svg, scores, meta) {
    applySemSelection(svg, { scores, meta });
  }

  function attachPanZoom(stage, svg) {
    const base = svg.viewBox.baseVal;
    const baseW = base.width;
    const baseH = base.height;
    const vb = () => svg.viewBox.baseVal;
    let scale = 1;
    let tx = 0;
    let ty = 0;
    let drag = false;
    let lx = 0;
    let ly = 0;
    const PAN = 0.28;
    const ZOOM = 0.00055;

    function apply() {
      const b = vb();
      const cx = b.x + b.width / 2;
      const cy = b.y + b.height / 2;
      const w = baseW / scale;
      const h = baseH / scale;
      svg.setAttribute(
        "viewBox",
        `${cx - w / 2 + tx} ${cy - h / 2 + ty} ${w} ${h}`
      );
    }

    stage.addEventListener("wheel", (e) => {
      e.preventDefault();
      let delta = e.deltaY;
      if (e.deltaMode === 1) delta *= 16;
      else if (e.deltaMode === 2) delta *= baseH;
      const factor = Math.exp(-delta * ZOOM);
      scale = Math.min(3, Math.max(0.65, scale * factor));
      apply();
    }, { passive: false });

    stage.addEventListener("pointerdown", (e) => {
      drag = true;
      lx = e.clientX;
      ly = e.clientY;
      stage.classList.add("dragging");
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const rect = stage.getBoundingClientRect();
      const dx = ((e.clientX - lx) / rect.width) * (baseW / scale) * PAN;
      const dy = ((e.clientY - ly) / rect.height) * (baseH / scale) * PAN;
      tx -= dx;
      ty -= dy;
      lx = e.clientX;
      ly = e.clientY;
      apply();
    });
    const end = () => {
      drag = false;
      stage.classList.remove("dragging");
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
  }

  global.BigFiveSem = {
    TRAIT_COLOR,
    TRAIT_NAME,
    renderDiagram,
    computeEvCorrelations,
    applySemSelection,
    applyFactorScores,
    attachPanZoom,
  };
})(typeof window !== "undefined" ? window : globalThis);
