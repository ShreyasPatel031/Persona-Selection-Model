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

  /** Direction each domain takes on the general evaluative factor (N reverse-keyed). */
  const GF_SIGN = {
    openness: 1,
    conscientiousness: 1,
    extraversion: 1,
    agreeableness: 1,
    neuroticism: -1,
  };

  /** Digman (1997) / DeYoung (2002) metatraits, drawn as second-order factors. */
  const METATRAITS = [
    {
      id: "ALPHA",
      label: "\u03b1",
      name: "Stability",
      color: "#5b4f8a",
      x: 645,
      y: 425,
      members: { conscientiousness: 1, agreeableness: 1, neuroticism: -1 },
    },
    {
      id: "BETA",
      label: "\u03b2",
      name: "Plasticity",
      color: "#2f5f9e",
      x: 318,
      y: 505,
      members: { openness: 1, extraversion: 1 },
    },
  ];

  function metatraitOf(trait) {
    return METATRAITS.find((m) => m.members[trait] != null) || null;
  }

  function sameMetatrait(a, b) {
    const ma = metatraitOf(a);
    const mb = metatraitOf(b);
    return Boolean(ma && mb && ma.id === mb.id);
  }

  /**
   * Sign a bipolar alpha-vs-beta contrast predicts for a domain pair. Same-metatrait
   * members co-vary in their keyed direction; cross-metatrait members oppose, because
   * once the general factor is removed alpha and beta sit at opposite ends of one axis.
   */
  function contrastSign(a, b) {
    const ma = metatraitOf(a);
    const mb = metatraitOf(b);
    if (!ma || !mb) return null;
    const prod = ma.members[a] * mb.members[b];
    return ma.id === mb.id ? prod : -prod;
  }

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

  function buildLayout(data, options = {}) {
    const factorById = Object.fromEntries(data.factors.map((f) => [f.id, f]));
    const layout = data.layout;
    const nodes = { factors: {}, items: {}, errors: {} };

    for (const f of data.factors) {
      nodes.factors[f.id] = { ...f, trait: f.trait };
    }

    // Support both schemas:
    //   legacy: data.clusters[fid] = { baseAngle, spread, items: [...] }
    //   new:    data.items[] + data.default_item_ids + layout.clusterAngles
    let clusters = data.clusters;
    if (!clusters && Array.isArray(data.items)) {
      const selected = new Set(
        options.itemIds
        || data.default_item_ids
        || data.items.map((it) => it.item)
      );
      const angles = layout.clusterAngles || {};
      clusters = {};
      for (const f of data.factors) {
        const angle = angles[f.id] || { baseAngle: 0, spread: 90 };
        clusters[f.id] = {
          baseAngle: angle.baseAngle,
          spread: angle.spread,
          items: data.items.filter(
            (it) => selected.has(it.item) && (it.domain_letter === f.id || it.domain === f.trait)
          ),
        };
      }
    }

    for (const [fid, cluster] of Object.entries(clusters || {})) {
      const factor = factorById[fid];
      if (!factor) continue;
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

  function zscore(vals) {
    const n = vals.length;
    const m = vals.reduce((s, v) => s + v, 0) / n;
    const sd = Math.sqrt(vals.reduce((s, v) => s + (v - m) * (v - m), 0) / n) || 1;
    return vals.map((v) => (v - m) / sd);
  }

  /** Residual of ys after removing the linear effect of xs. */
  function residualize(ys, xs) {
    const r = pearson(xs, ys);
    const mx = xs.reduce((s, v) => s + v, 0) / xs.length;
    const my = ys.reduce((s, v) => s + v, 0) / ys.length;
    const sdx = Math.sqrt(xs.reduce((s, v) => s + (v - mx) * (v - mx), 0) / xs.length) || 1;
    const sdy = Math.sqrt(ys.reduce((s, v) => s + (v - my) * (v - my), 0) / ys.length) || 1;
    const b = r * (sdy / sdx);
    return ys.map((v, i) => v - my - b * (xs[i] - mx));
  }

  /**
   * Pool every ladder run, then partial the general evaluative factor out of the
   * domain scores. Raw MPI-120 EVs load ~0.84 on a single factor, which swamps the
   * Digman metatrait contrast; the residual correlations expose it.
   */
  function computeEvStructure(points, factors) {
    const traits = factors.map((f) => f.trait);
    const usable = points.filter(
      (p) => p.ev_scores && traits.every((t) => p.ev_scores[t] != null)
    );
    if (usable.length < 3) return null;

    const z = {};
    for (const t of traits) z[t] = zscore(usable.map((p) => p.ev_scores[t]));

    // General factor: mean of sign-aligned domain z-scores.
    const gf = usable.map((_, i) =>
      traits.reduce((s, t) => s + GF_SIGN[t] * z[t][i], 0) / traits.length
    );

    const gfLoadings = {};
    for (const t of traits) gfLoadings[t] = pearson(gf, z[t]);
    const gfStrength =
      traits.reduce((s, t) => s + Math.abs(gfLoadings[t]), 0) / traits.length;

    const resid = {};
    for (const t of traits) resid[t] = residualize(z[t], gf);

    const idOf = Object.fromEntries(factors.map((f) => [f.trait, f.id]));
    const raw = [];
    const residual = [];
    for (let i = 0; i < traits.length; i += 1) {
      for (let j = i + 1; j < traits.length; j += 1) {
        const a = traits[i];
        const b = traits[j];
        const within = sameMetatrait(a, b);
        const predicted = contrastSign(a, b);
        raw.push({ a: idOf[a], b: idOf[b], r: pearson(z[a], z[b]), within, predicted });
        const rr = pearson(resid[a], resid[b]);
        residual.push({
          a: idOf[a],
          b: idOf[b],
          r: rr,
          within,
          predicted,
          match: predicted == null ? null : Math.sign(rr) === predicted,
        });
      }
    }

    // Second-order factors: composites of sign-weighted member domains.
    const composites = {};
    for (const m of METATRAITS) {
      const members = Object.keys(m.members).filter((t) => z[t]);
      if (!members.length) continue;
      composites[m.id] = usable.map((_, i) =>
        members.reduce((s, t) => s + m.members[t] * z[t][i], 0) / members.length
      );
    }

    const metatraits = METATRAITS.map((m) => ({
      ...m,
      loadings: Object.keys(m.members)
        .filter((t) => z[t])
        .map((t) => ({
          trait: t,
          id: idOf[t],
          // Signed so a reverse-keyed member (N on alpha) reads negative.
          loading: pearson(composites[m.id], z[t]),
        })),
    })).filter((m) => composites[m.id]);

    const [ma, mb] = METATRAITS;
    const metaCorr =
      composites[ma.id] && composites[mb.id]
        ? pearson(composites[ma.id], composites[mb.id])
        : null;

    // Score each pair against the alpha/beta contrast prediction: aligning by the
    // predicted sign makes a correct reverse-keyed pair count as agreement.
    const aligned = (list) => {
      const s = list.filter((p) => p.predicted != null);
      return s.length ? s.reduce((a, p) => a + p.r * p.predicted, 0) / s.length : null;
    };
    const matches = residual.filter((p) => p.match === true).length;
    const testable = residual.filter((p) => p.predicted != null).length;

    return {
      n: usable.length,
      raw,
      residual,
      metatraits,
      metaCorr,
      generalFactor: { loadings: gfLoadings, strength: gfStrength },
      summary: {
        rawAligned: aligned(raw),
        residAligned: aligned(residual),
        contrastMatches: matches,
        contrastTestable: testable,
      },
    };
  }

  function computeEvCorrelations(points, factors) {
    const s = computeEvStructure(points, factors);
    // Raw (not residual) so Digman partners stay the strongest |r| for a
    // selected trait — residual GF removal inflates cross-metatrait negatives.
    return s ? s.raw : null;
  }

  function corrStrokeWidth(r) {
    const mag = Math.min(1, Math.abs(r));
    return 0.75 + mag * 5.5;
  }

  const POS_CORR = "#3d8b6e";
  const NEG_CORR = "#c45c4a";
  const GRAY_CORR = "#c8bfb3";

  function factorIdForTrait(nodes, trait) {
    return Object.values(nodes.factors).find((f) => f.trait === trait)?.id || null;
  }

  function renderFactorCorrelations(gCorr, nodes, pairs, options = {}) {
    const observed = options.observed === true;
    const focusId = options.focusId || null;
    const keep = new Set();

    for (const pair of pairs) {
      const { a, b, r } = pair;
      if (r == null || Number.isNaN(r)) continue;
      const fa = nodes.factors[a];
      const fb = nodes.factors[b];
      if (!fa || !fb) continue;
      const key = `${a}-${b}`;
      keep.add(key);
      keep.add(`${key}-label`);
      const mx = (fa.x + fb.x) / 2;
      const my = (fa.y + fb.y) / 2;
      const cx = 500 + (mx - 500) * 0.35;
      const cy = 430 + (my - 430) * 0.35;
      const mag = Math.abs(r);
      const w = corrStrokeWidth(r);
      const within = pair.within != null
        ? pair.within
        : sameMetatrait(fa.trait, fb.trait);
      // Color the whole Digman clique for the selected metatrait (α has three
      // edges C–A, C–N, A–N — not just the two that touch the clicked node).
      const focusTrait = options.focusTrait || null;
      const focusMeta = focusTrait ? metatraitOf(focusTrait) : null;
      const inFocusMeta = Boolean(
        focusMeta
        && focusMeta.members[fa.trait] != null
        && focusMeta.members[fb.trait] != null
      );
      let stroke = GRAY_CORR;
      if (observed && within && (!focusTrait || inFocusMeta)) {
        stroke = r >= 0 ? POS_CORR : NEG_CORR;
      }
      const opacity = String(0.45 + mag * 0.45);

      let path = gCorr.querySelector(`[data-corr-key="${key}"]`);
      if (!path) {
        path = el("path", {
          "data-corr-key": key,
          d: `M ${fa.x} ${fa.y} Q ${cx} ${cy} ${fb.x} ${fb.y}`,
          fill: "none",
          "marker-start": "url(#arrow-corr-start)",
          "marker-end": "url(#arrow-corr-end)",
        });
        path.style.transition = "stroke-width 280ms ease, opacity 280ms ease, stroke 280ms ease";
        gCorr.appendChild(path);
      }
      path.setAttribute("stroke", stroke);
      path.setAttribute("stroke-width", String(w));
      path.setAttribute("opacity", opacity);

      let label = gCorr.querySelector(`[data-corr-key="${key}-label"]`);
      const labelText = fmtCorr(r);
      if (!label) {
        label = midLabel(fa.x, fa.y, fb.x, fb.y, labelText, "corr-coef", 10);
        label.setAttribute("data-corr-key", `${key}-label`);
        label.style.transition = "opacity 280ms ease, fill 280ms ease";
        gCorr.appendChild(label);
      } else {
        label.textContent = labelText;
      }
      const colored = observed && within && (!focusTrait || inFocusMeta);
      label.setAttribute("class", colored ? "corr-coef corr-observed" : "corr-coef");
      label.setAttribute("fill", colored ? stroke : "#7a7268");
      label.setAttribute("opacity", "1");
    }

    for (const child of [...gCorr.children]) {
      const key = child.getAttribute("data-corr-key");
      if (!key || !keep.has(key)) gCorr.removeChild(child);
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
      const hit = el("circle", {
        "data-factor-hit": f.id,
        "data-factor-trait": f.trait,
        cx: f.x, cy: f.y, r: data.layout.factorRadius + 8,
        fill: "transparent",
        stroke: "none",
      });
      hit.style.cursor = "pointer";
      gNodes.appendChild(hit);
      gNodes.appendChild(el("circle", {
        "data-factor-id": f.id,
        cx: f.x, cy: f.y, r: data.layout.factorRadius,
        fill: "var(--panel)",
        stroke: color,
        "stroke-width": "2",
        style: "pointer-events: none",
      }));
      const label = el("text", {
        "data-factor-label": f.id,
        x: f.x, y: f.y,
        class: "factor-label",
        fill: color,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        style: "pointer-events: none",
      }, f.label);
      gNodes.appendChild(label);
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
    const structure = selection?.structure || null;
    // Prefer raw correlations so Digman same-metatrait partners (e.g. E–O) stay
    // the thickest lines when a trait is focused.
    const correlations =
      selection?.correlations || structure?.raw || data.factor_correlations;
    const observed = Boolean(selection?.correlations || structure);
    const steered = meta?.trait || null;
    const focusId = steered ? factorIdForTrait(nodes, steered) : null;

    svg.querySelectorAll(".score-overlay").forEach((n) => n.remove());

    const gNodes = svg.querySelector("#nodes");
    const gCorr = svg.querySelector("#correlations");
    const gLoad = svg.querySelector("#loadings");
    const baseR = data.layout.factorRadius;

    renderFactorCorrelations(gCorr, nodes, correlations, {
      observed,
      focusId,
      focusTrait: steered,
    });

    for (const line of gLoad.querySelectorAll("line[data-factor-id]")) {
      const fid = line.getAttribute("data-factor-id");
      const factor = nodes.factors[fid];
      const emphasized = steered && factor?.trait === steered;
      line.setAttribute("opacity", emphasized ? "1" : observed ? "0.55" : "0.85");
      line.setAttribute("stroke-width", emphasized ? "1.75" : "1.25");
    }

    for (const f of Object.values(nodes.factors)) {
      const circle = svg.querySelector(`circle[data-factor-id="${f.id}"]`);
      const hit = svg.querySelector(`circle[data-factor-hit="${f.id}"]`);
      const label = svg.querySelector(`text[data-factor-label="${f.id}"]`);
      if (!circle || !label) continue;

      const score = scores?.[f.trait];
      const t = score != null ? scoreNorm(score) : null;
      const emphasized = steered === f.trait;

      circle.setAttribute("r", String(baseR * (t != null ? 0.88 + 0.28 * t : 1)));
      circle.setAttribute("stroke-width", emphasized ? "3.5" : "2");
      circle.setAttribute("fill-opacity", t != null ? String(0.45 + 0.5 * t) : "1");
      if (hit) {
        hit.setAttribute("r", String(baseR + 8));
        hit.style.cursor = "pointer";
      }
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
      if (e.target.closest?.("[data-factor-hit]")) return;
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

  function bindFactorSelect(svg, onSelect) {
    svg.__onFactorSelect = onSelect;
    if (svg.__factorSelectBound) return;
    svg.__factorSelectBound = true;
    svg.addEventListener("click", (e) => {
      const hit = e.target.closest?.("[data-factor-hit]");
      if (!hit) return;
      e.stopPropagation();
      const trait = hit.getAttribute("data-factor-trait");
      const id = hit.getAttribute("data-factor-hit");
      if (svg.__onFactorSelect && trait) svg.__onFactorSelect({ id, trait });
    });
  }

  global.BigFiveSem = {
    TRAIT_COLOR,
    TRAIT_NAME,
    METATRAITS,
    metatraitOf,
    renderDiagram,
    computeEvStructure,
    computeEvCorrelations,
    applySemSelection,
    applyFactorScores,
    attachPanZoom,
    bindFactorSelect,
    factorIdForTrait,
  };
})(typeof window !== "undefined" ? window : globalThis);
