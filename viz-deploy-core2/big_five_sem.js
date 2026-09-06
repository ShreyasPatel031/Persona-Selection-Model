/** MPI-120 force-directed measurement diagram — shared by big_five_sem.html and big_five_tsne.html */
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

  const GF_SIGN = {
    openness: 1,
    conscientiousness: 1,
    extraversion: 1,
    agreeableness: 1,
    neuroticism: -1,
  };

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

  function contrastSign(a, b) {
    const ma = metatraitOf(a);
    const mb = metatraitOf(b);
    if (!ma || !mb) return null;
    const prod = ma.members[a] * mb.members[b];
    return ma.id === mb.id ? prod : -prod;
  }

  const NS = "http://www.w3.org/2000/svg";
  const POS_CORR = "#3d8b6e";
  const NEG_CORR = "#c45c4a";
  const TOP_N_DEFAULT = 20;
  const TOP_N_FOCUSED = 8;

  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    return node;
  }

  function htmlEl(tag, attrs = {}, text) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "className") node.className = v;
      else node.setAttribute(k, v);
    }
    if (text != null) node.textContent = text;
    return node;
  }

  function fmtLoading(v) {
    const s = v < 0 ? "−" : "";
    return s + Math.abs(v).toFixed(2).replace(/^0/, "");
  }

  function fmtCorr(v) {
    const s = v < 0 ? "−" : "";
    return s + Math.abs(v).toFixed(2).replace(/^0/, "");
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

  function factorIdForTrait(nodesOrFactors, trait) {
    if (Array.isArray(nodesOrFactors)) {
      return nodesOrFactors.find((f) => f.trait === trait)?.id || null;
    }
    if (nodesOrFactors?.factors) {
      return Object.values(nodesOrFactors.factors).find((f) => f.trait === trait)?.id || null;
    }
    return null;
  }

  function resolveVisibleItems(data, trait) {
    const all = data.items || [];
    const defaultN = data.default_top_n || TOP_N_DEFAULT;
    const focusN = data.focus_top_n || TOP_N_FOCUSED;
    if (!trait) {
      if (data.default_item_ids?.length) {
        const keep = new Set(data.default_item_ids.slice(0, defaultN));
        const selected = all.filter((it) => keep.has(it.item));
        // Preserve rank order from default_item_ids
        const rank = new Map(data.default_item_ids.map((id, i) => [id, i]));
        selected.sort((a, b) => (rank.get(a.item) ?? 999) - (rank.get(b.item) ?? 999));
        return selected.slice(0, defaultN);
      }
      return [...all]
        .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
        .slice(0, defaultN);
    }
    return all
      .filter((it) => it.domain === trait)
      .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
      .slice(0, focusN);
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

  function residualize(ys, xs) {
    const r = pearson(xs, ys);
    const mx = xs.reduce((s, v) => s + v, 0) / xs.length;
    const my = ys.reduce((s, v) => s + v, 0) / ys.length;
    const sdx = Math.sqrt(xs.reduce((s, v) => s + (v - mx) * (v - mx), 0) / xs.length) || 1;
    const sdy = Math.sqrt(ys.reduce((s, v) => s + (v - my) * (v - my), 0) / ys.length) || 1;
    const b = r * (sdy / sdx);
    return ys.map((v, i) => v - my - b * (xs[i] - mx));
  }

  function computeEvStructure(points, factors) {
    const traits = factors.map((f) => f.trait);
    const usable = points.filter(
      (p) => p.ev_scores && traits.every((t) => p.ev_scores[t] != null)
    );
    if (usable.length < 3) return null;

    const z = {};
    for (const t of traits) z[t] = zscore(usable.map((p) => p.ev_scores[t]));

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
          loading: pearson(composites[m.id], z[t]),
        })),
    })).filter((m) => composites[m.id]);

    const [ma, mb] = METATRAITS;
    const metaCorr =
      composites[ma.id] && composites[mb.id]
        ? pearson(composites[ma.id], composites[mb.id])
        : null;

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
    return s ? s.raw : null;
  }

  function corrStrokeWidth(r) {
    const mag = Math.min(1, Math.abs(r));
    return 0.75 + mag * 5.5;
  }

  const GRAY_CORR = "#c8bfb3";

  function factorTraitById(data, id) {
    return data.factors.find((f) => f.id === id)?.trait || null;
  }

  /** Digman clique for the focused trait: color all α edges if N/A/C selected, etc. */
  function corrStrokeStyle(data, a, b, r, observed, focusTrait) {
    const ta = factorTraitById(data, a);
    const tb = factorTraitById(data, b);
    const within = ta && tb && sameMetatrait(ta, tb);
    const focusMeta = focusTrait ? metatraitOf(focusTrait) : null;
    const inFocusMeta = Boolean(
      focusMeta
      && focusMeta.members[ta] != null
      && focusMeta.members[tb] != null
    );
    const colored = observed && within && (!focusTrait || inFocusMeta);
    return {
      stroke: colored ? (r >= 0 ? POS_CORR : NEG_CORR) : GRAY_CORR,
      colored,
      within,
    };
  }

  function ensureD3() {
    if (global.d3?.forceSimulation) return Promise.resolve(global.d3);
    return new Promise((resolve, reject) => {
      const existing = document.querySelector("script[data-bigfive-d3]");
      if (existing) {
        existing.addEventListener("load", () => resolve(global.d3));
        existing.addEventListener("error", () => reject(new Error("d3 load failed")));
        return;
      }
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js";
      s.async = true;
      s.dataset.bigfiveD3 = "1";
      s.onload = () => resolve(global.d3);
      s.onerror = () => reject(new Error("d3 load failed"));
      document.head.appendChild(s);
    });
  }

  function scoreNorm(score) {
    return Math.max(0, Math.min(1, (score - 1) / 4));
  }

  function setFactorLabel(labelEl, f, score) {
    while (labelEl.firstChild) labelEl.removeChild(labelEl.firstChild);
    labelEl.setAttribute("x", 0);
    labelEl.setAttribute("y", 0);
    labelEl.setAttribute("text-anchor", "middle");
    labelEl.setAttribute("dominant-baseline", "middle");

    const letter = document.createElementNS(NS, "tspan");
    letter.setAttribute("x", 0);
    letter.setAttribute("dy", score != null ? "-0.55em" : "0");
    letter.textContent = f.label;
    labelEl.appendChild(letter);

    if (score != null) {
      const val = document.createElementNS(NS, "tspan");
      val.setAttribute("x", 0);
      val.setAttribute("dy", "1.35em");
      val.setAttribute("font-size", "10");
      val.setAttribute("font-weight", "700");
      val.textContent = score.toFixed(2);
      labelEl.appendChild(val);
    }
  }

  function ensureTooltip(stage) {
    let tip = document.getElementById("cross-tip");
    if (tip) return tip;
    tip = stage?.querySelector?.(".sem-tooltip");
    if (tip) return tip;
    tip = htmlEl("div", { className: "sem-tooltip", id: "cross-tip", hidden: "true" });
    (stage || document.body).appendChild(tip);
    return tip;
  }

  function positionTip(tip, stage, clientX, clientY) {
    tip.hidden = false;
    const pad = 12;
    const tw = tip.offsetWidth || 260;
    const th = tip.offsetHeight || 150;
    let x = clientX + 14;
    let y = clientY + 14;
    if (x + tw > window.innerWidth - pad) x = clientX - tw - 12;
    if (y + th > window.innerHeight - pad) y = clientY - th - 12;
    tip.style.left = `${Math.max(pad, x)}px`;
    tip.style.top = `${Math.max(pad, y)}px`;
  }

  function bindHoverTip(g, tip, stage, htmlFn, hoverPayload) {
    if (!tip) return;
    g.style.cursor = g.style.cursor || "help";
    g.addEventListener("pointerenter", (e) => {
      tip.innerHTML = htmlFn();
      positionTip(tip, stage, e.clientX, e.clientY);
      const cb = g.ownerSVGElement?.__onHover;
      if (cb) cb(hoverPayload || null);
    });
    g.addEventListener("pointermove", (e) => {
      if (tip.hidden) tip.innerHTML = htmlFn();
      positionTip(tip, stage, e.clientX, e.clientY);
    });
    g.addEventListener("pointerleave", () => {
      hideTooltip(tip);
      const cb = g.ownerSVGElement?.__onHover;
      if (cb) cb(null);
    });
  }

  const TRAIT_BLURB = {
    neuroticism: "Tendency toward negative emotion, worry, and emotional reactivity. Digman α (Stability) reverse-keys this pole.",
    extraversion: "Sociability, assertiveness, and energetic engagement with the world. Digman β (Plasticity) member.",
    conscientiousness: "Order, dutifulness, and goal-directed self-control. Digman α (Stability) member.",
    openness: "Imagination, intellect, and preference for novelty/complexity. Digman β (Plasticity) member.",
    agreeableness: "Cooperation, trust, and concern for others. Digman α (Stability) member.",
  };

  function showItemTooltip(tip, stage, clientX, clientY, item, factor) {
    const domain = TRAIT_NAME[item.domain] || item.domain;
    const keyCls = item.key < 0 ? "tip-key-rev" : "tip-key-pos";
    const keyNote = item.key < 0 ? "reverse-keyed" : "positively keyed";
    const letter = factor?.label || "?";
    const lam = fmtLoading(item.loading);
    const steerRow = item.loading_steer != null
      ? `<div class="tip-row"><span class="tip-label">steer corr</span><span class="tip-value">${fmtLoading(item.loading_steer)}</span></div>`
      : "";
    tip.innerHTML =
      `<div class="tip-kicker">${domain} · ${item.facet}</div>` +
      `<div class="tip-title">Item ${item.item}</div>` +
      `<div class="tip-body">“${item.text}”</div>` +
      `<div class="tip-divider"></div>` +
      `<div class="tip-row"><span class="tip-label">keying</span><span class="${keyCls}">${keyNote}</span></div>` +
      `<div class="tip-row"><span class="tip-label">corr</span><span class="tip-value tip-lambda">${lam}</span></div>` +
      `<div class="tip-row tip-explain">Pearson r(keyed item EV, ${letter} domain EV)</div>` +
      steerRow;
    positionTip(tip, stage, clientX, clientY);
  }

  function factorTooltipHtml(n) {
    const meta = metatraitOf(n.trait);
    const partners = meta
      ? Object.keys(meta.members)
        .filter((t) => t !== n.trait)
        .map((t) => `${TRAIT_NAME[t]}${meta.members[t] < 0 ? " (−)" : ""}`)
        .join(", ")
      : "—";
    return (
      `<div class="tip-kicker">Big Five domain</div>` +
      `<div class="tip-title">${n.label} · ${TRAIT_NAME[n.trait]}</div>` +
      `<div class="tip-body">${TRAIT_BLURB[n.trait] || ""}</div>` +
      `<div class="tip-meta">Digman ${meta?.label || "?"} ${meta?.name || ""} · partners: ${partners}<br/>Click to focus top-8 items</div>`
    );
  }

  function digmanTooltipHtml(n) {
    const members = (n.members || [])
      .map((t) => {
        const m = METATRAITS.find((x) => x.id === n.metaId);
        const sign = m?.members[t] < 0 ? "−" : "+";
        return `${TRAIT_NAME[t]} (${sign})`;
      })
      .join(" · ");
    const blurb = n.metaId === "ALPHA"
      ? "Stability: shared covariance among C+, A+, and N− (emotional/motivational control)."
      : "Plasticity: shared covariance among E+ and O+ (exploration and engagement).";
    return (
      `<div class="tip-kicker">Digman metatrait</div>` +
      `<div class="tip-title">${n.label} · ${n.name}</div>` +
      `<div class="tip-body">${blurb}</div>` +
      `<div class="tip-meta">Members: ${members}<br/>Appears on trait selection · dashed edge = reverse-keyed member</div>`
    );
  }

  const TRAIT_LETTER = {
    openness: "O",
    conscientiousness: "C",
    extraversion: "E",
    agreeableness: "A",
    neuroticism: "N",
  };

  // Radar axis order (clockwise from top): O → C → E → A → N
  const RADAR_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
  ];
  const EV_SCALE_MIN = 1;
  const EV_SCALE_MAX = 5;
  const EV_MID = 3;
  let EV_BASELINE = null;

  function setEvBaseline(scores) {
    EV_BASELINE = scores ? { ...scores } : null;
  }

  function defaultEvBaseline() {
    const out = {};
    for (const t of RADAR_TRAITS) out[t] = EV_MID;
    return out;
  }

  function resolveEvBaseline() {
    return EV_BASELINE || defaultEvBaseline();
  }

  function clampEv(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return EV_MID;
    return Math.min(EV_SCALE_MAX, Math.max(EV_SCALE_MIN, n));
  }

  function radarPoint(cx, cy, rMax, i, n, value) {
    const t = (value - EV_SCALE_MIN) / (EV_SCALE_MAX - EV_SCALE_MIN);
    const r = Math.max(0, Math.min(1, t)) * rMax;
    const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
    return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
  }

  function radarPolygon(cx, cy, rMax, scores) {
    return RADAR_TRAITS.map((t, i) => {
      const [x, y] = radarPoint(cx, cy, rMax, i, RADAR_TRAITS.length, clampEv(scores?.[t]));
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }

  function radarChartSvg(scores, baseline, focusTrait) {
    const W = 168;
    const H = 158;
    const cx = W / 2;
    const cy = H / 2 + 2;
    const rMax = 52;
    const n = RADAR_TRAITS.length;
    const base = baseline || defaultEvBaseline();
    const color = TRAIT_COLOR[focusTrait] || "var(--ink)";
    const focusMeta = focusTrait ? metatraitOf(focusTrait) : null;
    const digmanSet = new Set(
      focusMeta ? Object.keys(focusMeta.members) : []
    );

    const rings = [0.25, 0.5, 0.75, 1].map((f) => {
      const pts = RADAR_TRAITS.map((_, i) => {
        const [x, y] = radarPoint(cx, cy, rMax * f, i, n, EV_SCALE_MAX);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      return `<polygon points="${pts}" class="tip-radar-ring" />`;
    }).join("");

    const spokes = RADAR_TRAITS.map((_, i) => {
      const [x, y] = radarPoint(cx, cy, rMax, i, n, EV_SCALE_MAX);
      return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="tip-radar-spoke" />`;
    }).join("");

    const midPts = radarPolygon(cx, cy, rMax, defaultEvBaseline());
    const basePts = radarPolygon(cx, cy, rMax, base);
    const livePts = radarPolygon(cx, cy, rMax, scores);

    const labels = RADAR_TRAITS.map((t, i) => {
      const [x, y] = radarPoint(cx, cy, rMax + 14, i, n, EV_SCALE_MAX);
      const live = clampEv(scores?.[t]);
      const ref = clampEv(base?.[t]);
      const d = live - ref;
      const dig = digmanSet.has(t);
      const focus = t === focusTrait ? " tip-radar-label-focus" : "";
      const digCls = dig ? " tip-radar-label-digman" : "";
      const rev = focusMeta?.members?.[t] < 0 ? "−" : "";
      const delta = Math.abs(d) >= 0.12
        ? `<tspan class="tip-radar-delta" dy="1.1em" x="${x.toFixed(1)}">${d >= 0 ? "+" : ""}${d.toFixed(1)}</tspan>`
        : "";
      const fill = dig && focusMeta ? ` style="fill:${focusMeta.color}"` : "";
      return (
        `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle" dominant-baseline="middle" ` +
        `class="tip-radar-label${focus}${digCls}"${fill}>${TRAIT_LETTER[t]}${rev}${delta}</text>`
      );
    }).join("");

    return (
      `<svg class="tip-radar" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true">` +
        `<g class="tip-radar-grid">${rings}${spokes}</g>` +
        `<polygon points="${midPts}" class="tip-radar-mid" />` +
        `<polygon points="${basePts}" class="tip-radar-base" />` +
        `<polygon points="${livePts}" class="tip-radar-live" style="stroke:${color};fill:${color}" />` +
        labels +
      `</svg>`
    );
  }

  function pointTooltipHtml(p) {
    const meta = metatraitOf(p.trait);
    const letter = TRAIT_LETTER[p.trait] || "?";
    const baseline = resolveEvBaseline();
    const hasScores = p.ev_scores && RADAR_TRAITS.every((t) => p.ev_scores[t] != null);
    const partners = meta
      ? Object.keys(meta.members)
          .map((t) => `${TRAIT_LETTER[t]}${meta.members[t] < 0 ? "−" : ""}`)
          .join(" · ")
      : "";
    const radar = hasScores
      ? (
          `<div class="tip-radar-wrap">${radarChartSvg(p.ev_scores, baseline, p.trait)}</div>` +
          `<div class="tip-radar-legend">` +
            `<span><i class="swatch swatch-live" style="background:${TRAIT_COLOR[p.trait] || "var(--ink)"}"></i>this persona</span>` +
            `<span><i class="swatch swatch-base"></i>pool base</span>` +
            `<span><i class="swatch swatch-mid"></i>scale mid (3)</span>` +
            (meta
              ? `<span><i class="swatch swatch-digman" style="background:${meta.color}"></i>${meta.label} ${partners}</span>`
              : "") +
          `</div>`
        )
      : `<div class="tip-meta">No full EV profile for this point.</div>`;
    return (
      `<div class="tip-kicker">t-SNE · ${p.kind}${p.level != null ? ` · L${p.level}` : ""}</div>` +
      `<div class="tip-title">${letter} · ${TRAIT_NAME[p.trait]}</div>` +
      `<div class="tip-body">${TRAIT_BLURB[p.trait] || ""}</div>` +
      `<div class="tip-divider"></div>` +
      radar +
      `<div class="tip-meta">Digman ${meta?.label || "?"} ${meta?.name || ""} · ${p.id}</div>`
    );
  }

  function hideTooltip(tip) {
    if (tip) tip.hidden = true;
  }

  function resolveHoverMeta(link, itemById = {}) {
    if (!link) return { trait: null, meta: null };
    if (link.metaId) {
      return {
        trait: link.trait || null,
        meta: METATRAITS.find((m) => m.id === link.metaId) || null,
      };
    }
    const trait = link.trait
      || (link.item != null ? itemById[String(link.item)]?.domain : null)
      || null;
    return { trait, meta: trait ? metatraitOf(trait) : null };
  }

  function clearHoverDigman(svg) {
    svg?.querySelector?.("#hover-digman")?.remove();
  }

  /** Shorten a hub→node spoke so it meets circle rims (not centers). */
  function trimSpoke(x1, y1, r1, x2, y2, r2, endPad = 3) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.hypot(dx, dy) || 1;
    const ux = dx / dist;
    const uy = dy / dist;
    const start = Math.min(r1, dist * 0.45);
    const end = Math.min(r2 + endPad, dist * 0.45);
    return {
      x1: x1 + ux * start,
      y1: y1 + uy * start,
      x2: x2 - ux * end,
      y2: y2 - uy * end,
      mx: (x1 + ux * start + x2 - ux * end) / 2,
      my: (y1 + uy * start + y2 - uy * end) / 2,
    };
  }

  function paintHoverDigman(svg, meta, focusTrait) {
    clearHoverDigman(svg);
    if (!svg || !meta) return;
    // Real Digman hub already present from selection — just leave it.
    if (svg.querySelector(`#nodes [data-node-id="digman:${meta.id}"]`)) return;

    const gNodes = svg.querySelector("#nodes");
    if (!gNodes) return;
    const data = svg.__semData;
    const factorR = data?.layout?.factorRadius || 40;
    const digR = 30;
    const members = (data?.factors || []).filter((f) => meta.members[f.trait] != null);
    const positions = [];
    for (const f of members) {
      const g = gNodes.querySelector(`[data-node-id="factor:${f.id}"]`);
      if (!g) continue;
      const tr = g.getAttribute("transform") || "";
      const m = /translate\(([^,]+),([^)]+)\)/.exec(tr);
      if (!m) continue;
      positions.push({ f, x: +m[1], y: +m[2] });
    }
    if (!positions.length) return;

    const cx = positions.reduce((s, p) => s + p.x, 0) / positions.length;
    const cy = positions.reduce((s, p) => s + p.y, 0) / positions.length + 70;
    const layer = el("g", {
      id: "hover-digman",
      class: "hover-match",
      "pointer-events": "none",
    });

    for (const p of positions) {
      const signed = meta.members[p.f.trait];
      const spoke = trimSpoke(cx, cy, digR, p.x, p.y, factorR, 4);
      const lineAttrs = {
        x1: String(spoke.x1),
        y1: String(spoke.y1),
        x2: String(spoke.x2),
        y2: String(spoke.y2),
        stroke: meta.color,
        "stroke-width": p.f.trait === focusTrait ? "2.4" : "1.6",
        opacity: "0.95",
        "marker-end": "url(#arrow-ink)",
        class: "hover-match",
      };
      if (signed < 0) lineAttrs["stroke-dasharray"] = "5 3";
      layer.appendChild(el("line", lineAttrs));
      layer.appendChild(el("text", {
        x: String(spoke.mx),
        y: String(spoke.my),
        class: "coef hover-match",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        fill: meta.color,
        "font-weight": "700",
      }, signed < 0 ? "−" : "+"));
    }

    const node = el("g", {
      transform: `translate(${cx},${cy})`,
      class: "sem-node sem-digman hover-match",
      "data-kind": "digman",
      "data-node-id": `digman:${meta.id}`,
    });
    node.appendChild(el("circle", {
      r: "35",
      fill: "none",
      stroke: meta.color,
      "stroke-width": "1.25",
      opacity: "0.55",
    }));
    node.appendChild(el("circle", {
      r: String(digR),
      fill: "var(--panel)",
      stroke: meta.color,
      "stroke-width": "2.5",
    }));
    node.appendChild(el("text", {
      class: "factor-label",
      fill: meta.color,
      "text-anchor": "middle",
      "dominant-baseline": "middle",
      y: "-0.35em",
      "font-size": "16",
    }, meta.label));
    node.appendChild(el("text", {
      fill: meta.color,
      "text-anchor": "middle",
      "dominant-baseline": "middle",
      y: "1.05em",
      "font-size": "8.5",
      "font-weight": "700",
      opacity: "0.9",
    }, meta.name));
    layer.appendChild(node);
    // Keep spokes under trait circles so arrows meet the rim instead of painting over.
    svg.insertBefore(layer, gNodes);
  }

  function setHoverHighlight(svg, link) {
    if (!svg) return;
    const active = Boolean(link);
    svg.classList.toggle("hover-active", active);
    const items = svg.__semData?.items || [];
    const factors = svg.__semData?.factors || [];
    const factorById = Object.fromEntries(factors.map((f) => [f.id, f]));
    const itemById = Object.fromEntries(items.map((it) => [String(it.item), it]));
    const { trait: matchTrait, meta: focusMeta } = resolveHoverMeta(link, itemById);
    const matchMeta = link?.metaId || focusMeta?.id || null;
    const matchItem = link?.item != null ? String(link.item) : null;

    if (!active) {
      clearHoverDigman(svg);
    } else if (focusMeta) {
      paintHoverDigman(svg, focusMeta, matchTrait);
    } else {
      clearHoverDigman(svg);
    }

    function inClique(trait) {
      return Boolean(trait && focusMeta && focusMeta.members[trait] != null);
    }

    function homeTraitMatches(trait) {
      if (!trait) return false;
      if (matchTrait && trait === matchTrait) return true;
      if (matchItem) {
        const it = itemById[matchItem];
        return it && it.domain === trait;
      }
      // Digman hub hover: all clique members' items
      if (matchMeta && !matchTrait && inClique(trait)) return true;
      return false;
    }

    for (const g of svg.querySelectorAll(".sem-node")) {
      const kind = g.getAttribute("data-kind");
      const id = g.getAttribute("data-node-id") || "";
      let on = false;
      if (!active) {
        g.classList.remove("hover-match", "hover-miss", "hover-focus");
        continue;
      }
      if (kind === "factor") {
        const trait = factorById[id.replace("factor:", "")]?.trait;
        on = inClique(trait);
        g.classList.toggle("hover-focus", Boolean(matchTrait && trait === matchTrait));
      } else if (kind === "item" || kind === "error") {
        const itemId = id.split(":")[1];
        const item = itemById[itemId];
        on = Boolean(
          (matchItem && itemId === matchItem)
          || homeTraitMatches(item?.domain)
        );
        g.classList.remove("hover-focus");
      } else if (kind === "digman") {
        on = id === `digman:${focusMeta?.id}`;
        g.classList.remove("hover-focus");
      } else {
        g.classList.remove("hover-focus");
      }
      g.classList.toggle("hover-match", on);
      g.classList.toggle("hover-miss", !on);
    }

    for (const el of svg.querySelectorAll(
      "#loadings [data-link-id], #loadings [data-link-label], #correlations [data-corr-key], #hover-digman .hover-match"
    )) {
      const lid = el.getAttribute("data-link-id")
        || el.getAttribute("data-link-label")
        || el.getAttribute("data-corr-key")
        || "";
      if (!active) {
        el.classList.remove("hover-match", "hover-miss");
        continue;
      }
      if (!lid) {
        // ghost Digman spokes already tagged hover-match
        continue;
      }
      let on = false;
      if (lid.startsWith("load:") || lid.startsWith("err:")) {
        const itemId = lid.split(":")[1];
        const item = itemById[itemId];
        on = Boolean((matchItem && itemId === matchItem) || homeTraitMatches(item?.domain));
      } else if (lid.startsWith("digman:")) {
        const metaId = lid.split(":")[1]?.split("->")[0];
        on = Boolean(focusMeta && metaId === focusMeta.id);
      } else if (lid.startsWith("errcorr:")) {
        const pair = lid.split(":")[1] || "";
        const [a, b] = pair.split("-");
        if (matchItem) on = matchItem === a || matchItem === b;
        else {
          const ia = itemById[a];
          const ib = itemById[b];
          on = homeTraitMatches(ia?.domain) && homeTraitMatches(ib?.domain);
        }
      } else {
        // Factor correlations: light the whole Digman α/β clique, as on select.
        const key = lid.replace(/-label$/, "");
        const [a, b] = key.split("-");
        const fa = factorById[a]?.trait;
        const fb = factorById[b]?.trait;
        on = Boolean(inClique(fa) && inClique(fb));
      }
      el.classList.toggle("hover-match", on);
      el.classList.toggle("hover-miss", !on);
    }
  }

  function onHover(svg, cb) {
    svg.__onHover = cb;
  }

  function buildGraph(data, visibleItems, focusTrait = null) {
    const factors = data.factors.map((f) => ({
      id: `factor:${f.id}`,
      kind: "factor",
      factorId: f.id,
      trait: f.trait,
      label: f.label,
      x: f.x,
      y: f.y,
      radius: data.layout.factorRadius,
    }));

    const items = visibleItems.map((it) => {
      const factor = data.factors.find((f) => f.trait === it.domain);
      const nHome = Math.max(1, visibleItems.filter((v) => v.domain === it.domain).length);
      const homeIdx = visibleItems.filter((v) => v.domain === it.domain).findIndex((v) => v.item === it.item);
      const base = (data.layout.clusterAngles?.[factor?.id]?.baseAngle ?? -90) * Math.PI / 180;
      const spread = ((data.layout.clusterAngles?.[factor?.id]?.spread ?? 100) * Math.PI / 180);
      const t = nHome === 1 ? 0 : (homeIdx / (nHome - 1) - 0.5);
      const ang = base + t * spread;
      const fx = factor?.x || 500;
      const fy = factor?.y || 460;
      return {
        id: `item:${it.item}`,
        kind: "item",
        item: it.item,
        meta: it,
        factorId: factor?.id,
        trait: it.domain,
        loading: it.loading,
        x: fx + Math.cos(ang) * 110,
        y: fy + Math.sin(ang) * 110,
        size: data.layout.itemSize,
      };
    });

    const errors = items.map((it) => ({
      id: `error:${it.item}`,
      kind: "error",
      item: it.item,
      factorId: it.factorId,
      x: it.x + 22,
      y: it.y - 10,
      radius: data.layout.errorRadius,
    }));

    const nodes = [...factors, ...items, ...errors];
    const links = [];

    for (const it of items) {
      links.push({
        id: `load:${it.item}`,
        source: `factor:${it.factorId}`,
        target: it.id,
        kind: "loading",
        loading: it.loading,
        factorId: it.factorId,
      });
      links.push({
        id: `err:${it.item}`,
        source: `error:${it.item}`,
        target: it.id,
        kind: "error",
        factorId: it.factorId,
      });
    }

    for (const { a, b, r } of data.factor_correlations || []) {
      links.push({
        id: `corr:${a}-${b}`,
        source: `factor:${a}`,
        target: `factor:${b}`,
        kind: "corr",
        r,
        a,
        b,
      });
    }

    // Residual inter-item correlations (only when both items are currently visible).
    const visibleIds = new Set(visibleItems.map((it) => it.item));
    for (const { a, b, r } of data.error_correlations || []) {
      if (!visibleIds.has(a) || !visibleIds.has(b)) continue;
      links.push({
        id: `errcorr:${a}-${b}`,
        source: `error:${a}`,
        target: `error:${b}`,
        kind: "errorCorr",
        r,
        a,
        b,
      });
    }

    // On trait focus: ephemeral Digman α/β hub linked to that metatrait's members.
    const meta = focusTrait ? metatraitOf(focusTrait) : null;
    if (meta) {
      const members = data.factors.filter((f) => meta.members[f.trait] != null);
      const cx = members.reduce((s, f) => s + f.x, 0) / Math.max(1, members.length);
      const cy = members.reduce((s, f) => s + f.y, 0) / Math.max(1, members.length);
      nodes.push({
        id: `digman:${meta.id}`,
        kind: "digman",
        metaId: meta.id,
        label: meta.label,
        name: meta.name,
        color: meta.color,
        members: members.map((f) => f.trait),
        x: cx,
        y: cy + 70,
        radius: 30,
      });
      for (const f of members) {
        const signed = meta.members[f.trait];
        links.push({
          id: `digman:${meta.id}->${f.id}`,
          source: `digman:${meta.id}`,
          target: `factor:${f.id}`,
          kind: "digman",
          loading: signed,
          memberTrait: f.trait,
          focused: f.trait === focusTrait,
        });
      }
    }

    return { nodes, links, factors, items, errors, digman: meta };
  }

  function renderDiagram(data, mountOptions = {}) {
    const stage = mountOptions.stage || null;
    const [vbX, vbY, vbW, vbH] = data.layout.viewBox;
    const svg = el("svg", {
      viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
      preserveAspectRatio: "xMidYMid meet",
    });
    svg.__semData = data;
    svg.__semFocusTrait = null;
    svg.__semSelection = null;
    svg.__semNodes = { factors: Object.fromEntries(data.factors.map((f) => [f.id, f])) };

    const defs = el("defs");
    defs.appendChild(arrowMarker("arrow-ink", "#1a1612"));
    defs.appendChild(arrowMarker("arrow-muted", "#7a7268"));
    defs.appendChild(arrowMarker("arrow-corr-start", "#c8bfb3", "1"));
    defs.appendChild(arrowMarker("arrow-corr-end", "#c8bfb3", "6"));
    svg.appendChild(defs);

    const gCorr = el("g", { id: "correlations" });
    const gLoad = el("g", { id: "loadings" });
    const gNodes = el("g", { id: "nodes" });
    svg.appendChild(gCorr);
    svg.appendChild(gLoad);
    svg.appendChild(gNodes);

    const tip = stage ? ensureTooltip(stage) : null;

    const state = {
      sim: null,
      nodeSel: null,
      linkLoad: null,
      linkCorr: null,
      graph: null,
      d3: null,
    };
    svg.__semState = state;

    function syncPositions() {
      const { graph } = state;
      if (!graph) return;

      function quadAt(x0, y0, x1, y1, x2, y2, t) {
        const u = 1 - t;
        return {
          x: u * u * x0 + 2 * u * t * x1 + t * t * x2,
          y: u * u * y0 + 2 * u * t * y1 + t * t * y2,
        };
      }
      function quadTangent(x0, y0, x1, y1, x2, y2, t) {
        return {
          dx: 2 * (1 - t) * (x1 - x0) + 2 * t * (x2 - x1),
          dy: 2 * (1 - t) * (y1 - y0) + 2 * t * (y2 - y1),
        };
      }

      for (const link of gLoad.querySelectorAll("[data-link-id]")) {
        const id = link.getAttribute("data-link-id");
        const L = graph.links.find((l) => l.id === id);
        if (!L) continue;
        const s = typeof L.source === "object" ? L.source : null;
        const t = typeof L.target === "object" ? L.target : null;
        if (!s || !t) continue;
        if (link.tagName === "line") {
          if (L.kind === "digman") {
            const sr = s.radius || 30;
            const tr = t.radius || data.layout.factorRadius || 40;
            const spoke = trimSpoke(s.x, s.y, sr, t.x, t.y, tr, 4);
            link.setAttribute("x1", spoke.x1);
            link.setAttribute("y1", spoke.y1);
            link.setAttribute("x2", spoke.x2);
            link.setAttribute("y2", spoke.y2);
            const label = gLoad.querySelector(`[data-link-label="${id}"]`);
            if (label) {
              label.setAttribute("x", spoke.mx);
              label.setAttribute("y", spoke.my);
            }
            continue;
          }
          link.setAttribute("x1", s.x);
          link.setAttribute("y1", s.y);
          link.setAttribute("x2", t.x);
          link.setAttribute("y2", t.y);
        } else if (link.tagName === "path" && L.kind === "errorCorr") {
          const mx = (s.x + t.x) / 2;
          const my = (s.y + t.y) / 2;
          const cpx = mx;
          const cpy = my - 24;
          link.setAttribute("d", `M ${s.x} ${s.y} Q ${cpx} ${cpy} ${t.x} ${t.y}`);
          const label = gLoad.querySelector(`[data-link-label="${id}"]`);
          if (label) {
            const p = quadAt(s.x, s.y, cpx, cpy, t.x, t.y, 0.5);
            const tan = quadTangent(s.x, s.y, cpx, cpy, t.x, t.y, 0.5);
            const len = Math.hypot(tan.dx, tan.dy) || 1;
            label.setAttribute("x", p.x + (-tan.dy / len) * 10);
            label.setAttribute("y", p.y + (tan.dx / len) * 10);
          }
          continue;
        }
        const label = gLoad.querySelector(`[data-link-label="${id}"]`);
        if (label) {
          label.setAttribute("x", (s.x + t.x) / 2);
          label.setAttribute("y", (s.y + t.y) / 2);
        }
      }

      for (const path of gCorr.querySelectorAll("path[data-corr-key]")) {
        const key = path.getAttribute("data-corr-key");
        if (!key || key.endsWith("-label")) continue;
        const L = graph.links.find((l) => l.id === `corr:${key}`);
        if (!L) continue;
        const s = typeof L.source === "object" ? L.source : null;
        const t = typeof L.target === "object" ? L.target : null;
        if (!s || !t) continue;
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        const cx = 500 + (mx - 500) * 0.2;
        const cy = 430 + (my - 430) * 0.2;
        path.setAttribute("d", `M ${s.x} ${s.y} Q ${cx} ${cy} ${t.x} ${t.y}`);
        const label = gCorr.querySelector(`text[data-corr-key="${key}-label"]`);
        if (label) {
          const p = quadAt(s.x, s.y, cx, cy, t.x, t.y, 0.5);
          const tan = quadTangent(s.x, s.y, cx, cy, t.x, t.y, 0.5);
          const len = Math.hypot(tan.dx, tan.dy) || 1;
          // Sit on the curve, nudged outward along the normal so it doesn't ride the stroke.
          label.setAttribute("x", p.x + (-tan.dy / len) * 11);
          label.setAttribute("y", p.y + (tan.dx / len) * 11);
        }
      }

      for (const g of gNodes.querySelectorAll("[data-node-id]")) {
        const id = g.getAttribute("data-node-id");
        const n = graph.nodes.find((node) => node.id === id);
        if (!n) continue;
        g.setAttribute("transform", `translate(${n.x},${n.y})`);
      }
    }

    function paintGraph(graph, selection) {
      clearHoverDigman(svg);
      while (gLoad.firstChild) gLoad.removeChild(gLoad.firstChild);
      while (gCorr.firstChild) gCorr.removeChild(gCorr.firstChild);
      while (gNodes.firstChild) gNodes.removeChild(gNodes.firstChild);

      const steered = selection?.meta?.trait || svg.__semFocusTrait || null;
      const focusId = steered ? data.factors.find((f) => f.trait === steered)?.id : null;
      const correlations = selection?.correlations
        || selection?.structure?.raw
        || data.factor_correlations;
      const observed = Boolean(selection?.correlations || selection?.structure);
      const scores = selection?.scores || null;

      const corrByPair = {};
      for (const p of correlations || []) corrByPair[`${p.a}-${p.b}`] = p.r;

      for (const L of graph.links.filter((l) => l.kind === "corr")) {
        const r = corrByPair[`${L.a}-${L.b}`] ?? L.r;
        const mag = Math.abs(r || 0);
        const { stroke, colored } = corrStrokeStyle(
          data, L.a, L.b, r || 0, observed, steered
        );
        const path = el("path", {
          "data-corr-key": `${L.a}-${L.b}`,
          fill: "none",
          stroke,
          "stroke-width": String(corrStrokeWidth(r || 0)),
          opacity: String(0.45 + mag * 0.45),
          "marker-start": "url(#arrow-corr-start)",
          "marker-end": "url(#arrow-corr-end)",
        });
        path.style.transition = "stroke-width 280ms ease, opacity 280ms ease, stroke 280ms ease";
        gCorr.appendChild(path);
        const label = el("text", {
          "data-corr-key": `${L.a}-${L.b}-label`,
          class: colored ? "corr-coef corr-observed" : "corr-coef",
          fill: colored ? stroke : "#7a7268",
          opacity: "1",
          "text-anchor": "middle",
          "dominant-baseline": "middle",
        }, fmtCorr(r || 0));
        label.style.transition = "opacity 280ms ease, fill 280ms ease";
        // Position is owned by syncPositions each tick — don't CSS-transition x/y.
        gCorr.appendChild(label);
      }

      for (const L of graph.links.filter((l) => l.kind === "loading" || l.kind === "error" || l.kind === "digman")) {
        if (L.kind === "digman") {
          const dig = graph.nodes.find((n) => n.kind === "digman");
          const lineAttrs = {
            "data-link-id": L.id,
            stroke: dig?.color || "#5b4f8a",
            "stroke-width": L.focused ? "2.4" : "1.6",
            opacity: "0.95",
            "marker-end": "url(#arrow-ink)",
          };
          if (L.loading < 0) lineAttrs["stroke-dasharray"] = "5 3";
          gLoad.appendChild(el("line", lineAttrs));
          gLoad.appendChild(el("text", {
            "data-link-label": L.id,
            class: "coef",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            fill: dig?.color || "#5b4f8a",
            "font-weight": "700",
          }, L.loading < 0 ? "−" : "+"));
          continue;
        }
        const factor = data.factors.find((f) => f.id === L.factorId);
        const emphasized = steered && factor?.trait === steered;
        if (L.kind === "loading") {
      gLoad.appendChild(el("line", {
            "data-link-id": L.id,
            "data-factor-id": L.factorId,
            stroke: TRAIT_COLOR[factor?.trait] || "#1a1612",
            "stroke-width": emphasized ? "1.75" : "1.25",
        "marker-end": "url(#arrow-ink)",
            opacity: emphasized ? "1" : steered ? "0.28" : "0.85",
          }));
          gLoad.appendChild(el("text", {
            "data-link-label": L.id,
            class: "coef",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
          }, fmtLoading(L.loading)));
        } else {
      gLoad.appendChild(el("line", {
            "data-link-id": L.id,
        stroke: "#c8bfb3",
        "stroke-width": "1",
        "marker-end": "url(#arrow-muted)",
            opacity: steered && !emphasized ? "0.25" : "0.9",
          }));
        }
      }

      for (const L of graph.links.filter((l) => l.kind === "errorCorr")) {
        const mag = Math.abs(L.r || 0);
        gLoad.appendChild(el("path", {
          "data-link-id": L.id,
        fill: "none",
        stroke: "#b5a99a",
          "stroke-width": String(0.8 + mag * 2.2),
          "stroke-dasharray": "4 3",
          opacity: String(0.45 + mag * 0.4),
        }));
        gLoad.appendChild(el("text", {
          "data-link-label": L.id,
          class: "corr-coef",
          "text-anchor": "middle",
          "dominant-baseline": "middle",
          fill: "#7a7268",
        }, fmtCorr(L.r || 0)));
      }

      for (const n of graph.nodes) {
        const g = el("g", {
          "data-node-id": n.id,
          "data-kind": n.kind,
          class: `sem-node sem-${n.kind}`,
        });

        if (n.kind === "factor") {
          const score = scores?.[n.trait];
          const t = score != null ? scoreNorm(score) : null;
          const emphasized = steered === n.trait;
          const baseR = data.layout.factorRadius;
          const r = baseR * (t != null ? 0.88 + 0.28 * t : 1);
          g.appendChild(el("circle", {
            "data-factor-id": n.factorId,
            r: String(r),
        fill: "var(--panel)",
            stroke: TRAIT_COLOR[n.trait],
            "stroke-width": emphasized ? "3.5" : "2",
            "fill-opacity": "1",
          }));
          g.appendChild(el("circle", {
            class: "factor-hot-ring",
            r: String(baseR + 9),
            fill: "none",
            stroke: TRAIT_COLOR[n.trait],
            "stroke-width": "2.25",
            opacity: "0",
            "pointer-events": "none",
          }));
          if (emphasized && t != null) {
            g.appendChild(el("circle", {
              class: "score-overlay steer-halo",
              r: String(baseR + 10),
              fill: "none",
              stroke: TRAIT_COLOR[n.trait],
              "stroke-width": "1.5",
              opacity: "0.55",
            }));
          }
          const label = el("text", {
            "data-factor-label": n.factorId,
        class: "factor-label",
            fill: TRAIT_COLOR[n.trait],
          });
          setFactorLabel(label, n, score);
          g.appendChild(label);
          if (t != null) {
            g.appendChild(el("text", {
              class: "score-overlay trait-name",
              y: String(baseR + 18),
              fill: TRAIT_COLOR[n.trait],
        "text-anchor": "middle",
              "font-size": "9",
              "font-weight": "600",
              opacity: emphasized ? "1" : "0.72",
            }, TRAIT_NAME[n.trait]));
          }
          g.style.cursor = "pointer";
          g.addEventListener("click", (e) => {
            e.stopPropagation();
            const cb = svg.__onFactorClick;
            if (cb) cb(n.trait);
          });
          bindHoverTip(g, tip, stage, () => factorTooltipHtml(n), {
            trait: n.trait,
            metaId: metatraitOf(n.trait)?.id || null,
          });
        } else if (n.kind === "item") {
          const half = n.size / 2;
          const rect = el("rect", {
            x: -half,
            y: -half,
            width: n.size,
            height: n.size,
        fill: "var(--panel)",
        stroke: "#1a1612",
        "stroke-width": "1",
        rx: "2",
            style: "cursor:help",
          });
          g.appendChild(rect);
          g.appendChild(el("text", {
        class: "item-label",
        fill: "#1a1612",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
          }, String(n.item)));
          const itemPayload = {
            trait: n.trait,
            item: n.item,
            metaId: metatraitOf(n.trait)?.id || null,
          };
          g.addEventListener("pointerenter", (e) => {
            if (!tip) return;
            const factor = data.factors.find((f) => f.id === n.factorId);
            showItemTooltip(tip, stage, e.clientX, e.clientY, n.meta, factor);
            const cb = svg.__onHover;
            if (cb) cb(itemPayload);
          });
          g.addEventListener("pointermove", (e) => {
            if (!tip || tip.hidden) return;
            const factor = data.factors.find((f) => f.id === n.factorId);
            showItemTooltip(tip, stage, e.clientX, e.clientY, n.meta, factor);
          });
          g.addEventListener("pointerleave", () => {
            hideTooltip(tip);
            const cb = svg.__onHover;
            if (cb) cb(null);
          });
        } else if (n.kind === "digman") {
          g.appendChild(el("circle", {
            r: String(n.radius + 5),
            fill: "none",
            stroke: n.color,
            "stroke-width": "1.25",
            opacity: "0.55",
          }));
          g.appendChild(el("circle", {
            r: String(n.radius),
            fill: "var(--panel)",
            stroke: n.color,
            "stroke-width": "2.5",
          }));
          g.appendChild(el("text", {
            class: "factor-label",
            fill: n.color,
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            y: "-0.35em",
            "font-size": "16",
          }, n.label));
          g.appendChild(el("text", {
            fill: n.color,
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            y: "1.05em",
            "font-size": "8.5",
            "font-weight": "700",
            opacity: "0.9",
          }, n.name));
          bindHoverTip(g, tip, stage, () => digmanTooltipHtml(n), { metaId: n.metaId });
        } else if (n.kind === "error") {
          g.appendChild(el("circle", {
            r: String(n.radius),
        fill: "var(--bg)",
        stroke: "#7a7268",
        "stroke-width": "1",
      }));
          g.appendChild(el("text", {
        class: "error-label",
            y: "0.5",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
          }, `e${n.item}`));
        }

        gNodes.appendChild(g);
      }

      syncPositions();
    }

    function runSimulation(visibleItems, selection) {
      const focusTrait = selection?.meta?.trait || svg.__semFocusTrait || null;
      const graph = buildGraph(data, visibleItems, focusTrait);
      state.graph = graph;
      svg.__semGraph = graph;
      paintGraph(graph, selection);

      ensureD3().then((d3) => {
        state.d3 = d3;
        if (state.sim) state.sim.stop();

        const factorR = data.layout.factorRadius;
        const itemHalf = data.layout.itemSize * 0.55;
        const errR = data.layout.errorRadius;

        function repelFromForeignFactors(alpha) {
          const factors = graph.nodes.filter((n) => n.kind === "factor");
          for (const n of graph.nodes) {
            if (n.kind === "factor" || n.kind === "digman") continue;
            const own = n.kind === "item" ? itemHalf : errR;
            for (const f of factors) {
              if (n.factorId === f.factorId) continue;
              let dx = n.x - f.x;
              let dy = n.y - f.y;
              let dist = Math.hypot(dx, dy);
              if (dist < 1e-6) {
                dx = (Math.random() - 0.5) || 0.01;
                dy = (Math.random() - 0.5) || 0.01;
                dist = Math.hypot(dx, dy);
              }
              const pad = n.kind === "error" ? 78 : 56;
              const minDist = factorR + own + pad;
              if (dist >= minDist) continue;
              const push = ((minDist - dist) / dist) * alpha * 2.4;
              n.vx += dx * push;
              n.vy += dy * push;
              f.vx -= dx * push * 0.2;
              f.vy -= dy * push * 0.2;
            }
          }
        }

        const sim = d3.forceSimulation(graph.nodes)
          .force("link", d3.forceLink(graph.links)
            .id((d) => d.id)
            .distance((l) => {
              if (l.kind === "digman") return 95;
              if (l.kind === "errorCorr") return 70;
              if (l.kind === "error") return 36;
              if (l.kind === "loading") return 118;
              return 240;
            })
            .strength((l) => {
              if (l.kind === "digman") return 0.85;
              if (l.kind === "errorCorr") return 0.15;
              if (l.kind === "error") return 0.9;
              if (l.kind === "loading") return 0.5;
              return 0.05;
            }))
          .force("charge", d3.forceManyBody()
            .strength((d) => {
              if (d.kind === "digman") return -780;
              if (d.kind === "factor") return -1100;
              if (d.kind === "item") return -220;
              return -8;
            })
            .distanceMin(8)
            .distanceMax(480))
          .force("center", d3.forceCenter(500, 460).strength(0.03))
          .force("collide", d3.forceCollide()
            .radius((d) => {
              if (d.kind === "digman") return 36;
              if (d.kind === "factor") return factorR + 18;
              if (d.kind === "item") return itemHalf + 7;
              return errR + 1;
            })
            .strength(0.55)
            .iterations(3))
          .force("repelForeign", repelFromForeignFactors)
          .force("factorAnchor", d3.forceX((d) => {
            if (d.kind === "digman") return d.x;
            if (d.kind !== "factor") return 500;
            return data.factors.find((f) => f.id === d.factorId)?.x || 500;
          }).strength((d) => {
            if (d.kind === "digman") return 0.08;
            return d.kind === "factor" ? 0.14 : 0;
          }))
          .force("factorAnchorY", d3.forceY((d) => {
            if (d.kind === "digman") return d.y;
            if (d.kind !== "factor") return 460;
            return data.factors.find((f) => f.id === d.factorId)?.y || 460;
          }).strength((d) => {
            if (d.kind === "digman") return 0.08;
            return d.kind === "factor" ? 0.14 : 0;
          }))
          .alpha(0.9)
          .alphaDecay(0.024);

        state.sim = sim;
        svg.__semSim = sim;

        sim.on("tick", syncPositions);

        const drag = d3.drag()
          .on("start", (event, d) => {
            if (!d) return;
            if (!event.active) sim.alphaTarget(0.25).restart();
            d.fx = d.x;
            d.fy = d.y;
            event.sourceEvent?.stopPropagation?.();
          })
          .on("drag", (event, d) => {
            if (!d) return;
            d.fx = event.x;
            d.fy = event.y;
            event.sourceEvent?.stopPropagation?.();
          })
          .on("end", (event, d) => {
            if (!d) return;
            if (!event.active) sim.alphaTarget(0);
            if (d.kind !== "factor") {
              d.fx = null;
              d.fy = null;
            }
          });

        const nodeSel = d3.select(gNodes).selectAll("g.sem-node");
        nodeSel.each(function bindDatum() {
          const id = this.getAttribute("data-node-id");
          d3.select(this).datum(graph.nodes.find((n) => n.id === id));
        });
        nodeSel.call(drag);
      }).catch((err) => {
        console.warn("force layout unavailable, static positions used", err);
        syncPositions();
      });
    }

    function setItemFocus(trait, selection = svg.__semSelection) {
      svg.__semFocusTrait = trait || null;
      const visible = resolveVisibleItems(data, trait || null);
      runSimulation(visible, selection || null);
      return visible;
    }

    svg.__setItemFocus = setItemFocus;
    setItemFocus(null, null);

    return svg;
  }

  function applySemSelection(svg, selection) {
    svg.__semSelection = selection || null;
    const steered = selection?.meta?.trait || null;
    if (typeof svg.__setItemFocus === "function") {
      svg.__setItemFocus(steered, selection);
      return;
    }
  }

  function clearSemSelection(svg) {
    applySemSelection(svg, null);
  }

  function applyFactorScores(svg, scores, meta) {
    applySemSelection(svg, { scores, meta });
  }

  function attachPanZoom(stage, svg) {
    // Viewport pan is locked: background drag used to slide the viewBox and
    // felt like the diagram kept drifting. Wheel zoom still works; nodes still drag.
    const base = svg.viewBox.baseVal;
    const baseW = base.width;
    const baseH = base.height;
    let scale = 1;
    let press = false;
    let moved = false;
    let lx = 0;
    let ly = 0;
    const ZOOM = 0.00055;

    function applyZoom() {
      const b = svg.viewBox.baseVal;
      const cx = b.x + b.width / 2;
      const cy = b.y + b.height / 2;
      const w = baseW / scale;
      const h = baseH / scale;
      svg.setAttribute("viewBox", `${cx - w / 2} ${cy - h / 2} ${w} ${h}`);
    }

    stage.addEventListener("wheel", (e) => {
      e.preventDefault();
      let delta = e.deltaY;
      if (e.deltaMode === 1) delta *= 16;
      else if (e.deltaMode === 2) delta *= baseH;
      const factor = Math.exp(-delta * ZOOM);
      scale = Math.min(3, Math.max(0.65, scale * factor));
      applyZoom();
    }, { passive: false });

    stage.addEventListener("pointerdown", (e) => {
      if (e.target.closest?.(".sem-node")) return;
      press = true;
      moved = false;
      lx = e.clientX;
      ly = e.clientY;
    });
    stage.addEventListener("pointermove", (e) => {
      if (!press) return;
      if (Math.abs(e.clientX - lx) + Math.abs(e.clientY - ly) > 3) moved = true;
    });
    const end = (e) => {
      const wasPress = press;
      const didMove = moved;
      press = false;
      if (wasPress && !didMove && e.type === "pointerup") {
        const cb = svg.__onBackgroundClick;
        if (cb) cb();
      }
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
  }

  function onFactorClick(svg, cb) {
    svg.__onFactorClick = cb;
  }

  function onBackgroundClick(svg, cb) {
    svg.__onBackgroundClick = cb;
  }

  global.BigFiveSem = {
    TRAIT_COLOR,
    TRAIT_NAME,
    METATRAITS,
    TOP_N_DEFAULT,
    TOP_N_FOCUSED,
    metatraitOf,
    factorIdForTrait,
    resolveVisibleItems,
    renderDiagram,
    computeEvStructure,
    computeEvCorrelations,
    applySemSelection,
    clearSemSelection,
    applyFactorScores,
    attachPanZoom,
    onFactorClick,
    onBackgroundClick,
    onHover,
    setHoverHighlight,
    pointTooltipHtml,
    setEvBaseline,
    factorTooltipHtml,
    digmanTooltipHtml,
    ensureTooltip,
    positionTip,
    hideTooltip,
  };
})(typeof window !== "undefined" ? window : globalThis);
