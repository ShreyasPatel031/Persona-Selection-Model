/** Shared chapter strip for the Inferno persona series. */
window.VIZ_SERIES = [
  {
    href: "layer3d.html",
    title: "Layer Viz",
    blurb: "Where the trait lives across layers",
  },
  {
    href: "inferno_cone.html",
    title: "Nine Alphas",
    blurb: "α climbs — refuse turns accept",
  },
  {
    href: "omp_reconstruction_3d.html",
    title: "OMP Recon",
    blurb: "Rebuild the vector from SAE features",
  },
  {
    href: "ssv_bubble_viz_omp.html",
    title: "Feature Bubbles",
    blurb: "What fires as the sparse code grows",
  },
  {
    href: "dnd_composition_board.html",
    title: "Composition",
    blurb: "Blend alignment vectors into a reply",
  },
  {
    href: "big_five_persona.html",
    title: "Five Personas",
    blurb: "Compose OCEAN — inventory silhouette",
  },
];

(function injectSeriesNav() {
  const path = (location.pathname.split("/").pop() || "").split("?")[0];
  const i = window.VIZ_SERIES.findIndex((s) => s.href === path);
  if (i < 0) return;

  const prev = window.VIZ_SERIES[i - 1];
  const next = window.VIZ_SERIES[i + 1];
  const cur = window.VIZ_SERIES[i];

  const css = document.createElement("style");
  css.textContent = `
    #viz-series-nav {
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 40;
      display: grid; grid-template-columns: 1fr auto 1fr;
      align-items: center; gap: 0.75rem;
      padding: 0.45rem 1rem;
      background: rgba(246, 243, 238, 0.94);
      border-top: 1px solid #e4ddd3;
      font-family: "Instrument Sans", system-ui, sans-serif;
      backdrop-filter: blur(8px);
    }
    #viz-series-nav a {
      color: #1a1612; text-decoration: none; font-size: 0.72rem; font-weight: 600;
    }
    #viz-series-nav a.muted, #viz-series-nav .muted { color: #7a7268; font-weight: 500; }
    #viz-series-nav a:hover { color: #3d8b6e; }
    #viz-series-nav .center {
      text-align: center; font-size: 0.68rem; letter-spacing: 0.06em;
      text-transform: uppercase; color: #7a7268; font-weight: 700;
    }
    #viz-series-nav .center strong { color: #1a1612; font-weight: 700; }
    #viz-series-nav .right { text-align: right; }
    body.has-viz-series { padding-bottom: 2.6rem; }
  `;
  document.head.appendChild(css);
  document.body.classList.add("has-viz-series");

  const nav = document.createElement("nav");
  nav.id = "viz-series-nav";
  nav.setAttribute("aria-label", "Persona visualization series");
  nav.innerHTML = `
    <div class="left">${
      prev
        ? `<a href="${prev.href}">← ${prev.title}</a>`
        : `<span class="muted">start</span>`
    }</div>
    <div class="center"><strong>${i + 1}</strong> / ${window.VIZ_SERIES.length} · ${cur.title}</div>
    <div class="right">${
      next
        ? `<a href="${next.href}">${next.title} →</a>`
        : `<span class="muted">finale</span>`
    }</div>
  `;
  document.body.appendChild(nav);
})();
