(() => {
  const CHAPTERS = [
    { n: "01", file: "layer3d.html", title: "Layer geometry", tag: "where traits live" },
    { n: "02", file: "inferno_cone.html", title: "Nine alphas", tag: "α dose ladder" },
    { n: "03", file: "omp_reconstruction_3d.html", title: "OMP reconstruction", tag: "sparse SAE fit" },
    { n: "04", file: "ssv_bubble_viz_omp.html", title: "Feature bubbles", tag: "code grows with K" },
    { n: "05", file: "dnd_composition_board.html", title: "Composition board", tag: "alignment grid" },
    { n: "06", file: "big_five_persona.html", title: "Five personas", tag: "OCEAN silhouette" },
    { n: "07", file: "big_five_sem.html", title: "Measurement model", tag: "MPI-120 CFA" },
    { n: "—", file: "big_five_tsne.html", title: "t-SNE ladder", tag: "supplement" },
  ];

  const here = (location.pathname.split("/").pop() || "").toLowerCase();
  const idx = CHAPTERS.findIndex((c) => c.file.toLowerCase() === here);
  if (idx < 0) return;

  const cur = CHAPTERS[idx];
  const prev = idx > 0 ? CHAPTERS[idx - 1] : null;
  const next = idx < CHAPTERS.length - 1 ? CHAPTERS[idx + 1] : null;

  const nav = document.createElement("nav");
  nav.id = "viz-series-nav";
  nav.setAttribute("aria-label", "Inferno viz series");
  nav.innerHTML = `
    <a class="hub" href="./viz_series.html" title="All chapters">Series</a>
    ${prev ? `<a class="prev" href="./${prev.file}">← ${prev.n}</a>` : `<span class="spacer"></span>`}
    <span class="here"><span class="num">${cur.n}</span> ${cur.title}</span>
    ${next ? `<a class="next" href="./${next.file}">${next.n} →</a>` : `<span class="spacer"></span>`}
  `;

  const css = document.createElement("style");
  css.textContent = `
    #viz-series-nav {
      position: fixed;
      left: 0; right: 0; bottom: 0;
      z-index: 40;
      display: grid;
      grid-template-columns: auto 1fr auto 1fr auto;
      align-items: center;
      gap: 0.5rem 0.75rem;
      padding: 0.4rem 1rem;
      font-family: "Instrument Sans", system-ui, sans-serif;
      font-size: 0.68rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #7a7268;
      background: rgba(246, 243, 238, 0.96);
      border-top: 1px solid #e4ddd3;
      pointer-events: none;
    }
    #viz-series-nav a, #viz-series-nav .here { pointer-events: auto; }
    #viz-series-nav a {
      color: #1a1612;
      text-decoration: none;
      white-space: nowrap;
    }
    #viz-series-nav a:hover { text-decoration: underline; }
    #viz-series-nav .hub { justify-self: start; color: #7a7268; }
    #viz-series-nav .prev { justify-self: end; }
    #viz-series-nav .next { justify-self: start; }
    #viz-series-nav .here {
      justify-self: center;
      text-align: center;
      color: #1a1612;
      max-width: 14rem;
      line-height: 1.25;
    }
    #viz-series-nav .here .num { color: #7a7268; margin-right: 0.35rem; }
    #viz-series-nav .spacer { display: block; }
    body:has(#viz-series-nav) { padding-bottom: 2.1rem !important; }
  `;

  document.head.appendChild(css);
  document.body.appendChild(nav);
})();
