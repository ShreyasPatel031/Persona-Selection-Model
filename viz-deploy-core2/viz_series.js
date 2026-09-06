(() => {
  const CHAPTERS = [
    { n: "01", file: "big_five_tsne.html", title: "Embedding + MPI-120", tag: "what persona means", group: "core" },
    { n: "02", file: "big_five_persona.html", title: "Five personas", tag: "monotonic steer", group: "core" },
  ];

  window.VIZ_SERIES = { chapters: CHAPTERS };

  const here = (location.pathname.split("/").pop() || "").toLowerCase();
  if (here === "viz_series.html" || here === "") return;

  const idx = CHAPTERS.findIndex((c) => c.file.toLowerCase() === here);
  if (idx < 0) return;

  const cur = CHAPTERS[idx];
  const prev = idx > 0 ? CHAPTERS[idx - 1] : null;
  const next = idx < CHAPTERS.length - 1 ? CHAPTERS[idx + 1] : null;

  /* ── bottom chapter nav ─────────────────────────────────────── */
  const nav = document.createElement("nav");
  nav.id = "viz-series-nav";
  nav.setAttribute("aria-label", "Big Five Steering");
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

    /* ── notes sidebar ─────────────────────────────────────────── */
    #viz-notes-toggle {
      position: fixed;
      top: 50%;
      right: 0;
      z-index: 52;
      transform: translateY(-50%);
      writing-mode: vertical-rl;
      text-orientation: mixed;
      border: 1px solid #e4ddd3;
      border-right: none;
      border-radius: 3px 0 0 3px;
      background: #fffefb;
      color: #1a1612;
      font-family: "Instrument Sans", system-ui, sans-serif;
      font-size: 0.62rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      padding: 0.9rem 0.4rem;
      cursor: pointer;
      box-shadow: -2px 0 10px rgba(26, 22, 18, 0.04);
    }
    #viz-notes-toggle:hover { background: #f6f3ee; }
    body.viz-notes-open #viz-notes-toggle {
      right: min(26rem, 92vw);
      transform: translateY(-50%);
    }

    #viz-notes {
      position: fixed;
      top: 0; right: 0; bottom: 2.1rem;
      z-index: 51;
      width: min(26rem, 92vw);
      display: flex;
      flex-direction: column;
      background: rgba(255, 254, 251, 0.98);
      border-left: 1px solid #e4ddd3;
      box-shadow: -8px 0 28px rgba(26, 22, 18, 0.06);
      transform: translateX(105%);
      transition: transform 0.22s ease;
      font-family: "Instrument Sans", system-ui, sans-serif;
      color: #1a1612;
    }
    body.viz-notes-open #viz-notes { transform: translateX(0); }

    #viz-notes .vn-head {
      flex: 0 0 auto;
      padding: 0.85rem 1rem 0.7rem;
      border-bottom: 1px solid #e4ddd3;
    }
    #viz-notes .vn-kicker {
      font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
      text-transform: uppercase; color: #7a7268; margin-bottom: 0.2rem;
    }
    #viz-notes .vn-title {
      font-family: "Newsreader", Georgia, serif;
      font-size: 1.2rem; font-weight: 600; letter-spacing: -0.02em;
      margin: 0 0 0.35rem;
    }
    #viz-notes .vn-blurb {
      font-size: 0.78rem; color: #7a7268; line-height: 1.45; margin: 0;
    }
    #viz-notes .vn-tabs {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      border-bottom: 1px solid #e4ddd3;
    }
    #viz-notes .vn-tabs button {
      border: none; background: transparent; cursor: pointer;
      font-family: inherit; font-size: 0.62rem; font-weight: 700;
      letter-spacing: 0.07em; text-transform: uppercase;
      color: #7a7268; padding: 0.65rem 0.4rem;
      border-bottom: 2px solid transparent; margin-bottom: -1px;
    }
    #viz-notes .vn-tabs button:hover { color: #1a1612; }
    #viz-notes .vn-tabs button.active {
      color: #1a1612; border-bottom-color: #1a1612;
    }
    #viz-notes .vn-tabs button.fail.active {
      color: #c45c4a; border-bottom-color: #c45c4a;
    }
    #viz-notes .vn-tabs button.ok.active {
      color: #3d8b6e; border-bottom-color: #3d8b6e;
    }
    #viz-notes .vn-body {
      flex: 1 1 auto;
      overflow: auto;
      padding: 0.85rem 1rem 1.4rem;
      -webkit-overflow-scrolling: touch;
    }
    #viz-notes .vn-section { display: none; }
    #viz-notes .vn-section.active { display: block; }
    #viz-notes .vn-card {
      padding: 0.7rem 0;
      border-bottom: 1px solid #e4ddd3;
    }
    #viz-notes .vn-card:last-child { border-bottom: none; }
    #viz-notes .vn-card h3 {
      font-size: 0.82rem; font-weight: 700; margin: 0 0 0.35rem;
      letter-spacing: -0.01em; line-height: 1.3;
    }
    #viz-notes .vn-card p {
      font-size: 0.78rem; line-height: 1.5; color: #3a342e; margin: 0;
    }
    #viz-notes .vn-card.fail h3 { color: #c45c4a; }
    #viz-notes .vn-card.ok h3 { color: #3d8b6e; }
    #viz-notes .vn-tex {
      display: block;
      margin: 0.45rem 0 0.55rem;
      padding: 0.55rem 0.65rem;
      background: #f6f3ee;
      border: 1px solid #e4ddd3;
      border-radius: 3px;
      overflow-x: auto;
      font-size: 0.95rem;
      text-align: center;
    }
    #viz-notes .vn-empty {
      font-size: 0.78rem; color: #7a7268; line-height: 1.45;
    }
    @media (max-width: 720px) {
      #viz-notes { bottom: 2.1rem; width: min(100vw, 22rem); }
    }
  `;

  document.head.appendChild(css);
  document.body.appendChild(nav);

  /* ── KaTeX (for formula tab) ────────────────────────────────── */
  function loadKatex() {
    if (window.katex) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css";
      document.head.appendChild(link);
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js";
      s.onload = () => resolve();
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  function renderTex(el, tex) {
    if (!tex) return;
    try {
      if (window.katex) {
        window.katex.render(tex, el, { throwOnError: false, displayMode: true });
      } else {
        el.textContent = tex;
      }
    } catch (_) {
      el.textContent = tex;
    }
  }

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function cardHtml(item, kind) {
    const cls = kind === "fail" ? "fail" : kind === "ok" ? "ok" : "";
    const tex = item.tex
      ? `<div class="vn-tex" data-tex="${esc(item.tex)}"></div>`
      : "";
    return `<article class="vn-card ${cls}">
      <h3>${esc(item.name || item.title)}</h3>
      ${tex}
      <p>${esc(item.explain || item.body)}</p>
    </article>`;
  }

  async function mountNotes(note) {
    const toggle = document.createElement("button");
    toggle.id = "viz-notes-toggle";
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-controls", "viz-notes");
    toggle.textContent = "Notes";

    const panel = document.createElement("aside");
    panel.id = "viz-notes";
    panel.setAttribute("aria-label", "Chapter notes");
    panel.innerHTML = `
      <div class="vn-head">
        <div class="vn-kicker">Chapter ${esc(note.n)} · theory</div>
        <h2 class="vn-title">${esc(note.title)}</h2>
        <p class="vn-blurb">${esc(note.blurb)}</p>
      </div>
      <div class="vn-tabs" role="tablist">
        <button type="button" role="tab" data-tab="ok" class="ok active" aria-selected="true">Implemented</button>
        <button type="button" role="tab" data-tab="formulas" aria-selected="false">Formulas</button>
        <button type="button" role="tab" data-tab="fail" class="fail" aria-selected="false">Failed</button>
      </div>
      <div class="vn-body">
        <div class="vn-section active" data-panel="ok">
          ${(note.implemented || []).map((x) => cardHtml(x, "ok")).join("") || '<p class="vn-empty">No notes.</p>'}
        </div>
        <div class="vn-section" data-panel="formulas">
          ${(note.formulas || []).map((x) => cardHtml(x, "")).join("") || '<p class="vn-empty">No formulas.</p>'}
        </div>
        <div class="vn-section" data-panel="fail">
          ${(note.failed || []).map((x) => cardHtml(x, "fail")).join("") || '<p class="vn-empty">No failures logged.</p>'}
        </div>
      </div>
    `;

    document.body.appendChild(toggle);
    document.body.appendChild(panel);

    const open = () => {
      document.body.classList.add("viz-notes-open");
      toggle.setAttribute("aria-expanded", "true");
      toggle.textContent = "Close";
    };
    const close = () => {
      document.body.classList.remove("viz-notes-open");
      toggle.setAttribute("aria-expanded", "false");
      toggle.textContent = "Notes";
    };
    toggle.addEventListener("click", () => {
      if (document.body.classList.contains("viz-notes-open")) close();
      else open();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && document.body.classList.contains("viz-notes-open")) close();
    });

    panel.querySelectorAll(".vn-tabs button").forEach((btn) => {
      btn.addEventListener("click", () => {
        panel.querySelectorAll(".vn-tabs button").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        panel.querySelectorAll(".vn-section").forEach((s) => s.classList.remove("active"));
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        panel.querySelector(`.vn-section[data-panel="${btn.dataset.tab}"]`)?.classList.add("active");
      });
    });

    try {
      await loadKatex();
      panel.querySelectorAll(".vn-tex[data-tex]").forEach((el) => {
        renderTex(el, el.getAttribute("data-tex"));
      });
    } catch (_) {
      /* plain tex fallback already in renderTex */
    }
  }

  fetch("./viz_chapter_notes.json?v=1")
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      const note = data?.chapters?.[cur.file];
      if (note) mountNotes(note);
    })
    .catch(() => {});
})();
