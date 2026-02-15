(function () {
  const MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
  const MATHLIVE_URL = "https://unpkg.com/mathlive@0.92.0/dist/mathlive.min.js";

  function detectLang() {
    const first = window.location.pathname.split("/").filter(Boolean)[0];
    return first === "it" || first === "en" ? first : "en";
  }

  async function loadComponent(targetId, url) {
    const target = document.getElementById(targetId);
    if (!target) return null;
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) throw new Error(`Failed loading ${url}`);
    target.innerHTML = await response.text();
    return target;
  }

  async function injectSidebar() {
    const host = document.createElement("div");
    host.id = "sidebarHost";
    document.body.appendChild(host);

    const response = await fetch("/frontend/components/sidebar.html", { cache: "no-cache" });
    if (!response.ok) throw new Error("Failed loading /frontend/components/sidebar.html");
    host.innerHTML = await response.text();

    const lang = detectLang();
    const currentSlug = window.location.pathname.split("/").filter(Boolean)[1] || "";

    host.querySelectorAll("a[data-slug]").forEach((link) => {
      const slug = link.getAttribute("data-slug");
      link.href = `/${lang}/${slug}`;
      if (slug === currentSlug) link.classList.add("active");
      link.addEventListener("click", () => {
        closeSidebar();
      });
    });

    const homeLink = host.querySelector("[data-home-link]");
    if (homeLink) homeLink.href = `/${lang}/`;

    host.querySelectorAll("[data-section-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-section-toggle");
        const panel = host.querySelector(`[data-section='${key}']`);
        if (!panel) return;
        const opening = panel.classList.contains("collapsed");
        panel.classList.toggle("collapsed", !opening);
        btn.classList.toggle("expanded", opening);
      });
    });

    const overlay = host.querySelector("#sidebarOverlay");
    if (overlay) overlay.addEventListener("click", closeSidebar);

    setupSidebarSearch(host);
  }

  function setupSidebarSearch(host) {
    const input = host.querySelector("#sidebarSearch-static");
    const clear = host.querySelector("#searchClear-static");
    const results = host.querySelector("#searchResults-static");
    if (!input || !clear || !results) return;

    const dataset = Array.from(host.querySelectorAll(".algorithm-link[data-slug]")).map((link) => {
      const icon = link.querySelector("i")?.className || "fas fa-square";
      const title = link.querySelector("span")?.textContent?.trim() || link.textContent.trim();
      const slug = link.getAttribute("data-slug");
      const category = link.closest(".category-section")?.querySelector(".category-title span")?.textContent?.trim() || "";
      return { icon, title, slug, category, href: link.href };
    });

    function closeResults() {
      results.style.display = "none";
      results.innerHTML = "";
      clear.style.display = "none";
    }

    function openResults(list, query) {
      results.innerHTML = "";
      if (list.length === 0) {
        results.innerHTML = `
          <div class="search-result" style="cursor: default; opacity: 0.75;">
            <div class="search-result-content">
              <div class="search-result-title">Nessun risultato per \"${query}\"</div>
              <div class="search-result-description">Prova con altri termini di ricerca</div>
            </div>
          </div>`;
        results.style.display = "block";
        return;
      }

      list.forEach((item) => {
        const row = document.createElement("div");
        row.className = "search-result";
        row.innerHTML = `
          <div class="search-result-content">
            <div class="search-result-header">
              <i class="${item.icon} search-result-icon"></i>
              <span class="search-result-title">${item.title}</span>
              <span class="search-result-category">${item.category}</span>
            </div>
          </div>`;
        row.addEventListener("click", () => {
          window.location.href = item.href;
        });
        results.appendChild(row);
      });
      results.style.display = "block";
    }

    input.addEventListener("input", (event) => {
      const q = event.target.value.trim().toLowerCase();
      if (!q) {
        closeResults();
        return;
      }
      clear.style.display = "flex";
      const filtered = dataset.filter((item) =>
        item.title.toLowerCase().includes(q) ||
        item.category.toLowerCase().includes(q),
      );
      openResults(filtered, q);
    });

    clear.addEventListener("click", () => {
      input.value = "";
      closeResults();
      input.focus();
    });

    document.addEventListener("click", (event) => {
      if (!results.contains(event.target) && event.target !== input && event.target !== clear) {
        if (!input.value.trim()) closeResults();
      }
    });
  }

  async function injectNotebookModal() {
    if (document.getElementById("notebookModal")) return;
    const response = await fetch("/frontend/components/notebook-modal.html", { cache: "no-cache" });
    if (!response.ok) throw new Error("Failed loading /frontend/components/notebook-modal.html");
    const host = document.createElement("div");
    host.id = "notebookModalHost";
    host.innerHTML = await response.text();
    document.body.appendChild(host);
  }

  async function injectFooter() {
    const response = await fetch("/frontend/components/footer.html", { cache: "no-cache" });
    if (!response.ok) throw new Error("Failed loading /frontend/components/footer.html");
    const host = document.createElement("div");
    host.id = "footerHost";
    host.innerHTML = await response.text();
    document.body.appendChild(host);
    const year = document.getElementById("footerYear");
    if (year) year.textContent = String(new Date().getFullYear());
  }

  function openSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebarOverlay");
    if (!sidebar) return;
    sidebar.classList.add("expanded");
    if (overlay) overlay.classList.add("visible");
  }

  function closeSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebarOverlay");
    if (!sidebar) return;
    sidebar.classList.remove("expanded");
    if (overlay) overlay.classList.remove("visible");
  }

  window.toggleSidebar = function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    if (!sidebar) return;
    const expanded = sidebar.classList.contains("expanded");
    if (expanded) closeSidebar();
    else openSidebar();
  };

  window.showHelp = window.showHelp || function showHelp() {
    const existing = document.getElementById("helpModal");
    if (existing) {
      const modal = new bootstrap.Modal(existing);
      modal.show();
      return;
    }

    const host = document.createElement("div");
    host.innerHTML = `
      <div class="modal fade" id="helpModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
          <div class="modal-content">
            <div class="modal-header">
              <h5 class="modal-title">Aiuto</h5>
              <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
              <p class="mb-0">Seleziona un modulo dalla sidebar e compila i campi richiesti.</p>
            </div>
          </div>
        </div>
      </div>`;
    document.body.appendChild(host.firstElementChild);
    const modal = new bootstrap.Modal(document.getElementById("helpModal"));
    modal.show();
  };

  function applyPageTitle() {
    const root = document.documentElement;
    const pageTitle = root.getAttribute("data-page-title") || "Matrix Tools";
    const topbarTitle = document.getElementById("topbarTitle");
    if (topbarTitle) topbarTitle.textContent = pageTitle;
    document.title = `${pageTitle} - Matrix Tools`;
  }

  function ensureScript(src, id) {
    return new Promise((resolve, reject) => {
      if (id && document.getElementById(id)) {
        resolve();
        return;
      }
      if ([...document.scripts].some((s) => s.src === src)) {
        resolve();
        return;
      }
      const script = document.createElement("script");
      if (id) script.id = id;
      script.src = src;
      script.async = false;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(script);
    });
  }

  function waitFor(predicate, timeoutMs = 6000, stepMs = 100) {
    return new Promise((resolve, reject) => {
      const start = Date.now();
      const timer = setInterval(() => {
        if (predicate()) {
          clearInterval(timer);
          resolve();
          return;
        }
        if (Date.now() - start > timeoutMs) {
          clearInterval(timer);
          reject(new Error("Timeout waiting for dependency"));
        }
      }, stepMs);
    });
  }

  async function ensureMathEngines() {
    if (!window.MathJax) {
      window.MathJax = {
        tex: {
          inlineMath: [["$", "$"], ["\\(", "\\)"]],
          displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        },
        svg: { fontCache: "global" },
      };
    }

    await Promise.all([
      ensureScript(MATHJAX_URL, "MathJax-script"),
      window.MathfieldElement ? Promise.resolve() : ensureScript(MATHLIVE_URL),
      ensureScript("/static/js/anti-autoscroll.js"),
      ensureScript("/static/js/math-render-cache.js"),
      ensureScript("/static/js/server-status.js"),
      ensureScript("/static/js/notebook-export.js"),
      ensureScript("/static/js/notebook.js"),
    ]);

    await Promise.all([
      waitFor(() => !!(window.MathJax && typeof window.MathJax.typesetPromise === "function")).catch(
        () => {},
      ),
      waitFor(() => !!window.MathfieldElement).catch(() => {}),
    ]);

    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      try {
        await window.MathJax.typesetPromise([document.body]);
      } catch (_) {
        // no-op
      }
    }
  }

  function setupNotebookCountSync() {
    document.addEventListener("notebookUpdated", (event) => {
      const count = event?.detail?.count;
      if (typeof count !== "number") return;
      const desktopCount = document.getElementById("exercise-count-desktop");
      if (desktopCount) desktopCount.textContent = String(count);
      const genericCount = document.getElementById("notebookCount");
      if (genericCount) genericCount.textContent = String(count);
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSidebar();
  });

  document.addEventListener("DOMContentLoaded", async () => {
    try {
      await Promise.all([
        loadComponent("navbar", "/frontend/components/navbar.html"),
        injectSidebar(),
        injectNotebookModal(),
        injectFooter(),
      ]);
      applyPageTitle();
      setupNotebookCountSync();
      await ensureMathEngines();
      if (typeof window.initServerStatus === "function") {
        window.initServerStatus();
      }
      document.dispatchEvent(new CustomEvent("components:loaded"));
    } catch (err) {
      console.error(err);
    }
  });
})();
