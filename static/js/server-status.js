(() => {
  const RETRY_MS = 3000;
  const COLD_START_GRACE_MS = 20000;

  let online = false;
  let startedAt = 0;
  let retryTimer = null;

  function resolveApiBase() {
    const explicit = window.MATRIX_API_BASE_URL || document.documentElement.getAttribute("data-api-base");
    if (explicit && explicit.trim()) return explicit.replace(/\/$/, "");
    return window.location.origin;
  }

  function healthUrl() {
    return `${resolveApiBase()}/health`;
  }

  function setBadge(status, label) {
    const badge = document.getElementById("serverStatusBadge");
    if (!badge) return;
    badge.classList.remove("status-online", "status-booting", "status-offline");
    badge.classList.add(status);
    badge.setAttribute("aria-label", label);
    badge.setAttribute("data-status-label", label);
  }

  function isApiActionButton(btn) {
    if (!btn) return false;

    const txt = (btn.textContent || "").trim().toLowerCase();
    const id = (btn.id || "").toLowerCase();
    const onclick = (btn.getAttribute("onclick") || "").toLowerCase();

    const positive = ["calcola", "compute", "verifica", "risolvi", "solve", "linearizza", "analizza", "genera"];
    const negative = ["pulisci", "clear", "reset", "descrizione", "quaderno", "notebook", "aiuto", "help", "export", "menu"];

    const looksPositive = positive.some((k) => txt.includes(k) || id.includes(k) || onclick.includes(k));
    const looksNegative = negative.some((k) => txt.includes(k) || id.includes(k) || onclick.includes(k));

    if (looksPositive && !looksNegative) return true;
    if (/\/api\//.test(onclick) && !looksNegative) return true;
    return false;
  }

  function setApiButtonsDisabled(disabled) {
    const buttons = Array.from(document.querySelectorAll("button, input[type='submit'], input[type='button']"));
    buttons.forEach((btn) => {
      if (!isApiActionButton(btn)) return;
      btn.disabled = disabled;
      btn.classList.toggle("api-disabled", disabled);
      if (disabled) {
        btn.dataset.prevTitle = btn.getAttribute("title") || "";
        btn.setAttribute("title", "Server offline");
      } else if (btn.dataset.prevTitle !== undefined) {
        btn.setAttribute("title", btn.dataset.prevTitle);
      }
    });
  }

  async function pingOnce() {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const response = await fetch(healthUrl(), {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      clearTimeout(timeout);
      if (!response.ok) return false;
      const data = await response.json().catch(() => ({}));
      return data.status === "ok" || data.ok === true;
    } catch (_) {
      return false;
    }
  }

  async function monitor() {
    const ok = await pingOnce();

    if (ok) {
      online = true;
      setBadge("status-online", "Server online");
      setApiButtonsDisabled(false);
      if (retryTimer) {
        clearInterval(retryTimer);
        retryTimer = null;
      }
      return;
    }

    const elapsed = Date.now() - startedAt;
    if (!online && elapsed < COLD_START_GRACE_MS) {
      setBadge("status-booting", "Avvio server...");
    } else {
      setBadge("status-offline", "Server offline");
    }
    setApiButtonsDisabled(true);

    if (!retryTimer) {
      retryTimer = setInterval(monitor, RETRY_MS);
    }
  }

  window.initServerStatus = function initServerStatus() {
    startedAt = Date.now();
    setBadge("status-booting", "Avvio server...");
    setApiButtonsDisabled(true);
    monitor();
  };
})();
