(() => {
  const SUPPORTED_LANGS = new Set(["it", "en"]);
  const BASE_PATH = (() => {
    const seg = window.location.pathname.split("/").filter(Boolean);
    if (window.location.hostname.endsWith("github.io") && seg.length > 0) {
      return `/${seg[0]}`;
    }
    return "";
  })();
  const splitPath = () => {
    const seg = window.location.pathname.split("/").filter(Boolean);
    const offset = BASE_PATH ? 1 : 0;
    return { seg, offset };
  };
  const FLAG_BY_LANG = {
    it: {
      src: "https://flagcdn.com/w40/it.png",
      srcset: "https://flagcdn.com/w80/it.png 2x",
      alt: "Italiano",
      code: "IT",
    },
    en: {
      src: "https://flagcdn.com/w40/gb.png",
      srcset: "https://flagcdn.com/w80/gb.png 2x",
      alt: "English",
      code: "EN",
    },
  };

  const EN_MAP = {
    "Aiuto": "Help",
    "Moduli disponibili": "Available modules",
    "Il tuo laboratorio matematico interattivo": "Your interactive mathematical laboratory",
    "Esegui algoritmi passo dopo passo, salva i tuoi esercizi, genera soluzioni matematiche in LaTeX, PDF e altro ancora.": "Run algorithms step by step, save your exercises, and generate mathematical solutions in LaTeX, PDF, and more.",
    "Inizia a calcolare": "Start computing",
    "Scopri cosa puoi fare": "Explore what you can do",
    "Esplora gli algoritmi disponibili": "Explore available algorithms",
    "Ultime novità": "Latest updates",
    "Navigazione": "Navigation",
    "Input e Scorciatoie": "Input and Shortcuts",
    "Quaderno": "Notebook",
    "Riferimenti utili": "Useful Links",
    "Ricerca Operativa": "Operations Research",
    "Analisi Matematica": "Mathematical Analysis",
    "Teoria dei Controlli": "Control Theory",
    "Algebra e Logica": "Algebra and Logic",
    "Algebra Lineare": "Linear Algebra",
    "Metodo del Simplesso": "Simplex Method",
    "Condizioni di Complementarita": "Complementarity Conditions",
    "Condizioni di Complementarita'": "Complementarity Conditions",
    "Condizioni di Complementarità": "Complementarity Conditions",
    "Studio di Funzione": "Function Study",
    "Sistemi Dinamici": "Dynamical Systems",
    "Equazioni Differenziali": "Differential Equations",
    "Equazioni alle Differenze": "Difference Equations",
    "Linearizzazione": "Linearization",
    "Decomposizione Spettrale": "Spectral Decomposition",
    "Potenze Modulari": "Modular Powers",
    "Diagramma di Hasse": "Hasse Diagram",
    "Polinomi Booleani": "Boolean Polynomials",
    "Teorema Cinese del Resto": "Chinese Remainder Theorem",
    "Identita di Bezout": "Bezout Identity",
    "Identità di Bézout": "Bezout Identity",
    "Determinante con Laplace": "Laplace Determinant",
    "Autovalori e Autovettori": "Eigenvalues and Eigenvectors",
    "Sistemi Lineari": "Linear Systems",
    "Mostra descrizione": "Show description",
    "Nascondi descrizione": "Hide description",
    "Calcola": "Compute",
    "Calcola Soluzione": "Compute Solution",
    "Pulisci": "Clear",
    "Pronto": "Ready",
    "Pronto per iniziare": "Ready to start",
    "Inserisci l'espressione:": "Enter expression:",
    "Inserisci i valori e premi \"Calcola\".": "Enter values and click \"Compute\".",
    "Output": "Output",
    "Input": "Input",
    "Cerca algoritmi...": "Search algorithms...",
    "Nessun risultato": "No results",
    "Prova con altri termini di ricerca": "Try different search terms",
    "Aggiungi al quaderno": "Add to notebook",
    "Salva": "Save",
  };

  const EN_REPLACE = [
    ["Nessun risultato per", "No results for"],
    ["Cerca algoritmi...", "Search algorithms..."],
    ["Inserisci", "Enter"],
    ["Calcola", "Compute"],
    ["Pulisci", "Clear"],
    ["Mostra descrizione", "Show description"],
    ["Nascondi descrizione", "Hide description"],
    ["Pronto per iniziare", "Ready to start"],
    ["Pronto", "Ready"],
    ["Aggiungi al quaderno", "Add to notebook"],
    ["Salva nel quaderno", "Save to notebook"],
    ["Inizia a calcolare", "Start computing"],
    ["Scopri cosa puoi fare", "Explore what you can do"],
    ["Esplora gli algoritmi disponibili", "Explore available algorithms"],
  ];

  function currentLangFromPath() {
    const { seg, offset } = splitPath();
    const first = seg[offset];
    return SUPPORTED_LANGS.has(first) ? first : "it";
  }

  function targetLang(current) {
    return current === "it" ? "en" : "it";
  }

  function languagePath(lang) {
    const { seg, offset } = splitPath();
    const parts = seg.slice(offset);
    if (parts.length > 0 && SUPPORTED_LANGS.has(parts[0])) {
      parts[0] = lang;
    } else {
      parts.unshift(lang);
    }
    const trailingSlash = window.location.pathname.endsWith("/");
    const path = `${BASE_PATH}/${parts.join("/")}${trailingSlash ? "/" : ""}`;
    return `${path}${window.location.search}${window.location.hash}`;
  }

  function applyNodeTranslation(node, exactMap, partialMap) {
    const original = node.nodeValue;
    if (!original || !original.trim()) return;

    const trimmed = original.trim();
    if (Object.prototype.hasOwnProperty.call(exactMap, trimmed)) {
      node.nodeValue = original.replace(trimmed, exactMap[trimmed]);
      return;
    }

    let translated = original;
    partialMap.forEach(([from, to]) => {
      translated = translated.split(from).join(to);
    });

    if (translated !== original) {
      node.nodeValue = translated;
    }
  }

  function applyAttributesTranslation(root, exactMap, partialMap) {
    const attrs = ["placeholder", "title", "aria-label"];
    root.querySelectorAll("*").forEach((el) => {
      attrs.forEach((attr) => {
        const value = el.getAttribute(attr);
        if (!value) return;
        if (Object.prototype.hasOwnProperty.call(exactMap, value)) {
          el.setAttribute(attr, exactMap[value]);
          return;
        }
        let out = value;
        partialMap.forEach(([from, to]) => {
          out = out.split(from).join(to);
        });
        if (out !== value) el.setAttribute(attr, out);
      });
    });
  }

  function applyEnglishTranslation() {
    const root = document.body;
    if (!root) return;

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) {
      const parent = node.parentElement;
      if (!parent) continue;
      if (
        parent.closest("script, style, math-field, code, pre, kbd, .mjx-container, .MathJax") ||
        parent.hasAttribute("data-no-i18n")
      ) {
        continue;
      }
      nodes.push(node);
    }

    nodes.forEach((textNode) => applyNodeTranslation(textNode, EN_MAP, EN_REPLACE));
    applyAttributesTranslation(root, EN_MAP, EN_REPLACE);
  }

  function updateToggleLabel(lang) {
    const flag = document.getElementById("languageToggleFlag");
    const toggleBtn = document.getElementById("languageToggleBtn");
    const nextLang = targetLang(lang);
    const langAsset = FLAG_BY_LANG[lang];
    const nextAsset = FLAG_BY_LANG[nextLang];

    if (flag && langAsset) {
      flag.src = langAsset.src;
      flag.srcset = langAsset.srcset;
      flag.alt = langAsset.alt;
    }

    if (toggleBtn && nextAsset) {
      const title = lang === "it" ? "Switch to English" : "Passa a Italiano";
      toggleBtn.setAttribute("title", title);
      toggleBtn.setAttribute("aria-label", title);
      toggleBtn.setAttribute("data-next-lang", nextAsset.code);
    }
  }

  window.toggleLanguage = function toggleLanguage() {
    const current = currentLangFromPath();
    const next = targetLang(current);
    window.location.href = languagePath(next);
  };

  function applyLanguage(lang) {
    document.documentElement.setAttribute("lang", lang);
    updateToggleLabel(lang);
    if (lang !== "en") return;
    applyEnglishTranslation();
  }

  function scheduleEnglishReapply() {
    if (currentLangFromPath() !== "en") return;
    requestAnimationFrame(() => applyLanguage("en"));
    setTimeout(() => applyLanguage("en"), 80);
    setTimeout(() => applyLanguage("en"), 250);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const lang = currentLangFromPath();
    applyLanguage(lang);
  });

  // Components are injected after DOMContentLoaded in static mode.
  document.addEventListener("components:loaded", () => {
    scheduleEnglishReapply();
  });

  // Translate dynamically injected/updated content (e.g. search results, async blocks).
  const observer = new MutationObserver(() => {
    if (currentLangFromPath() !== "en") return;
    scheduleEnglishReapply();
  });

  if (document.documentElement) {
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: false,
    });
  }
})();
