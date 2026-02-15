(() => {
  let mathJaxPatched = false;
  let mathLivePatched = false;

  const mjSignatureCache = new WeakMap();

  function hashString(value) {
    let hash = 2166136261;
    for (let i = 0; i < value.length; i++) {
      hash ^= value.charCodeAt(i);
      hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return (hash >>> 0).toString(16);
  }

  function elementSignature(el) {
    const html = el?.innerHTML || "";
    return `${html.length}:${hashString(html)}`;
  }

  function normalizeElements(elements) {
    if (!elements) return null;
    if (Array.isArray(elements)) return elements.filter(Boolean);
    if (elements instanceof Element) return [elements];
    if (typeof elements.length === "number") return Array.from(elements).filter(Boolean);
    return null;
  }

  function patchMathJax() {
    if (mathJaxPatched || !window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
      return;
    }

    const originalTypesetPromise = window.MathJax.typesetPromise.bind(window.MathJax);
    window.MathJax.typesetPromise = function patchedTypesetPromise(elements) {
      const normalized = normalizeElements(elements);
      if (!normalized) {
        return originalTypesetPromise(elements);
      }

      const dirty = normalized.filter((el) => mjSignatureCache.get(el) !== elementSignature(el));
      if (dirty.length === 0) {
        return Promise.resolve();
      }

      return originalTypesetPromise(dirty).then((result) => {
        dirty.forEach((el) => mjSignatureCache.set(el, elementSignature(el)));
        return result;
      });
    };

    window.invalidateMathRenderCache = function invalidateMathRenderCache(target) {
      if (!target) return;
      const list = normalizeElements(target);
      if (!list) return;
      list.forEach((el) => mjSignatureCache.delete(el));
    };

    mathJaxPatched = true;
  }

  function patchMathLive() {
    if (mathLivePatched || !window.MathfieldElement || !window.MathfieldElement.prototype) {
      return;
    }

    const proto = window.MathfieldElement.prototype;
    if (typeof proto.setValue !== "function" || proto.__cachedSetValuePatched) {
      return;
    }

    const originalSetValue = proto.setValue;
    proto.setValue = function patchedSetValue(nextValue, ...rest) {
      if (typeof nextValue === "string" && rest.length === 0 && typeof this.getValue === "function") {
        const current = this.getValue();
        if (current === nextValue) return;
      }
      return originalSetValue.call(this, nextValue, ...rest);
    };

    proto.__cachedSetValuePatched = true;
    mathLivePatched = true;
  }

  function tryPatchAll() {
    patchMathJax();
    patchMathLive();
    if (mathJaxPatched && mathLivePatched) {
      clearInterval(intervalId);
    }
  }

  const intervalId = setInterval(tryPatchAll, 250);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tryPatchAll);
  } else {
    tryPatchAll();
  }
})();
