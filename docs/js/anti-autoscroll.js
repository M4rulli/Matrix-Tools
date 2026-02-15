// === Anti auto-scroll patch (prevents MathLive from jumping on tab-switch) ===
(() => {
  if (window.__antiAutoscrollInstalled) {
    return;
  }
  window.__antiAutoscrollInstalled = true;

  // Use window state to avoid duplicate lexical declarations if the file is injected twice.
  var antiScrollState = window.__antiAutoscrollState || {
    windowHasFocus: true,
    isTabSwitching: false,
  };
  window.__antiAutoscrollState = antiScrollState;

  window.addEventListener("blur", () => {
    antiScrollState.windowHasFocus = false;
    antiScrollState.isTabSwitching = true;
  });

  window.addEventListener("focus", () => {
    antiScrollState.windowHasFocus = true;
    setTimeout(() => {
      antiScrollState.isTabSwitching = false;
    }, 150);
  });

  const originalScrollIntoView = Element.prototype.scrollIntoView;
  Element.prototype.scrollIntoView = function (options) {
    if (antiScrollState.windowHasFocus && !antiScrollState.isTabSwitching) {
      return originalScrollIntoView.call(this, options);
    }
    // suppress scroll otherwise
  };
})();
// === End anti auto-scroll patch ===
