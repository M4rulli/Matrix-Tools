// === Anti auto-scroll patch (prevents MathLive from jumping on tab-switch) ===
let windowHasFocus = true;
let isTabSwitching = false;

window.addEventListener('blur', () => {
  windowHasFocus = false;
  isTabSwitching = true;
});

window.addEventListener('focus', () => {
  windowHasFocus = true;
  setTimeout(() => { isTabSwitching = false; }, 150);
});

const originalScrollIntoView = Element.prototype.scrollIntoView;
Element.prototype.scrollIntoView = function (options) {
  if (windowHasFocus && !isTabSwitching) {
    return originalScrollIntoView.call(this, options);
  }
  // suppress scroll otherwise
};
// === End anti auto-scroll patch ===