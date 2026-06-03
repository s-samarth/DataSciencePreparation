// MathJax v3 configuration for pymdownx.arithmatex (generic mode).
// No polyfill.io dependency: MathJax v3 ships its own polyfills.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset on instant navigation (the navigation.instant feature swaps page
// content without a full reload). The reset chain is required: without
// clearCache() the CHTML output's glyph stylesheet detaches across instant
// navigations and insertRule() fails on null, leaving math laid out but blank.
document$.subscribe(() => {
  if (typeof MathJax === "undefined" || !MathJax.typesetPromise) {
    return; // MathJax CDN not loaded yet; it auto-typesets on first load
  }
  if (MathJax.startup && MathJax.startup.output && MathJax.startup.output.clearCache) {
    MathJax.startup.output.clearCache();
  }
  if (MathJax.typesetClear) MathJax.typesetClear();
  if (MathJax.texReset) MathJax.texReset();
  MathJax.typesetPromise();
});
