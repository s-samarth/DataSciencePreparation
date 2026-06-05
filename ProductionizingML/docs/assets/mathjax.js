// Arithmatex (generic mode) configuration for MathJax v3.
// Loaded from jsDelivr in mkdocs.yml, with no external polyfill CDN dependency.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset on instant navigation (navigation.instant swaps page content
// without a full reload, so MathJax must run again on each page change).
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
