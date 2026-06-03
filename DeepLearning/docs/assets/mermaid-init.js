// Render Mermaid diagrams reliably, including after instant navigation.
// Material lazy-loads the mermaid library; we drive the render ourselves so
// diagrams appear regardless of auto-init timing. We only target nodes that
// have not been processed yet and swallow the benign promise rejection that
// occurs if another renderer races us on the same node.
document$.subscribe(() => {
  const render = (attempt) => {
    if (window.mermaid && typeof window.mermaid.run === "function") {
      const nodes = document.querySelectorAll(".mermaid:not([data-processed])");
      if (nodes.length) {
        window.mermaid.run({ nodes }).catch(() => {});
      }
    } else if (attempt < 25) {
      setTimeout(() => render(attempt + 1), 200);
    }
  };
  render(0);
});
