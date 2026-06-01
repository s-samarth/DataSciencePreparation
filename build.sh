#!/usr/bin/env bash
# Build the hub + all four sub-sites and assemble them into a single ./_site folder.
# Used by CI (.github/workflows/deploy.yml) and runnable locally for a smoke test.
set -euo pipefail

cd "$(dirname "$0")"

SITES=(Hub MLCaseStudies LLM AgenticAI ClassicalML)

echo "==> Building each MkDocs site"
for s in "${SITES[@]}"; do
  echo "  - $s"
  mkdocs build --clean -f "$s/mkdocs.yml"
done

echo "==> Assembling combined output into ./_site"
rm -rf _site
mkdir -p _site
cp -R Hub/site/. _site/                       # hub at the root
for s in MLCaseStudies LLM AgenticAI ClassicalML; do
  cp -R "$s/site" "_site/$s"                   # each sub-site in its own folder
done

echo "==> Done. Combined site is in ./_site"
