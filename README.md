# Data Science Interview Prep

A single, hosted **study hub** that ties together six independent
[MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/)
knowledge bases. Open the hub, click a card, and you're routed into that topic's own site; a
**← Study Hub** link in each sub-site's top navigation brings you back.

🔗 **Live site:** https://s-samarth.github.io/DataSciencePreparation/

---

## Table of contents

- [What's inside](#whats-inside)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Local development](#local-development)
- [Making changes](#making-changes)
- [Deployment](#deployment)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Raw notes](#raw-notes)

---

## What's inside

| Sub-site            | Folder           | What it covers |
|---------------------|------------------|----------------|
| **Mathematics**     | `Mathematics/`   | Probability, inferential statistics, and linear algebra for ML |
| **Classical ML**    | `ClassicalML/`   | Linear models, trees, ensembles, unsupervised learning, evaluation metrics |
| **Deep Learning**   | `DeepLearning/`  | NN mechanics, gradient flow, optimization, losses, CNNs, vision/generative, sequences & attention |
| **LLM Study Notes** | `LLM/`           | Architecture, training, alignment, PEFT, inference, serving, 2026 landscape |
| **Agentic AI**      | `AgenticAI/`     | RAG, agent loops, LangGraph, protocols (MCP/A2A), eval, serving, system design |
| **ML Case Studies** | `MLCaseStudies/` | 20 ML & GenAI system-design case studies with interview transcripts |
| **Productionizing ML** | `ProductionizingML/` | Production ML system design, data/training/serving, the production loop, infra, build-vs-buy, implementation masterclass |
| **Hub**             | `Hub/`           | The landing page with cards linking into the seven sites above |

Each folder is a **fully self-contained MkDocs site** — its own `mkdocs.yml`, `docs/`, theme,
navigation, and search index. They are **not** merged into one giant navigation. The hub just
links to them, and at deploy time all eight are stacked side by side as sub-folders of one
GitHub Pages site.

There is also a **`RawNotes/`** folder — the original source material (standalone HTML pages,
Jupyter notebooks, and Markdown) the sites were distilled from. It is kept in the repo but is
**not** built or deployed. See [Raw notes](#raw-notes) for what's in it and how to use it.

---

## How it works

```
                          ┌─────────────────────────────┐
                          │  Hub  (cards landing page)   │   /DataSciencePreparation/
                          └──────────────┬──────────────┘
   ┌──────────┬─────────┬─────────┬───┼───┬─────────┬─────────────┬───────────────┐
   ▼          ▼         ▼         ▼   ▼   ▼         ▼             ▼               ▼
Mathematics ClassicalML DeepLearning LLM AgenticAI MLCaseStudies ProductionizingML (← Study
                                                                                    Hub returns)
```

(Each sub-site is served at `/DataSciencePreparation/<Folder>/`.)

The trick that keeps this simple: **MkDocs Material emits relative links** for all assets,
navigation, and search. That makes each built site *relocatable* — it runs correctly from any
sub-folder without rewriting a single doc. The only base-URL-aware settings are:

1. **`site_url`** in each `mkdocs.yml` — drives canonical links and the sitemap.
2. The **`← Study Hub`** nav link — an absolute URL back to the hub root.

`build.sh` builds all eight sites, then assembles them into one `_site/` folder (hub at the
root, each sub-site in its own directory). GitHub Actions runs that on every push and
publishes the result. No docs are duplicated or rewritten — only built and copied.

---

## Repository layout

```
DataSciencePreparation/
├── Hub/                       # central landing site
│   ├── mkdocs.yml             #   site_url = repo root
│   └── docs/
│       ├── index.md           #   the 7 cards
│       └── assets/extra.css   #   shared styling (copied from the study sites)
├── Mathematics/               # ┐
│   ├── mkdocs.yml             # │ each study site is independent:
│   └── docs/                  # │   site_url + "← Study Hub" nav entry + content
├── ClassicalML/               # │
├── DeepLearning/              # │
├── LLM/                       # │
├── AgenticAI/                 # │
├── MLCaseStudies/             # │
├── ProductionizingML/         # ┘
├── RawNotes/                  # source material (HTML / notebooks / Markdown) — NOT built or deployed
├── build.sh                   # builds all 8 sites → assembles ./_site
├── .github/workflows/deploy.yml   # CI: build + publish to GitHub Pages on push to main
├── .gitignore                 # ignores build output (site/, _site/, _serve/) + envs
└── README.md
```

> **Never committed:** `*/site/`, `_site/`, `_serve/`, and any virtual env. These are build
> artifacts, regenerated on demand by `build.sh` or CI. They're listed in `.gitignore`.

---

## Quick start

```bash
# 1. preview one site locally (assumes mkdocs-material is installed — see below)
mkdocs serve -f LLM/mkdocs.yml          # → http://127.0.0.1:8000/

# 2. when happy, ship it
git add . && git commit -m "docs: update LLM notes" && git push
```

That's the whole everyday loop. Details for each step follow.

---

## Local development

### Prerequisites

You need `mkdocs-material` available on your `PATH`. Use **either** your existing
environment **or** a throwaway one — both work:

```bash
# Option A — conda (what this repo's author uses)
conda activate <your-env>      # an env that has mkdocs-material
mkdocs --version               # confirm it's available

# Option B — disposable virtualenv (git-ignored)
python3 -m venv .venv
source .venv/bin/activate
pip install mkdocs-material
```

### Preview one site (fast, live-reload)

Best while writing — it watches the docs **and** `mkdocs.yml` and reloads on save:

```bash
mkdocs serve -f LLM/mkdocs.yml          # → http://127.0.0.1:8000/
```

Swap `LLM` for `Hub`, `ClassicalML`, `AgenticAI`, or `MLCaseStudies`.

### Preview the whole hub (cards + routing, exactly like production)

The cards and back-links assume the `/DataSciencePreparation/` base path, so serve the
combined build *under that path*:

```bash
./build.sh                                      # builds all 5 → ./_site
mkdir -p _serve/DataSciencePreparation
cp -R _site/. _serve/DataSciencePreparation/
python3 -m http.server 8000 -d _serve
# open http://localhost:8000/DataSciencePreparation/
```

`_site/` and `_serve/` are git-ignored scratch — delete them anytime.

---

## Making changes

| I want to…                       | Do this |
|----------------------------------|---------|
| **Edit a page**                  | Change the Markdown under the relevant `*/docs/` folder. |
| **Add a page**                   | Drop a new `.md` in that site's `docs/`, then add it to that site's `mkdocs.yml` `nav:`. |
| **Edit the hub landing cards**   | Edit `Hub/docs/index.md`. |
| **Restyle a site**               | Edit that site's `docs/assets/extra.css`. The seven study sites share the same Material theme block in their `mkdocs.yml`. |
| **Add a 5th study site**         | Create the new MkDocs folder with a `site_url` + `← Study Hub` nav entry, add a card in `Hub/docs/index.md`, and add it to the loops in `build.sh`. |

After any change: commit and push to `main`. CI rebuilds and redeploys automatically.

---

## Deployment

Hosting is **GitHub Pages**, built and published by GitHub Actions
(`.github/workflows/deploy.yml`). On every push to `main` it installs `mkdocs-material`, runs
`build.sh`, and publishes the assembled `_site/`. You never build or upload anything by hand.

### First-time setup (once)

> Assumes GitHub user **`s-samarth`** and repo name **`DataSciencePreparation`** — these are
> baked into every `site_url`. Using different values? See [Configuration reference](#configuration-reference) first.

1. Create an **empty** GitHub repo named **`DataSciencePreparation`** (no README/.gitignore —
   this repo already has them).
2. Push this repo:
   ```bash
   git add .
   git commit -m "feat: study hub + seven MkDocs sites with Pages deploy"
   git branch -M main
   git remote add origin https://github.com/s-samarth/DataSciencePreparation.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source = "GitHub Actions"**.
4. Open the **Actions** tab and watch *"Deploy site to GitHub Pages"* finish (~1–2 min).
5. Visit https://s-samarth.github.io/DataSciencePreparation/.

### Ongoing deployments

Just push to `main`:

```bash
git add .
git commit -m "docs: <what changed>"
git push
```

The workflow redeploys automatically. You can also run it on demand from the **Actions** tab
→ **Run workflow** (it has `workflow_dispatch` enabled).

### Optional: custom domain later

Add a `CNAME` file containing your domain into `Hub/docs/` (so it lands at the site root),
point your DNS at GitHub Pages, set the custom domain in **Settings → Pages**, and update the
base URL in the eight `mkdocs.yml` files from the `github.io` path to your domain root.

---

## Configuration reference

The deployed base URL `https://s-samarth.github.io/DataSciencePreparation/` is hardcoded in
**eight** `mkdocs.yml` files — as each site's `site_url:` and (in the seven sub-sites) as the
`← Study Hub` nav link.

If you rename the repo, switch accounts, or move to a custom domain, update it everywhere:

```bash
# find every occurrence
grep -rl "s-samarth.github.io" */mkdocs.yml

# example: change the GitHub user across all configs (macOS sed)
sed -i '' 's#s-samarth.github.io#NEWUSER.github.io#g' */mkdocs.yml
```

The repo name appears as the `/DataSciencePreparation/` path segment in those same URLs —
update it the same way if you rename the repo.

---

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| **A "Sign in" / auth popup from an external domain** appears when serving | A third-party script in `extra_javascript` is compromised or requires auth. The known offender `polyfill.io` was already removed; MathJax loads from `cdn.jsdelivr.net` / `unpkg.com`, which are fine. Don't re-add `polyfill.io`. |
| **Math doesn't render** | Check that site's `extra_javascript` still includes the MathJax CDN line and its local `assets/mathjax*.js` config. |
| **Cards or back-links 404 locally** | You're serving at `/` instead of the project base path. Use the [whole-hub preview](#preview-the-whole-hub-cards--routing-exactly-like-production) steps so it's served under `/DataSciencePreparation/`. |
| **Links broken after renaming the repo** | Update the base URL everywhere — see [Configuration reference](#configuration-reference). |
| **`mkdocs: command not found`** | Activate the env that has `mkdocs-material` (conda or `.venv`). |
| **Stale page after deploy** | Hard-refresh the browser (Cmd/Ctrl + Shift + R) to bypass cached assets. |

---

## Raw notes

`RawNotes/` holds the **original source material** the polished study sites were distilled
from — the standalone reference HTML pages, Jupyter notebooks, and Markdown files created
while studying. It is kept in the repo on purpose, but it is **not** part of any MkDocs
site: nothing here is built, themed, navigated, or deployed, and the hub and its sub-sites
never link into it.

Reach for it when you want the **raw, unprocessed version** — the original standalone page or
the runnable notebook — rather than the curated, cross-linked site.

It's organized into one subfolder per topic, mirroring the study sites (plus **`DSA`**, a
data-structures-and-algorithms set that exists only as raw notes, with no published site):

```
RawNotes/
├── Mathematics/        # probability, inferential statistics, linear algebra  (HTML)
├── ClassicalML/        # classical ML references + notebooks                  (HTML, ipynb, md)
├── DeepLearning/       # neural nets, CNN/RNN                                 (HTML, ipynb)
├── AgenticAI/          # RAG, agents, LangGraph, protocols                    (HTML, ipynb, py)
├── MLCaseStudies/      # the 20 case studies as raw Markdown                  (md)
├── ProductionizingML/  # system design, data, training, serving, infra       (HTML, ipynb)
└── DSA/                # data structures & algorithms masterclasses           (ipynb)  ← no site
```

**How to open each file type:**

- **`.html`** — open directly in a browser (double-click, or `open <file>.html`). These are
  self-contained standalone pages; no server or build step needed.
- **`.ipynb`** — open in Jupyter / VS Code / Colab to read or run the code cells
  (e.g. `jupyter notebook <file>.ipynb`).
- **`.md` / `.py`** — open in any text editor or Markdown viewer.

Roughly 24 HTML references, 21 Markdown files, 19 notebooks, and 1 Python script across the
seven topic folders. Think of these as the archival "working notes"; the deployed sites are
the cleaned-up version of the same material.
