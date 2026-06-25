# Changelog

All notable changes to NeuroForge. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); dates are `YYYY-MM-DD`.

## [1.1.0] — 2026-06-25

Portfolio-grade polish and narrative pages.

### Added
- **About the author** page (`about.html`) — intro, "what I learned building this", a portfolio tease, contact card, and a backend-free `mailto:` contact form.
- **About the project** page (`about-project.html`) — the why, a build timeline, key features, technical decisions/challenges, and the tech stack with rationale.
- **`pages.css`** — a content-page design layer that mirrors the app's design tokens.
- **Social sharing** — Open Graph + Twitter Card meta on every page, plus an on-brand `og.svg` / `og.png` (1200×630) share image.
- **Navigation** — header links between the app and the narrative pages; a subtle "Built by Charan" footer link.
- **`vercel.json`** — clean URLs, cache headers, and basic security headers.
- **Docs** — `PROJECT_DEEP_DIVE.md`, `INTERVIEW_PREP.md`, `DEPLOYMENT.md`, and this changelog.

### Changed
- Rewrote `README.md` with a live demo link, share image, tech-stack table, and an "About the developer" block.

### Fixed
- Header **GitHub** link pointed at `#`; now links to the repository.
- README live-demo link was empty; now points at the deployment.

### Accessibility
- Added descriptive `aria-label`s to the boundary, network, and chart canvases.
- Labelled the GitHub button and decorative SVGs; ensured visible focus states throughout.

## [1.0.0] — 2026-06

Initial release.

### Added
- From-scratch neural-network engine: forward pass, manual backpropagation, and SGD / Momentum / Adam optimizers (`src/nn.js`).
- Five datasets with adjustable noise, sample count, and train/test split; seven engineered input features (`src/datasets.js`).
- Live decision-boundary heatmap, weight graph, and loss/accuracy charts on raw canvas.
- Live architecture editing, optimizer controls, presets, model export, and keyboard shortcuts.
- Numerical gradient check confirming analytic gradients to ~2e-10.
