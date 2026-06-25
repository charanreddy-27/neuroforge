# Deploying NeuroForge

NeuroForge is a **static site** — plain HTML, CSS, and JS with no build step and no dependencies. That makes deployment about as easy as it gets. This guide covers Vercel (recommended), plus the short list of things only **you** can do.

---

## 🚀 Deploy to Vercel (recommended)

### Option A — Git integration (best; auto-deploys on every push)

1. Push this repo to GitHub (see the checklist below if it isn't there yet).
2. Go to **[vercel.com/new](https://vercel.com/new)** and **Import** the `neuroforge` repository.
3. Configure the project:
   - **Framework Preset:** `Other`
   - **Build Command:** *(leave empty)* — there is nothing to build.
   - **Output Directory:** *(leave empty / `.`)* — the site is served from the repo root.
   - **Install Command:** *(leave empty)* — there are no dependencies.
4. Click **Deploy**. You'll get a `*.vercel.app` URL in a few seconds.
5. Every future `git push` to `main` redeploys automatically. Pull requests get preview URLs.

> The included [`vercel.json`](vercel.json) already sets **clean URLs** (so `/about` works without `.html`), sensible **cache headers**, and basic **security headers**. You don't need to touch the dashboard for any of that.

### Option B — Vercel CLI (one-off from your machine)

```bash
npm i -g vercel     # once
vercel              # from the project root → follow the prompts (preview deploy)
vercel --prod       # promote to production
```

When asked about build settings, accept the defaults / leave them blank — there's no build.

---

## 🌐 Custom domain (optional)

1. In the Vercel project → **Settings → Domains → Add**.
2. Enter your domain (e.g. `neuroforge.charanreddy.dev`).
3. Add the DNS record Vercel shows you at your registrar:
   - **Subdomain** → `CNAME` to `cname.vercel-dns.com`
   - **Apex/root** → the `A` record Vercel provides
4. Wait for DNS to propagate (minutes to a couple of hours). Vercel issues the SSL cert automatically.
5. If you change the canonical domain, update the absolute URLs in the `<head>` of `index.html`, `about.html`, and `about-project.html` (the `og:image`, `og:url`, and `canonical` tags) so social previews point at the right place.

---

## 🧩 Environment variables / secrets

**None.** NeuroForge runs entirely client-side and talks to no backend. The contact form on the About page composes a pre-filled `mailto:` link — nothing is sent to a server, so there are no keys to manage.

---

## ✅ Manual checklist — the things only you can do

These are the few steps that need a human (your accounts, your assets, your judgment):

1. **Confirm the GitHub repo URL.** The site and docs link to `https://github.com/charanreddy-27/neuroforge`. If your repo slug is different, update the links in `index.html` (the header GitHub button), `about.html`, `about-project.html`, and the docs.
2. **Push to GitHub** (if you haven't):
   ```bash
   git add -A
   git commit -m "Portfolio polish: about pages, SEO, docs"
   git remote add origin https://github.com/charanreddy-27/neuroforge.git
   git push -u origin main
   ```
3. **Deploy on Vercel** using Option A above.
4. **Drop in a screenshot/GIF.** Record ~5 seconds of the spiral being solved and add it to the README (replace the placeholder note) — it's the single highest-leverage thing for a recruiter skim. A clean PNG/GIF in the repo root, referenced from `README.md`, is enough.
5. **Verify the social card.** After deploy, paste your live URL into:
   - LinkedIn Post Inspector → <https://www.linkedin.com/post-inspector/>
   - Twitter/X Card Validator (or just paste into a draft)
   - The card should show `og.png`. If it's stale, those tools have a "re-scrape" button.
6. **Check the Résumé link.** The About page's "Résumé" button currently points at your portfolio (`charanreddy.dev`). If you have a dedicated resume PDF or `/resume` route, point it there.
7. **Add the LinkedIn post link.** On `about-project.html`, the "LinkedIn write-up" button is a placeholder. Once you post about NeuroForge, replace it with the post URL.
8. **(Optional) Custom domain + DNS** as above.
9. **Post it.** Share the live link on LinkedIn — lead with the gradient-bug story or the "the library *is* the demo" angle; both land well.

---

## 🔁 Other hosts (if not Vercel)

It's a static folder, so anything works:

- **Netlify** — drag the folder onto the dashboard, or connect the repo with no build command.
- **GitHub Pages** — push and enable Pages on the branch. (Note: clean URLs from `vercel.json` won't apply; links use `.html` so they still work.)
- **Cloudflare Pages** — connect the repo, build command empty, output directory `/`.

---

## 🩺 Post-deploy smoke test

- [ ] Home page trains a spiral when you hit **Train**.
- [ ] `/about` and `/about-project` load (clean URLs) and link back to the app.
- [ ] Header **GitHub** button and footer links go to the right places.
- [ ] Social card renders with `og.png` in the inspectors above.
- [ ] Looks right on a phone (resize the window narrow to sanity-check).
- [ ] Contact form opens your mail app pre-filled.
