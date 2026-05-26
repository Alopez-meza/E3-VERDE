<<<<<<< HEAD
# VERDE Materials DB

A minimalist, modern, fully static website for **VERDE Materials DB** — a curated molecular materials database focused on photoredox chemistry, excited-state properties, molecular geometries, and machine-learning-ready data.

---

## File Structure

```
/
├── index.html    ← All HTML (sections, molecule cards, navigation)
├── styles.css    ← All styling and the design system (colors, layout, components)
├── script.js     ← All JavaScript (nav, search/filter, animations, copy citation)
└── README.md     ← This file
```

No build tools, no npm, no dependencies. Open `index.html` in a browser and it works.

---

## GitHub Pages Deployment

### Option A — Project site (recommended)

Your site will be at: `https://USERNAME.github.io/REPOSITORY-NAME/`

**Step 1 — Create a GitHub repository**

- Go to [github.com](https://github.com) → **New repository**
- Choose any name, e.g. `verde-materials-db`
- Set visibility to **Public** (required for free GitHub Pages)

**Step 2 — Add the files**

Upload `index.html`, `styles.css`, `script.js`, and `README.md` to the **root** of the repository.

Using git on the command line:

```bash
git init
git add index.html styles.css script.js README.md
git commit -m "Initial commit: VERDE Materials DB site"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY-NAME.git
git push -u origin main
```

**Step 3 — Enable GitHub Pages**

1. Go to your repository on GitHub
2. Click **Settings** → **Pages** (left sidebar, under "Code and automation")
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**
4. Set **Branch** → `main`
5. Set **Folder** → `/ (root)`
6. Click **Save**

**Step 4 — Wait and open**

GitHub will build and deploy the site (usually 1–3 minutes). A green banner will appear in Settings → Pages when it's ready.

Open your site at:

```
https://USERNAME.github.io/REPOSITORY-NAME/
```

---

### Option B — Personal / organization site

Your site will be at: `https://USERNAME.github.io/` (no repository name in the URL)

1. Create a repository named exactly **`USERNAME.github.io`**
   - Example: if your username is `lopezlab`, name it `lopezlab.github.io`
2. Add all four files to the repository root
3. Push to `main` — GitHub Pages is automatically enabled for this repository type
4. Your site is live at `https://USERNAME.github.io/`

---

## How to Edit the Site

### Change text content

Open `index.html` in any text editor. Each section has a comment block explaining what is below it, for example:

```html
<!-- ============================================================
     HERO SECTION
     To edit: update the main title, subtitle, and button labels.
============================================================ -->
```

### Change the color palette

Open `styles.css` and find **Section 1: CSS Variables**. Edit the values inside `:root {}`:

```css
:root {
  --c-accent:   #52b788;   /* main lime-green accent */
  --c-forest:   #1b4332;   /* deep forest green      */
  --c-bg:       #f3f6f2;   /* page background         */
  /* … */
}
```

### Add a new molecule card

1. Open `index.html`
2. Find the `<!-- DATABASE PREVIEW SECTION -->` comment
3. Copy any existing `.molecule-card` block
4. Paste it inside `<div class="molecule-grid" id="moleculeGrid">`
5. Update:
   - `data-name="…"` — lowercase keywords used by the search bar
   - `data-family="…"` — must match an `<option value="…">` in the dropdown
   - The molecule name, property values, and tags

The search and family filter will include the new card automatically.

### Add a new family filter option

In `index.html`, find the `<select id="familyFilter">` element and add:

```html
<option value="Your Family Name">Display Label</option>
```

Then add `data-family="Your Family Name"` to the corresponding molecule cards.

### Update the BibTeX citation

Open `script.js`, find **Section 6**, and update the `BIBTEX` constant:

```js
const BIBTEX = `@article{verde2024,
  title   = {Your actual paper title},
  author  = {Lopez, Steven A. and others},
  ...
}`;
```

### Update footer links

Open `index.html`, find the `<!-- CONTACT & FOOTER SECTION -->` comment, and replace the `href="#"` placeholders with your actual GitHub, documentation, and lab page URLs.

### Animate a new element on scroll

Add `class="reveal"` to any HTML element. Optionally add `reveal-delay-1` through `reveal-delay-4` to stagger multiple elements:

```html
<div class="my-card reveal reveal-delay-2">…</div>
```

---

## No Build Required

This site requires zero installation. There is no package.json, no node_modules, no bundler, and no build command. Just open `index.html` directly in a browser, or push to GitHub Pages.

---

## License

© VERDE Materials DB · Lopez Lab · All rights reserved.  
Data freely available for research use.
=======
# E(3)-VERDE
E(3)-equivariant model for predicting organic photocatalyst redox windows from molecular geometries.
>>>>>>> 71000bcba8ff51dd2c7ac57d352e911c45fb7ebb
