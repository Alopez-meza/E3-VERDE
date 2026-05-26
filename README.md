# VERDE Materials DB

A minimalist, modern, fully static website for **VERDE Materials DB** — a curated molecular materials database focused on ground and excited-state properties, molecular geometries, and machine-learning-ready data from `verde+PCs-02142026.csv`.

---

## File Structure

```
/
├── index.html    ← All HTML (sections, database table, navigation)
├── styles.css    ← All styling and the design system (colors, layout, components)
├── script.js     ← All JavaScript (nav, search/filter, hero movie, property distributions, 3D gallery, animations, copy citation)
├── hero-molecule-movie-data.js ← Random S0 XYZ samples used by the home animation
├── molecule-gallery-data.js ← All available S0 XYZ geometries used by the 3D gallery
├── property-distributions-data.js ← Aggregated histograms generated from the CSV
├── generate-molecule-gallery.py ← Regenerates gallery geometries when the CSV changes
├── generate-property-distributions.py ← Regenerates property histograms when the CSV changes
├── verde+PCs-02142026.csv ← Source data file for the database preview and download link
└── README.md     ← This file
```

No build tools or npm dependencies are required. The 3D gallery uses 3Dmol.js from a CDN, so it needs internet access when opened.

---

## GitHub Pages Deployment

### Option A — Project site (recommended)

Your site will be at: `https://USERNAME.github.io/REPOSITORY-NAME/`

**Step 1 — Create a GitHub repository**

- Go to [github.com](https://github.com) → **New repository**
- Choose any name, e.g. `verde-materials-db`
- Set visibility to **Public** (required for free GitHub Pages)

**Step 2 — Add the files**

Upload `index.html`, `styles.css`, `script.js`, `hero-molecule-movie-data.js`, `molecule-gallery-data.js`, `property-distributions-data.js`, `generate-molecule-gallery.py`, `generate-property-distributions.py`, `verde+PCs-02142026.csv`, and `README.md` to the **root** of the repository.

Using git on the command line:

```bash
git init
git add index.html styles.css script.js hero-molecule-movie-data.js molecule-gallery-data.js property-distributions-data.js generate-molecule-gallery.py generate-property-distributions.py verde+PCs-02142026.csv README.md
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

### Add a new database preview row

1. Open `index.html`
2. Find the `<!-- DATABASE PREVIEW SECTION -->` comment
3. Copy any existing `<tr class="db-row" ...>` row
4. Paste it inside `<tbody id="moleculeTableBody">`
5. Update:
   - `data-name="..."` — lowercase InChIKey, SMILES, and dataset keywords used by the search bar
   - `data-family="..."` — must match a dataset `<option value="...">` in the dropdown
   - The table cells for InChIKey, SMILES, property values, atom count, and details link

The search and dataset filter will include the new row automatically.

### Add a new dataset filter option

In `index.html`, find the `<select id="familyFilter">` element and add:

```html
<option value="Your Dataset Name">Display Label</option>
```

Then add `data-family="Your Dataset Name"` to the corresponding table rows.

### Refresh 3D gallery after changing the CSV

The `3D Gallery` tab uses `molecule-gallery-data.js`, which is generated from all available `xyz_S0` geometries in `verde+PCs-02142026.csv`. After replacing or editing the CSV, regenerate the gallery data:

```bash
python generate-molecule-gallery.py
```

Commit both the updated CSV and the regenerated `molecule-gallery-data.js`.

### Refresh property distributions after changing the CSV

The `Properties` tab uses `property-distributions-data.js`, which is generated from `verde+PCs-02142026.csv`. After replacing or editing the CSV, regenerate the distribution data:

```bash
python generate-property-distributions.py
```

Commit both the updated CSV and the regenerated `property-distributions-data.js`.

### Update the BibTeX citation

Open `script.js`, find **Section 6**, and update the `BIBTEX` constant:

```js
const BIBTEX = `@article{abreha2019virtual,
  title   = {Virtual Excited State Reference for the Discovery of Electronic Materials Database: An Open-Access Resource for Ground and Excited State Properties of Organic Molecules},
  author  = {Abreha, Bayileyegn G. and Agarwal, Shashank and Foster, Ian and Blaiszik, Ben and Lopez, Steven A.},
  journal = {The Journal of Physical Chemistry Letters},
  year    = {2019},
  volume  = {10},
  number  = {21},
  pages   = {6835--6841},
  doi     = {10.1021/acs.jpclett.9b02577}
}`;
```

### Update footer links

Open `index.html`, find the `<!-- CONTACT & FOOTER SECTION -->` comment, and replace the contact placeholders with your actual lab page and email.

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
