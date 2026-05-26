/* =====================================================================
   VERDE Materials DB — script.js
   =====================================================================
   Sections:
    1. Navbar: scroll effect
    2. Navbar: mobile toggle
    3. Active nav link tracking
    4. Database: search & filter
    5. Scroll reveal animations
    6. Copy citation to clipboard
   ===================================================================== */


/* =====================================================================
   1. Navbar — scroll effect
   Adds the "scrolled" class when user scrolls past 40px, which
   triggers the frosted-glass background in CSS.
   ===================================================================== */
const navbar    = document.getElementById('navbar');
const NAV_THRESHOLD = 40;

function onScroll() {
  if (window.scrollY > NAV_THRESHOLD) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
}

window.addEventListener('scroll', onScroll, { passive: true });
onScroll(); // Run once on load in case page is already scrolled


/* =====================================================================
   2. Navbar — mobile toggle
   Opens/closes the nav-links list and animates the hamburger to an X.
   ===================================================================== */
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

navToggle.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('open');
  navToggle.classList.toggle('open', isOpen);
  navToggle.setAttribute('aria-expanded', String(isOpen));
});

// Close the mobile menu when any nav link is clicked
navLinks.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('open');
    navToggle.classList.remove('open');
    navToggle.setAttribute('aria-expanded', 'false');
  });
});

// Close mobile menu when clicking outside of it
document.addEventListener('click', (e) => {
  if (
    navLinks.classList.contains('open') &&
    !navLinks.contains(e.target) &&
    !navToggle.contains(e.target)
  ) {
    navLinks.classList.remove('open');
    navToggle.classList.remove('open');
    navToggle.setAttribute('aria-expanded', 'false');
  }
});


/* =====================================================================
   3. Active nav link tracking
   Uses IntersectionObserver to highlight the correct nav link as the
   user scrolls through sections.
   ===================================================================== */
const sections    = document.querySelectorAll('section[id], footer[id]');
const navLinkEls  = document.querySelectorAll('.nav-link');
const navHeight   = parseInt(
  getComputedStyle(document.documentElement).getPropertyValue('--nav-h'),
  10
) || 68;

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinkEls.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
      }
    });
  },
  {
    rootMargin: `-${navHeight}px 0px -55% 0px`,
    threshold: 0,
  }
);

sections.forEach(section => sectionObserver.observe(section));


/* =====================================================================
   4. Database — search & filter
   Filters table rows by their searchable text and dataset
   attributes in response to text input and dropdown changes.
   ===================================================================== */
const searchInput  = document.getElementById('searchInput');
const familyFilter = document.getElementById('familyFilter');
const moleculeGrid = document.getElementById('moleculeGrid');
const noResults    = document.getElementById('noResults');

function filterDatabaseRows() {
  if (!moleculeGrid || !searchInput) return;

  const query          = searchInput.value.toLowerCase().trim();
  const selectedFamily = familyFilter ? familyFilter.value : '';

  const rows = moleculeGrid.querySelectorAll('[data-db-row]');
  let visible = 0;

  rows.forEach(row => {
    const name   = (row.dataset.name   || '').toLowerCase();
    const family = (row.dataset.family || '');

    const matchesSearch = !query || name.includes(query) || family.toLowerCase().includes(query);
    const matchesFamily = !selectedFamily || family === selectedFamily;

    const show = matchesSearch && matchesFamily;
    row.hidden = !show;
    if (show) visible++;
  });

  if (noResults) noResults.hidden = visible > 0;
}

if (searchInput) searchInput.addEventListener('input', filterDatabaseRows);
if (familyFilter) familyFilter.addEventListener('change', filterDatabaseRows);


/* =====================================================================
   5. Hero Molecule Movie — random S0 geometry loop
   Renders a lightweight, button-free 3D preview inside the hero stats card.
   ===================================================================== */
const heroMovieData = window.HERO_MOLECULE_MOVIE || [];
const heroMovieEl = document.getElementById('heroMoleculeMovie');
const heroMovieLabel = document.getElementById('heroMovieLabel');

let heroMovieViewer = null;
let heroMovieIndex = 0;
let heroMovieSpinFrame = null;

function initHeroMoleculeMovie() {
  if (!heroMovieEl || heroMovieData.length === 0 || !window.$3Dmol) return;

  heroMovieViewer = window.$3Dmol.createViewer(heroMovieEl, {
    backgroundColor: '0x000000',
    backgroundAlpha: 0,
  });

  window.addEventListener('resize', () => {
    if (!heroMovieViewer) return;
    heroMovieViewer.resize();
    heroMovieViewer.render();
  });

  loadHeroMovieMolecule(0);
  startHeroMovieSpin();

  window.setInterval(() => {
    loadHeroMovieMolecule((heroMovieIndex + 1) % heroMovieData.length);
  }, 1450);
}

function loadHeroMovieMolecule(index) {
  const molecule = heroMovieData[index];
  if (!heroMovieViewer || !molecule) return;

  heroMovieIndex = index;
  heroMovieViewer.clear();
  heroMovieViewer.addModel(molecule.xyzS0, 'xyz');
  heroMovieViewer.setStyle({}, getMolViewStyle({ stickRadius: 0.16, sphereScale: 0.22 }));
  heroMovieViewer.zoomTo();
  heroMovieViewer.zoom(1.18);
  heroMovieViewer.rotate(18, { x: 0, y: 1, z: 0 });
  heroMovieViewer.render();

  if (heroMovieLabel) {
    heroMovieLabel.textContent = `${molecule.atoms} atoms`;
  }
}

function startHeroMovieSpin() {
  if (!heroMovieViewer || heroMovieSpinFrame) return;

  const spin = () => {
    if (heroMovieViewer) {
      heroMovieViewer.rotate(0.45, { x: 0, y: 1, z: 0 });
      heroMovieViewer.render();
    }
    heroMovieSpinFrame = window.requestAnimationFrame(spin);
  };
  heroMovieSpinFrame = window.requestAnimationFrame(spin);
}

initHeroMoleculeMovie();


/* =====================================================================
   6. Property Distributions — selectable histograms
   Uses aggregated bins from property-distributions-data.js so the page
   stays lightweight while representing the full CSV.
   ===================================================================== */
const distributionData = window.PROPERTY_DISTRIBUTIONS || null;
const distributionControls = document.getElementById('distributionControls');
const distributionCharts = document.getElementById('distributionCharts');
const distributionEmpty = document.getElementById('distributionEmpty');
const distributionSelectedCount = document.getElementById('distributionSelectedCount');
const distributionRowCount = document.getElementById('distributionRowCount');
const distributionPropertyCount = document.getElementById('distributionPropertyCount');
const distributionDefaultBtn = document.getElementById('distributionDefaultBtn');
const distributionSelectAllBtn = document.getElementById('distributionSelectAllBtn');
const distributionClearBtn = document.getElementById('distributionClearBtn');

let selectedDistributionKeys = new Set();

function initPropertyDistributions() {
  if (!distributionData || !distributionControls || !distributionCharts) return;

  selectedDistributionKeys = new Set(distributionData.defaultSelected || []);

  if (distributionRowCount) {
    distributionRowCount.textContent = formatCount(distributionData.totalRows);
  }
  if (distributionPropertyCount) {
    distributionPropertyCount.textContent = formatCount(distributionData.properties.length);
  }

  renderDistributionControls();
  renderDistributionCharts();

  if (distributionDefaultBtn) {
    distributionDefaultBtn.addEventListener('click', () => {
      selectedDistributionKeys = new Set(distributionData.defaultSelected || []);
      syncDistributionCheckboxes();
      renderDistributionCharts();
    });
  }

  if (distributionSelectAllBtn) {
    distributionSelectAllBtn.addEventListener('click', () => {
      selectedDistributionKeys = new Set(distributionData.properties.map(prop => prop.key));
      syncDistributionCheckboxes();
      renderDistributionCharts();
    });
  }

  if (distributionClearBtn) {
    distributionClearBtn.addEventListener('click', () => {
      selectedDistributionKeys.clear();
      syncDistributionCheckboxes();
      renderDistributionCharts();
    });
  }
}

function renderDistributionControls() {
  distributionControls.innerHTML = '';
  const groups = new Map();

  distributionData.properties.forEach(prop => {
    if (!groups.has(prop.group)) groups.set(prop.group, []);
    groups.get(prop.group).push(prop);
  });

  groups.forEach((properties, groupName) => {
    const group = document.createElement('div');
    group.className = 'distribution-group';

    const title = document.createElement('h4');
    title.className = 'distribution-group-title';
    title.textContent = groupName;
    group.appendChild(title);

    properties.forEach(prop => {
      const label = document.createElement('label');
      label.className = 'distribution-checkbox';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = prop.key;
      checkbox.checked = selectedDistributionKeys.has(prop.key);
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
          selectedDistributionKeys.add(prop.key);
        } else {
          selectedDistributionKeys.delete(prop.key);
        }
        renderDistributionCharts();
      });

      const text = document.createElement('span');
      text.textContent = prop.label;

      label.append(checkbox, text);
      group.appendChild(label);
    });

    distributionControls.appendChild(group);
  });
}

function syncDistributionCheckboxes() {
  distributionControls.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
    checkbox.checked = selectedDistributionKeys.has(checkbox.value);
  });
}

function renderDistributionCharts() {
  const selectedProperties = distributionData.properties.filter(prop => selectedDistributionKeys.has(prop.key));
  distributionCharts.innerHTML = '';

  if (distributionSelectedCount) {
    distributionSelectedCount.textContent = `${selectedProperties.length} selected`;
  }

  if (distributionEmpty) {
    distributionEmpty.hidden = selectedProperties.length > 0;
  }

  selectedProperties.forEach(prop => {
    distributionCharts.appendChild(createDistributionCard(prop));
  });
}

function createDistributionCard(prop) {
  const card = document.createElement('article');
  card.className = 'distribution-chart-card';

  const header = document.createElement('div');
  header.className = 'distribution-chart-header';

  const titleWrap = document.createElement('div');
  const title = document.createElement('h3');
  title.className = 'distribution-chart-title';
  title.textContent = prop.label;
  const group = document.createElement('span');
  group.className = 'distribution-chart-group';
  group.textContent = prop.group;
  titleWrap.append(title, group);

  header.appendChild(titleWrap);
  card.appendChild(header);

  const histogram = document.createElement('div');
  histogram.className = 'distribution-histogram';
  const maxCount = Math.max(...prop.bins.map(bin => bin.count), 1);

  prop.bins.forEach(bin => {
    const bar = document.createElement('div');
    bar.className = 'distribution-bar';
    bar.style.height = `${Math.max(2, (bin.count / maxCount) * 100)}%`;
    bar.title = `${formatValue(bin.start, prop.unit)} to ${formatValue(bin.end, prop.unit)}: ${formatCount(bin.count)} molecules`;
    histogram.appendChild(bar);
  });

  card.appendChild(histogram);

  const axis = document.createElement('div');
  axis.className = 'distribution-axis';
  axis.append(
    createTextSpan(formatValue(prop.min, prop.unit)),
    createTextSpan(formatValue(prop.max, prop.unit))
  );
  card.appendChild(axis);

  const stats = document.createElement('div');
  stats.className = 'distribution-stats';
  stats.append(
    createDistributionStat('Mean', formatValue(prop.mean, prop.unit)),
    createDistributionStat('Median', formatValue(prop.median, prop.unit)),
    createDistributionStat('P25', formatValue(prop.p25, prop.unit)),
    createDistributionStat('P75', formatValue(prop.p75, prop.unit))
  );
  card.appendChild(stats);

  return card;
}

function createDistributionStat(label, value) {
  const stat = document.createElement('div');
  stat.className = 'distribution-stat';

  const statLabel = document.createElement('span');
  statLabel.className = 'distribution-stat-label';
  statLabel.textContent = label;

  const statValue = document.createElement('span');
  statValue.className = 'distribution-stat-value';
  statValue.textContent = value;

  stat.append(statLabel, statValue);
  return stat;
}

function createTextSpan(text) {
  const span = document.createElement('span');
  span.textContent = text;
  return span;
}

function formatValue(value, unit = '') {
  if (value === null || value === undefined || value === '') return 'n/a';
  const rounded = Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 3,
  });
  return unit ? `${rounded} ${unit}` : rounded;
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

initPropertyDistributions();


/* =====================================================================
   7. 3D Gallery — S0 XYZ viewer
   Uses molecule-gallery-data.js and 3Dmol.js to render representative
   ground-state geometries extracted from the CSV.
   ===================================================================== */
const galleryMolecules = window.S0_GALLERY_MOLECULES || [];
const galleryViewerEl  = document.getElementById('moleculeViewer');
const galleryFallback  = document.getElementById('galleryFallback');
const gallerySlider    = document.getElementById('gallerySlider');
const galleryCounter   = document.getElementById('galleryCounter');
const galleryPrevBtn   = document.getElementById('galleryPrevBtn');
const galleryNextBtn   = document.getElementById('galleryNextBtn');
const galleryTitle     = document.getElementById('galleryTitle');
const gallerySmiles    = document.getElementById('gallerySmiles');
const galleryInchi     = document.getElementById('galleryInchi');
const galleryAtoms     = document.getElementById('galleryAtoms');
const galleryGap       = document.getElementById('galleryGap');
const galleryAbs       = document.getElementById('galleryAbsorption');

let galleryViewer = null;
let activeGalleryIndex = 0;

function initGallery() {
  if (!galleryViewerEl) return;

  if (galleryMolecules.length === 0) {
    showGalleryFallback('No S0 geometries are available for the gallery.');
    return;
  }

  if (gallerySlider) {
    gallerySlider.max = String(galleryMolecules.length);
    gallerySlider.addEventListener('input', () => {
      loadGalleryMolecule(Number(gallerySlider.value) - 1);
    });
  }

  if (galleryPrevBtn) {
    galleryPrevBtn.addEventListener('click', () => stepGalleryMolecule(-1));
  }

  if (galleryNextBtn) {
    galleryNextBtn.addEventListener('click', () => stepGalleryMolecule(1));
  }

  if (!window.$3Dmol) {
    showGalleryFallback('The 3D viewer library did not load. Check your internet connection and refresh the page.');
    updateGalleryDetails(galleryMolecules[0]);
    return;
  }

  galleryViewer = window.$3Dmol.createViewer(galleryViewerEl, {
    backgroundColor: '0x000000',
    backgroundAlpha: 0,
  });
  window.addEventListener('resize', () => {
    if (!galleryViewer) return;
    galleryViewer.resize();
    galleryViewer.render();
  });

  loadGalleryMolecule(0);
}

function loadGalleryMolecule(index) {
  const molecule = galleryMolecules[index];
  if (!molecule) return;

  activeGalleryIndex = Math.max(0, Math.min(index, galleryMolecules.length - 1));
  if (gallerySlider) gallerySlider.value = String(activeGalleryIndex + 1);
  updateGalleryDetails(molecule);
  updateGalleryCounter();

  if (!galleryViewer) return;

  try {
    galleryViewer.clear();
    galleryViewer.addModel(molecule.xyzS0, 'xyz');
    galleryViewer.setStyle({}, getMolViewStyle({ stickRadius: 0.13, sphereScale: 0.18 }));
    galleryViewer.zoomTo();
    galleryViewer.zoom(0.86);
    galleryViewer.render();
  } catch (err) {
    console.warn('Unable to render S0 geometry:', err);
    showGalleryFallback('This S0 geometry could not be rendered by the 3D viewer.');
  }
}

function stepGalleryMolecule(direction) {
  if (galleryMolecules.length === 0) return;
  const nextIndex = (activeGalleryIndex + direction + galleryMolecules.length) % galleryMolecules.length;
  loadGalleryMolecule(nextIndex);
}

function getMolViewStyle({ stickRadius, sphereScale }) {
  return {
    stick: {
      radius: stickRadius,
      colorscheme: 'Jmol',
    },
    sphere: {
      scale: sphereScale,
      colorscheme: 'Jmol',
    },
  };
}

function updateGalleryDetails(molecule) {
  galleryTitle.textContent = molecule.label || molecule.inchiKey || 'S0 Geometry';
  gallerySmiles.textContent = molecule.smiles || '';
  galleryInchi.textContent = molecule.label || '-';
  galleryAtoms.textContent = molecule.atoms || '-';
  galleryGap.textContent = molecule.gapEv || 'n/a';
  galleryAbs.textContent = molecule.absorptionNm || 'n/a';
}

function updateGalleryCounter() {
  if (!galleryCounter) return;
  if (galleryMolecules.length === 0) {
    galleryCounter.textContent = '0 / 0';
    return;
  }
  galleryCounter.textContent = `${(activeGalleryIndex + 1).toLocaleString()} / ${galleryMolecules.length.toLocaleString()}`;
}

function showGalleryFallback(message) {
  if (!galleryFallback) return;
  galleryFallback.textContent = message;
  galleryFallback.hidden = false;
}

initGallery();


/* =====================================================================
   8. Scroll reveal animations
   Observes elements with class "reveal" and adds "visible" once they
   enter the viewport, triggering the CSS transition.

   To animate any element on scroll:
     1. Add class="reveal" to the HTML element
     2. Optionally add reveal-delay-1 through reveal-delay-4 for stagger
   ===================================================================== */
const revealEls = document.querySelectorAll('.reveal');

if (revealEls.length > 0 && 'IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          revealObserver.unobserve(entry.target); // animate once only
        }
      });
    },
    { threshold: 0.08 }
  );

  revealEls.forEach(el => revealObserver.observe(el));
} else {
  // Fallback: show all immediately if IntersectionObserver not supported
  revealEls.forEach(el => el.classList.add('visible'));
}


/* =====================================================================
   9. Copy citation to clipboard
   Copies a BibTeX string when the footer "Copy BibTeX" button is clicked.

   To update: edit the `bibtex` string below with your actual citation.
   ===================================================================== */

// To edit: replace this BibTeX block if the preferred citation changes.
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

const copyBtn = document.getElementById('copyCitationBtn');

if (copyBtn) {
  copyBtn.addEventListener('click', () => {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(BIBTEX)
        .then(showCopySuccess)
        .catch(() => fallbackCopy(BIBTEX));
    } else {
      fallbackCopy(BIBTEX);
    }
  });
}

function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none;';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
    showCopySuccess();
  } catch (err) {
    console.warn('Copy to clipboard failed:', err);
  }
  document.body.removeChild(ta);
}

function showCopySuccess() {
  if (!copyBtn) return;
  const original = copyBtn.textContent;
  copyBtn.textContent = 'Copied!';
  copyBtn.style.color = 'var(--c-accent-light)';
  setTimeout(() => {
    copyBtn.textContent = original;
    copyBtn.style.color = '';
  }, 2200);
}
