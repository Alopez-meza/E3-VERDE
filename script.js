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
   Filters .molecule-card elements by their data-name and data-family
   attributes in response to text input and dropdown changes.

   To make a new card searchable:
     - Add data-name="..." (lowercase keywords) to the card div
     - Add data-family="..." matching an <option> value in the dropdown
   ===================================================================== */
const searchInput  = document.getElementById('searchInput');
const familyFilter = document.getElementById('familyFilter');
const moleculeGrid = document.getElementById('moleculeGrid');
const noResults    = document.getElementById('noResults');

function filterCards() {
  const query          = searchInput.value.toLowerCase().trim();
  const selectedFamily = familyFilter.value;

  const cards = moleculeGrid.querySelectorAll('.molecule-card');
  let visible = 0;

  cards.forEach(card => {
    const name   = (card.dataset.name   || '').toLowerCase();
    const family = (card.dataset.family || '');

    const matchesSearch = !query || name.includes(query) || family.toLowerCase().includes(query);
    const matchesFamily = !selectedFamily || family === selectedFamily;

    const show = matchesSearch && matchesFamily;
    card.classList.toggle('hidden', !show);
    if (show) visible++;
  });

  // Toggle the "no results" message
  noResults.hidden = visible > 0;
}

if (searchInput) searchInput.addEventListener('input',  filterCards);
if (familyFilter) familyFilter.addEventListener('change', filterCards);


/* =====================================================================
   5. Scroll reveal animations
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
   6. Copy citation to clipboard
   Copies a BibTeX string when the footer "Copy BibTeX" button is clicked.

   To update: edit the `bibtex` string below with your actual citation.
   ===================================================================== */

// To edit: replace this BibTeX block with your published citation
const BIBTEX = `@article{verde2024,
  title   = {VERDE Materials DB: A Curated Database for Photoredox Chemistry},
  author  = {Lopez, Steven A. and others},
  journal = {Journal Name},
  year    = {2024},
  volume  = {},
  pages   = {},
  doi     = {10.XXXX/XXXXXXX}
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
