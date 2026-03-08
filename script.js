/* ============================================
   MORENO LA QUATRA - ACADEMIC WEBSITE
   JavaScript: Interactions & Easter Eggs
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initEasterEggs();
  initTooltips();
  initTypewriter();
  initScrollEffects();
  initSearch();
});

/* ============================================
   NAVIGATION
   ============================================ */

function initNavigation() {
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  const navLinks = document.querySelectorAll('.nav-links a');

  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage || (currentPage === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
}

/* ============================================
   EASTER EGGS
   ============================================ */

function initEasterEggs() {
  // 1. Konami Code
  const konamiCode = [
    'ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown',
    'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight',
    'KeyB', 'KeyA'
  ];
  let konamiIndex = 0;

  document.addEventListener('keydown', (e) => {
    if (e.code === konamiCode[konamiIndex]) {
      konamiIndex++;
      if (konamiIndex === konamiCode.length) {
        activateKonamiMode();
        konamiIndex = 0;
      }
    } else {
      konamiIndex = 0;
    }
  });

  // 2. Secret click counter on logo
  let logoClicks = 0;
  const logo = document.querySelector('.logo');
  if (logo) {
    logo.addEventListener('click', (e) => {
      if (e.detail === 3) {
        showSecretMessage('You found the triple-click secret! Nice one.');
      }
      logoClicks++;
      if (logoClicks === 7) {
        activateMatrixMode();
        logoClicks = 0;
      }
    });
  }

  // 3. LaTeX compilation joke
  const latexElements = document.querySelectorAll('.latex-logo');
  latexElements.forEach(el => {
    el.addEventListener('dblclick', () => {
      showSecretMessage('Recompiling... Please wait 3-5 business days.');
    });
  });

  // 4. Console easter egg
  console.log('%c~ Welcome, fellow academic! ~', 'font-size: 20px; font-weight: bold;');
  console.log('%cYou found the console easter egg.', 'font-size: 14px; color: #8b0000;');
  console.log('%cTry: konami code, triple-click the logo, type "latex" anywhere, or press Space 5 times', 'font-size: 12px; color: #666;');
  console.log('%c\n  May your gradients never vanish.\n', 'font-family: serif; font-size: 14px;');

  // 5. Secret keyboard triggers
  let typedChars = '';
  document.addEventListener('keypress', (e) => {
    typedChars += e.key.toLowerCase();
    if (typedChars.length > 10) typedChars = typedChars.slice(-10);

    if (typedChars.includes('latex')) {
      showSecretMessage('\\documentclass{genius}');
      typedChars = '';
    }
    if (typedChars.includes('bibtex')) {
      showSecretMessage('Missing citation: [Citation needed]');
      typedChars = '';
    }
    if (typedChars.includes('phd')) {
      showSecretMessage('Permanent Head Damage activated!');
      typedChars = '';
    }
  });

  // 6. Hover on dates shows academic jokes
  const dates = document.querySelectorAll('.cv-date, .blog-date, .talk-date, .project-date, .news-date');
  dates.forEach(date => {
    date.classList.add('easter-egg');
    date.title = getRandomAcademicJoke();
  });

  // 7. Press Space 5× to reveal the full easter egg list
  let spaceCount = 0;
  let spaceTimer;
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      spaceCount++;
      clearTimeout(spaceTimer);
      spaceTimer = setTimeout(() => { spaceCount = 0; }, 2000);
      if (spaceCount >= 5) {
        spaceCount = 0;
        showEasterEggList();
      }
    }
  });
}

function activateKonamiMode() {
  document.body.classList.add('konami-activated');
  showSecretMessage('KONAMI MODE ACTIVATED! +30 citations');

  for (let i = 0; i < 10; i++) {
    setTimeout(() => createFloatingEquation(), i * 200);
  }

  setTimeout(() => {
    document.body.classList.remove('konami-activated');
  }, 5000);
}

function activateMatrixMode() {
  let canvas = document.getElementById('matrix-canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'matrix-canvas';
    document.body.appendChild(canvas);
    initMatrixRain(canvas);
  }
  canvas.classList.add('active');
  showSecretMessage('The Matrix has you... Follow the white rabbit.');

  setTimeout(() => {
    canvas.classList.remove('active');
  }, 8000);
}

function initMatrixRain(canvas) {
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const chars = '\u2202\u222B\u2211\u220F\u221A\u221E\u2248\u2260\u2264\u2265\u03B1\u03B2\u03B3\u03B4\u03B5\u03B6\u03B7\u03B8\u03BB\u03BC\u03BE\u03C0\u03C1\u03C3\u03C4\u03C6\u03C8\u03C9\u0394\u03A9\u2207\u2208\u2209\u2282\u2283\u222A\u222901';
  const fontSize = 14;
  const columns = canvas.width / fontSize;
  const drops = Array(Math.floor(columns)).fill(1);

  function draw() {
    ctx.fillStyle = 'rgba(254, 253, 251, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#8b0000';
    ctx.font = `${fontSize}px monospace`;

    for (let i = 0; i < drops.length; i++) {
      const char = chars[Math.floor(Math.random() * chars.length)];
      ctx.fillText(char, i * fontSize, drops[i] * fontSize);

      if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
        drops[i] = 0;
      }
      drops[i]++;
    }
  }

  setInterval(draw, 50);
}

function createFloatingEquation() {
  const equations = [
    'E = mc\u00B2',
    '\u2207 \u00D7 B = \u03BC\u2080J',
    'F = ma',
    'e^(i\u03C0) + 1 = 0',
    '\u222Bf(x)dx',
    '\u03A3(1/n\u00B2) = \u03C0\u00B2/6',
    'P(A|B) = P(B|A)P(A)/P(B)',
    '\u2202L/\u2202\u03B8 = 0',
    'H = -\u03A3p log p',
    'softmax(z)\u1D62'
  ];

  const eq = document.createElement('div');
  eq.textContent = equations[Math.floor(Math.random() * equations.length)];
  eq.style.cssText = `
    position: fixed;
    font-family: 'EB Garamond', serif;
    font-size: ${Math.random() * 20 + 16}px;
    color: #8b0000;
    pointer-events: none;
    z-index: 1001;
    left: ${Math.random() * 80 + 10}%;
    top: ${Math.random() * 80 + 10}%;
    opacity: 0;
    transition: opacity 0.5s, transform 2s;
  `;
  document.body.appendChild(eq);

  requestAnimationFrame(() => {
    eq.style.opacity = '1';
    eq.style.transform = `translateY(-50px) rotate(${Math.random() * 20 - 10}deg)`;
  });

  setTimeout(() => {
    eq.style.opacity = '0';
    setTimeout(() => eq.remove(), 500);
  }, 2000);
}

function showSecretMessage(message) {
  let tooltip = document.querySelector('.secret-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'secret-tooltip';
    document.body.appendChild(tooltip);
  }

  tooltip.textContent = message;
  tooltip.style.left = '50%';
  tooltip.style.top = '20%';
  tooltip.style.transform = 'translateX(-50%)';
  tooltip.classList.add('visible');

  setTimeout(() => {
    tooltip.classList.remove('visible');
  }, 3000);
}

function getRandomAcademicJoke() {
  const jokes = [
    'Time flies when you\'re debugging',
    'In academia, this counts as "recent"',
    'Before the great coffee shortage',
    'A simpler time (with simpler bugs)',
    'When GPUs were affordable',
    'Pre-transformer era archaeology',
    'When we still believed in feature engineering',
    'The dark ages of batch size 32',
    'When "deep" meant 3 layers',
    'Before attention was all we needed',
    'Back when Python 2 was acceptable',
    'When overfitting was still called "memorization"'
  ];
  return jokes[Math.floor(Math.random() * jokes.length)];
}

function showEasterEggList() {
  const existing = document.getElementById('easter-egg-list');
  if (existing) { existing.remove(); return; }

  const eggs = [
    { keys: '↑↑↓↓←→←→BA',  desc: 'Konami code' },
    { keys: 'triple-click',  desc: 'Click the logo three times' },
    { keys: '×7 clicks',     desc: 'Click the logo seven times' },
    { keys: 'dblclick',      desc: 'Double-click the LaTeX logo in the footer' },
    { keys: 'latex',         desc: 'Type "latex" anywhere on the page' },
    { keys: 'bibtex',        desc: 'Type "bibtex" anywhere on the page' },
    { keys: 'phd',           desc: 'Type "phd" anywhere on the page' },
    { keys: 'hover dates',   desc: 'Hover over any date for academic wisdom' },
    { keys: '> console',     desc: 'Open the browser console' },
    { keys: 'space ×5',      desc: 'You just found this one' },
  ];

  const modal = document.createElement('div');
  modal.id = 'easter-egg-list';
  modal.style.cssText = `
    position: fixed; inset: 0; z-index: 9999;
    background: rgba(0, 0, 0, 0.55);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-sans, sans-serif);
  `;

  const card = document.createElement('div');
  card.style.cssText = `
    background: var(--paper, #fefdfc);
    border: 1px solid var(--border, #ddd);
    padding: 2rem 2.4rem;
    max-width: 420px; width: 90%;
    line-height: 1.6;
  `;

  const title = document.createElement('p');
  title.style.cssText = `
    margin: 0 0 1rem;
    font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent, #8b0000); font-weight: 600;
  `;
  title.textContent = 'Hidden Easter Eggs';

  const list = document.createElement('ul');
  list.style.cssText = `
    list-style: none; margin: 0 0 1.2rem; padding: 0;
  `;

  eggs.forEach(egg => {
    const li = document.createElement('li');
    li.style.cssText = `
      display: flex; gap: 0.8rem; align-items: baseline;
      padding: 0.25rem 0; border-bottom: 1px solid var(--border, #eee);
      font-size: 0.88rem;
    `;
    li.innerHTML = `
      <code style="font-family: var(--font-mono, monospace); font-size: 0.8rem;
                   color: var(--accent, #8b0000); white-space: nowrap;">${egg.keys}</code>
      <span style="color: var(--text, #333);">${egg.desc}</span>
    `;
    list.appendChild(li);
  });

  const dismiss = document.createElement('p');
  dismiss.style.cssText = `
    margin: 0; font-size: 0.78rem; color: var(--text-light, #888); text-align: right;
  `;
  dismiss.textContent = 'Esc or click outside to close';

  card.appendChild(title);
  card.appendChild(list);
  card.appendChild(dismiss);
  modal.appendChild(card);
  document.body.appendChild(modal);

  modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });

  const escHandler = (e) => {
    if (e.key === 'Escape') { modal.remove(); document.removeEventListener('keydown', escHandler); }
  };
  document.addEventListener('keydown', escHandler);
}

/* ============================================
   SIDENOTES
   ============================================ */

function initSidenotes() {
  const labels = document.querySelectorAll('label.sidenote-number');
  labels.forEach(label => {
    const sidenote = label.nextElementSibling;
    if (!sidenote || !sidenote.classList.contains('sidenote')) return;

    label.addEventListener('mouseenter', () => {
      showSidenoteTooltip(label, sidenote.innerHTML);
    });
    label.addEventListener('mouseleave', hideSidenoteTooltip);
  });
}

function showSidenoteTooltip(element, html) {
  let tooltip = document.querySelector('.sidenote-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'sidenote-tooltip';
    document.body.appendChild(tooltip);
  }
  const rect = element.getBoundingClientRect();
  tooltip.innerHTML = html;
  const left = Math.min(rect.left + window.scrollX, window.innerWidth - 320);
  tooltip.style.left = `${Math.max(left, 8)}px`;
  tooltip.style.top = `${rect.bottom + window.scrollY + 8}px`;
  tooltip.classList.add('visible');
}

function hideSidenoteTooltip() {
  const tooltip = document.querySelector('.sidenote-tooltip');
  if (tooltip) tooltip.classList.remove('visible');
}

/* ============================================
   TOOLTIPS & FOOTNOTES
   ============================================ */

function initTooltips() {
  const footnotes = document.querySelectorAll('sup[data-note]');
  footnotes.forEach(fn => {
    fn.addEventListener('mouseenter', (e) => {
      showTooltip(e.target, fn.dataset.note);
    });
    fn.addEventListener('mouseleave', hideTooltip);
  });
}

function showTooltip(element, text) {
  let tooltip = document.querySelector('.footnote-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'secret-tooltip footnote-tooltip';
    document.body.appendChild(tooltip);
  }

  const rect = element.getBoundingClientRect();
  tooltip.textContent = text;
  tooltip.style.left = `${rect.left}px`;
  tooltip.style.top = `${rect.bottom + 10}px`;
  tooltip.style.transform = 'none';
  tooltip.classList.add('visible');
}

function hideTooltip() {
  const tooltip = document.querySelector('.footnote-tooltip');
  if (tooltip) tooltip.classList.remove('visible');
}

/* ============================================
   TYPEWRITER EFFECT
   ============================================ */

function initTypewriter() {
  const typewriterElements = document.querySelectorAll('[data-typewriter]');
  typewriterElements.forEach(el => {
    const text = el.textContent;
    el.textContent = '';
    el.style.borderRight = '2px solid var(--accent)';

    let i = 0;
    const interval = setInterval(() => {
      el.textContent += text[i];
      i++;
      if (i >= text.length) {
        clearInterval(interval);
        setTimeout(() => {
          el.style.borderRight = 'none';
        }, 1000);
      }
    }, 50);
  });
}

/* ============================================
   SCROLL EFFECTS
   ============================================ */

function initScrollEffects() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);

  const selectors = '.cv-entry, .course-card, .blog-entry, .publication, .talk-entry, .project-card, .quick-link';
  document.querySelectorAll(selectors).forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });
}

// Add visible class styles dynamically
const style = document.createElement('style');
style.textContent = `
  .cv-entry.visible, .course-card.visible, .blog-entry.visible, .publication.visible,
  .talk-entry.visible, .project-card.visible, .quick-link.visible {
    opacity: 1 !important;
    transform: translateY(0) !important;
  }
`;
document.head.appendChild(style);

/* ============================================
   SEARCH (Blog, Projects, Talks)
   ============================================ */

function initSearch() {
  const searchBar = document.querySelector('.search-bar');
  if (!searchBar) return;

  searchBar.addEventListener('input', () => {
    const query = searchBar.value.toUpperCase();
    const items = document.querySelectorAll('.search-item');
    items.forEach(item => {
      const title = item.querySelector('.search-title');
      if (!title) return;
      const text = title.textContent.toUpperCase();
      item.style.display = text.includes(query) ? '' : 'none';
    });
  });
}

/* ============================================
   UTILITY: Copy BibTeX
   ============================================ */

function copyBibtex(bibtexId) {
  const bibtex = document.getElementById(bibtexId);
  if (bibtex) {
    navigator.clipboard.writeText(bibtex.textContent).then(() => {
      showSecretMessage('BibTeX copied! Your h-index thanks you.');
    });
  }
}

window.copyBibtex = copyBibtex;

/* ============================================
   COMPONENT LOADER (replaces jQuery)
   ============================================ */

function loadComponents() {
  const includes = document.querySelectorAll('[data-include]');
  const promises = Array.from(includes).map(el => {
    const name = el.dataset.include;
    // Nested components are used from sub-folders:
    // nested-2 = two levels deep (e.g. teaching/SLP/)
    // nested-1 = one level deep (e.g. teaching/)
    const prefix = name.includes('nested-2') ? '../../components/' :
                   name.includes('nested') ? '../components/' : 'components/';
    const file = prefix + name + '.html';
    return fetch(file)
      .then(r => {
        if (!r.ok) throw new Error(r.status);
        return r.text();
      })
      .then(html => {
        el.innerHTML = html;
      })
      .catch(() => {});
  });

  Promise.all(promises).then(() => {
    // Re-init navigation after components are loaded
    initNavigation();
    initMobileMenu();
    initSidenotes();
  });
}

/* ============================================
   MOBILE MENU
   ============================================ */

function initMobileMenu() {
  const toggle = document.querySelector('.menu-toggle');
  const navLinks = document.querySelector('.nav-links');
  if (!toggle || !navLinks) return;

  toggle.addEventListener('click', () => {
    const isOpen = toggle.classList.toggle('open');
    navLinks.classList.toggle('open');
    toggle.setAttribute('aria-expanded', isOpen);
  });

  // Close menu when a link is clicked
  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      toggle.classList.remove('open');
      navLinks.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
}

// Load components on DOMContentLoaded
document.addEventListener('DOMContentLoaded', loadComponents);
