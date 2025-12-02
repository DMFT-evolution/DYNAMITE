(function () {
  const root = document.documentElement;
  const body = document.body;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function setupFadeObserver() {
    if (!root || !body) {
      return;
    }

    const hero = document.querySelector('.dmfe-hero');
    if (!hero) {
      body.classList.remove('dmfe-home');
      root.style.setProperty('--dmfe-logo-fade', '1');
      return;
    }

    body.classList.add('dmfe-home');

    const heroText = hero.querySelector('.dmfe-hero__text');
    let fadeThreshold = 1;
    let startTop = 0;

    const computeThreshold = () => {
      const target = heroText || hero;
      if (!target) {
        fadeThreshold = 1;
        startTop = 0;
        return;
      }
      const rect = target.getBoundingClientRect();
      startTop = rect.top;
      fadeThreshold = Math.max(rect.bottom, 1);
    };

    computeThreshold();

    let ticking = false;

    const updateFade = () => {
      const target = heroText || hero;
      if (!target) {
        root.style.setProperty('--dmfe-logo-fade', '1');
        return;
      }
      const currentTop = target.getBoundingClientRect().top;
      const delta = startTop - currentTop;
      const fade = clamp(2 * delta / fadeThreshold - 1, 0, 1);
      root.style.setProperty('--dmfe-logo-fade', fade.toFixed(3));
    };

    const requestTick = () => {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(() => {
          updateFade();
          ticking = false;
        });
      }
    };

    updateFade();
    window.addEventListener('scroll', requestTick, { passive: true });
    window.addEventListener('resize', () => {
      computeThreshold();
      requestTick();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupFadeObserver);
  } else {
    setupFadeObserver();
  }
})();
