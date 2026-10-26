(() => {
  const header = document.querySelector(".site-header");
  const button = header?.querySelector(".nav-toggle");
  const nav = header?.querySelector("nav");
  const links = nav ? [...nav.querySelectorAll('a[href^="#"]')] : [];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (header) {
    let scrollQueued = false;
    let isScrolled = window.scrollY > 24;
    const updateHeaderState = () => {
      const nextState = isScrolled ? window.scrollY > 8 : window.scrollY > 24;
      if (nextState !== isScrolled) {
        isScrolled = nextState;
        header.classList.toggle("is-scrolled", isScrolled);
      }
      scrollQueued = false;
    };
    window.addEventListener("scroll", () => {
      if (scrollQueued) return;
      scrollQueued = true;
      window.requestAnimationFrame(updateHeaderState);
    }, { passive: true });
    header.classList.toggle("is-scrolled", isScrolled);
  }

  const closeMenu = () => {
    if (!button || !nav) return;
    nav.classList.remove("open");
    button.setAttribute("aria-expanded", "false");
    button.textContent = "Menu";
  };

  if (button && nav) {
    button.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      button.setAttribute("aria-expanded", String(open));
      button.textContent = open ? "Close" : "Menu";
    });
    nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
  }

  const linkedSections = links
    .map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))))
    .filter(Boolean);

  const setActiveLink = (id) => {
    links.forEach((link) => {
      const active = link.hash === `#${id}`;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  const titleStage = document.querySelector(".hero-title-stage");
  if (titleStage && !reduceMotion) {
    const targets = [...titleStage.querySelectorAll("[data-type-text]")];
    const fullText = targets.map((target) => target.textContent);
    const characters = fullText.map((part) => Array.from(part));
    titleStage.classList.add("is-typing");
    targets.forEach((target) => { target.textContent = ""; });

    let segmentIndex = 0;
    let characterIndex = 0;
    const typeNextCharacter = () => {
      if (segmentIndex >= targets.length) {
        titleStage.classList.add("typing-complete");
        window.setTimeout(() => {
          targets.forEach((target) => target.removeAttribute("data-typing-active"));
          titleStage.classList.remove("is-typing", "typing-complete");
        }, 420);
        return;
      }

      const target = targets[segmentIndex];
      const segmentCharacters = characters[segmentIndex];
      if (characterIndex === 0) {
        if (segmentIndex > 0) targets[segmentIndex - 1].removeAttribute("data-typing-active");
        target.setAttribute("data-typing-active", "");
      }
      if (characterIndex >= segmentCharacters.length) {
        target.removeAttribute("data-typing-active");
        segmentIndex += 1;
        characterIndex = 0;
        typeNextCharacter();
        return;
      }

      const character = segmentCharacters[characterIndex++];
      target.textContent += character;
      window.setTimeout(typeNextCharacter, /[,.!?]/.test(character) ? 150 : 38);
    };

    window.setTimeout(typeNextCharacter, 160);
  }

  if ("IntersectionObserver" in window && !reduceMotion) {
    const revealTargets = new Set(document.querySelectorAll([
      "main h1",
      "main h2",
      ".hero-stats",
      ".dataset-data",
      "figure.wide-image",
      ".architecture-flow",
      ".model-grid > article",
      ".charts",
      ".metric-row",
      ".matrices > article",
      ".comparison-grid",
      ".demo-grid",
      ".repo-card",
    ].join(",")));

    [".hero-stats", ".model-grid", ".metric-row", ".matrices"].forEach((selector) => {
      document.querySelectorAll(selector).forEach((group) => {
        [...group.children].forEach((child, index) => {
          child.style.setProperty("--reveal-delay", `${Math.min(index, 3) * 70}ms`);
          revealTargets.add(child);
        });
      });
    });

    try {
      const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      }, { threshold: 0.08, rootMargin: "0px 0px -6% 0px" });

      revealTargets.forEach((element) => element.classList.add("reveal"));
      document.documentElement.classList.add("has-reveal-motion");
      revealTargets.forEach((element) => revealObserver.observe(element));
    } catch (error) {
      document.documentElement.classList.remove("has-reveal-motion");
      revealTargets.forEach((element) => element.classList.remove("reveal", "is-visible"));
    }

  }

  if ("IntersectionObserver" in window && linkedSections.length) {
    const updateActiveLink = () => {
      const headerBottom = header?.getBoundingClientRect().bottom || 0;
      const currentSection = linkedSections
        .filter((section) => section.getBoundingClientRect().top <= headerBottom + 32)
        .at(-1);
      setActiveLink(currentSection?.id || "");
    };
    const activeObserver = new IntersectionObserver(updateActiveLink, { threshold: 0.05 });

    linkedSections.forEach((section) => activeObserver.observe(section));
    window.addEventListener("resize", updateActiveLink, { passive: true });
    if (location.hash) {
      const initialId = decodeURIComponent(location.hash.slice(1));
      if (linkedSections.some((section) => section.id === initialId)) setActiveLink(initialId);
    }
  } else if (linkedSections.length && location.hash) {
    const initialId = decodeURIComponent(location.hash.slice(1));
    if (linkedSections.some((section) => section.id === initialId)) setActiveLink(initialId);
  }
})();
