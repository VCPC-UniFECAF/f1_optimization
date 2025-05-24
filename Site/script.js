gsap.registerPlugin(ScrollTrigger);


const isMobile = window.innerWidth < 768;


gsap.to("#car1", {
    x: "+=1500",
    scrollTrigger: {
      trigger: ".intro",
      start: "top top",
      end: "bottom top",
      scrub: true,
      markers: false
    },
    ease: "none"
  });


gsap.utils.toArray(".colaborador").forEach((colab, index) => {
  gsap.from(colab, {
    y: isMobile ? 30 : 50,
    opacity: 0,
    duration: 0.8,
    ease: "back.out(1.2)",
    scrollTrigger: {
      trigger: colab,
      start: isMobile ? "top 90%" : "top 80%",
      end: "top 40%",
      toggleActions: "play none none none",
      invalidateOnRefresh: true
    },
    delay: index * 0.1
  });
});

window.addEventListener("load", () => {
  gsap.set(".colaborador", { opacity: 1, y: 0 });
});

if (isMobile && !matchMedia("(prefers-reduced-motion: no-preference)").matches) {
    gsap.set(".colaborador", { clearProps: "all" });
  }

const carAnimation = gsap.to("#car1", {
    x: "+=1500",
    scrollTrigger: {
      trigger: ".intro",
      start: "top top",
      end: "bottom top",
      scrub: true
    },
    ease: "none"
  });

  window.addEventListener("resize", () => {
    carAnimation.kill();
    gsap.set("#car1", { x: -400 });
    carAnimation.restart();
  });

document.getElementById('github-button')?.addEventListener('click', function(e) {
    if (window.innerWidth <= 768) {
      gsap.globalTimeline.pause();
      setTimeout(() => {
        window.open(this.href, '_blank');
      }, 50);
      e.preventDefault();
    }
  });

  if (window.innerWidth <= 768) {
    document.querySelectorAll('a[href^="http"]').forEach(link => {
      link.addEventListener('click', function(e) {
        if (this.id === 'github-button') {
          e.preventDefault();
          window.open(this.href, '_blank', 'noopener');
        }
      });
    });
  }