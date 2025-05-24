gsap.registerPlugin(ScrollTrigger);

// Animação do carro (mantida como está)
gsap.to("#car1", {
  scrollTrigger: {
    trigger: ".intro",
    start: "top top",
    end: "bottom top",
    scrub: true
  },
  x: window.innerWidth + 700,
  duration: 2,
  ease: "power2.inOut"
});

// Animação das fotos dos colaboradores (versão melhorada)
gsap.utils.toArray(".foto").forEach((foto, index) => {
  gsap.from(foto, {
    y: 50,
    opacity: 0,
    duration: 1,
    ease: "back.out(1.2)",
    scrollTrigger: {
      trigger: ".fotos-section", // Ativa quando a SEÇÃO entra na tela
      start: "top 70%",         // Começa a animar quando 70% da seção está visível
      end: "bottom 30%",
      toggleActions: "play none none none", // Só anima uma vez
      // markers: true // (opcional) para debug
    },
    delay: index * 0.1 // Efeito "stagger" (cada foto aparece um pouco depois da outra)
  });
});

// Animação das fotos (versão revisada)
gsap.utils.toArray(".foto").forEach((foto, i) => {
    // Reset: Garante que as fotos estão visíveis ANTES da animação (fallback)
    gsap.set(foto, { opacity: 1, y: 0 }); 
  
    // Animação
    gsap.from(foto, {
      y: 50,
      opacity: 0,
      duration: 0.8,
      ease: "power2.out",
      scrollTrigger: {
        trigger: ".fotos-section",
        start: "top 80%",
        end: "top 50%",
        toggleActions: "play none none none",
        // markers: true // (descomente para debug)
      },
      delay: i * 0.1 // Efeito de sequência
    });
  });

  // Força a exibição inicial das imagens (caso o GSAP não as esteja encontrando)
document.addEventListener("DOMContentLoaded", () => {
  gsap.set(".foto", { opacity: 1, y: 0 });
});

gsap.utils.toArray(".colaborador").forEach((colab) => {
    gsap.from(colab, {
      y: 50,
      opacity: 0,
      duration: 0.8,
      ease: "power2.out",
      scrollTrigger: {
        trigger: colab, // Animação dispara quando CADA colaborador entra na tela
        start: "top 80%",
        toggleActions: "play none none none"
      }
    });
  });