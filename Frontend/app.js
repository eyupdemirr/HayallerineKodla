const menuButton = document.getElementById("menu-button");
const menu = document.getElementById("menu");

menuButton.addEventListener("click", function(event) {
  event.preventDefault();
  menu.classList.toggle("open");
});

document.addEventListener("click", function(event) {
  if (!event.target.closest("#menu") && !event.target.closest("#menu-button")) {
    menu.classList.remove("open");
  }
});

const menuItems = document.querySelectorAll("#menu ul li a");

    // Scroll-snap ile otomatik aktifleştirme
    window.addEventListener("load", function() {
      const sections = document.querySelectorAll(".kutu");
      sections.forEach((section, index) => {
        section.style.scrollSnapAlign = "start";
        if (index === 0) {
          section.classList.add("active");
          menuItems[index].classList.add("active");
        }
      });
    });

    window.addEventListener("scroll", function() {
      const sections = document.querySelectorAll(".kutu");
      const activeSection = sections.find(section =>
        section.getBoundingClientRect().top >= 0 &&
        section.getBoundingClientRect().bottom <= window.innerHeight
      );

      if (activeSection) {
        sections.forEach(section => section.classList.remove("active"));
        activeSection.classList.add("active");

        const activeIndex = Array.prototype.indexOf.call(sections, activeSection);
        menuItems.forEach(menuItem => menuItem.classList.remove("active"));
        menuItems[activeIndex].classList.add("active");
      }
    });
  
/*Yukarı Git Butonu*/
const yukariGitButonu = document.getElementById("yukari-git");

window.addEventListener("scroll", function() {
  if (window.pageYOffset > 100) {
    yukariGitButonu.classList.remove("gizli");
  } else {
    yukariGitButonu.classList.add("gizli");
  }
});

yukariGitButonu.addEventListener("click", function() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
});
