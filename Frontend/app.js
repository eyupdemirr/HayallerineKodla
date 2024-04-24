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


const ekran1 = document.querySelector("#ekran1");
const ekran2 = document.querySelector("#ekran2");
const icerik1 = document.querySelector("#ekran1-icerik");
const icerik2 = document.querySelector("#ekran2-icerik");

ekran1.addEventListener("click", () => {
  ekran1.classList.add("active");
  ekran2.classList.remove("active");
  icerik1.classList.add("active");
  icerik2.classList.remove("active");
});

ekran2.addEventListener("click", () => {
  ekran1.classList.remove("active");
  ekran2.classList.add("active");
  icerik1.classList.remove("active");
  icerik2.classList.add("active");
});
