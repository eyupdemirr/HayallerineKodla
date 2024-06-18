// Eğitim Başlığına Anasayfa Linki Ekler ve Farkedilmesini Sağlar
const h2Etiket = document.getElementById('link');

h2Etiket.onclick = function() {
    window.location.href = '/Frontend/index.html';
};

h2Etiket.style.cursor = 'pointer';
h2Etiket.title = 'AnaSayfa';
