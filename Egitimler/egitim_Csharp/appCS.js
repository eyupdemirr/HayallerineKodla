// Eğitim Başlığına Anasayfa Linki Ekler ve Farkedilmesini Sağlar
const h2Etiket = document.getElementById('link');

h2Etiket.onclick = function() {
    window.location.href = '/Frontend/index.html';
};

h2Etiket.style.cursor = 'pointer';
h2Etiket.title = 'AnaSayfa';


function checkAnswer() {
    var userInput = document.getElementById("userInput").value.trim();
    var expectedAnswer = 'Console.WriteLine("Hello World");';
    var resultElement = document.getElementById("result");
    var inputElement = document.getElementById("userInput");

    if (userInput === expectedAnswer) {
        resultElement.textContent = "Tebrikler! Doğru cevap.";
        resultElement.style.color = "green";
        inputElement.style.backgroundColor = "lightgreen";
    } 
    else 
    {
        resultElement.textContent = "Maalesef, yanlış cevap. Lütfen tekrar deneyin.";
        resultElement.style.color = "red";
        inputElement.style.backgroundColor = "red";
    }
}