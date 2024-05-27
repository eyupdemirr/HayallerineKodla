document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("signUp");

    form.addEventListener("submit", function (event) {
        event.preventDefault(); // Formun varsayılan davranışını engelle

        // Formdaki değerleri al
        const firstName = document.getElementById("first_name").value;
        const lastName = document.getElementById("last_name").value;
        const userName = document.getElementById("user_name").value;
        const password = document.getElementById("password").value;
        const birthDate = document.getElementById("birth_datetime").value;

        // POST isteği göndermek için FormData oluştur
        const formData = new FormData();
        formData.append("first_name", firstName);
        formData.append("last_name", lastName);
        formData.append("user_name", userName);
        formData.append("password", password);
        formData.append("birth_datetime", birthDate);

        // Sunucuya veri göndermek için fetch API'sini kullan
        fetch("/Backend/php/db_userSignUp.php", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.text(); // Sunucudan gelen metni al
            } else {
                throw new Error("Kayıt sırasında bir hata oluştu.");
            }
        })
        .then(data => {
            alert(data); // Sunucudan gelen mesajı göster
        })
        .catch(error => {
            console.error("Hata:", error);
            alert("Bir hata oluştu, lütfen tekrar deneyin.");
        });
    });
});
