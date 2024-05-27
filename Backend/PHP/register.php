<!DOCTYPE html>
<html lang="tr-TR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kayıt Ol</title>
</head>
<body>
    <h2>Kayıt Ol</h2>
    <form action="<?php echo $_SERVER['PHP_SELF']; ?>" method="POST">
        <label for="first_name">İsim:</label>
        <input type="text" id="first_name" name="first_name" required>
        <br><br>
        <label for="last_name">Soyisim:</label>
        <input type="text" id="last_name" name="last_name" required>
        <br><br>
        <label for="username">Kullanıcı Adı:</label>
        <input type="text" id="username" name="username" required>
        <br><br>
        <label for="password">Şifre:</label>
        <input type="password" id="password" name="password" required>
        <br><br>
        <label for="birth_date">Doğum Tarihi:</label>
        <input type="date" id="birth_date" name="birth_date" required>
        <br><br>
        <button type="submit">Kayıt Ol</button>
    </form>
    
    <?php
        include("db_connect.php");

        // Veritabanına başarıyla bağlandı mesajı
        echo "Veritabanına başarıyla bağlandı";

        // Formdan gelen verileri al
        $first_name = isset($_POST['first_name']) ? mysqli_real_escape_string($db, $_POST['first_name']) : '';
        $last_name = isset($_POST['last_name']) ? mysqli_real_escape_string($db, $_POST['last_name']) : '';
        $user_name = isset($_POST['user_name']) ? mysqli_real_escape_string($db, $_POST['user_name']) : '';
        $password = isset($_POST['password']) ? mysqli_real_escape_string($db, $_POST['password']) : '';
        $birth_datetime = isset($_POST['birth_datetime']) ? mysqli_real_escape_string($db, $_POST['birth_datetime']) : '';

        // Şifreyi hashleyin
        $hashed_password = password_hash($password, PASSWORD_DEFAULT);

        // Yeni kullanıcıyı ekleyecek SQL sorgusu
        $sql = "INSERT INTO kullanicilar (first_name, last_name, user_name, password, birth_datetime, role)
                VALUES ('$first_name', '$last_name', '$user_name', '$hashed_password', '$birth_datetime', 'user')";

        // Sorguyu çalıştır
        if ($db->query($sql) === TRUE) {
            echo "<script>alert('Kayıt başarılı!'); window.location.href = '/index.html';</script>";
        } else {
            echo "<script>alert('Kayıt sırasında hata oluştu: " . $sql . " Hata mesajı: " . $db->error . "');</script>";
        }

        // Veritabanı bağlantısını kapat
        $db->close();
    ?>

</body>
</html>
