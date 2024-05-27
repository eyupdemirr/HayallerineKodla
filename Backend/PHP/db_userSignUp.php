<?php
// Veritabanı bağlantısını yap
include("db_connect.php");

// POST verilerini al
$first_name = isset($_POST['first_name']) ? mysqli_real_escape_string($db, $_POST['first_name']) : '';
$last_name = isset($_POST['last_name']) ? mysqli_real_escape_string($db, $_POST['last_name']) : '';
$user_name = isset($_POST['user_name']) ? mysqli_real_escape_string($db, $_POST['user_name']) : '';
$password = isset($_POST['password']) ? mysqli_real_escape_string($db, $_POST['password']) : '';
$birth_datetime = isset($_POST['birth_datetime']) ? mysqli_real_escape_string($db, $_POST['birth_datetime']) : '';

// Şifreyi hashle
$hashed_password = password_hash($password, PASSWORD_DEFAULT);

// SQL sorgusunu hazırla ve çalıştır
$sql = "INSERT INTO kullanicilar (first_name, last_name, user_name, password, birth_datetime, role)
        VALUES ('$first_name', '$last_name', '$user_name', '$hashed_password', '$birth_datetime', 'user')";

if ($db->query($sql) === TRUE) {
    echo "Kayıt başarılı!";
} else {
    echo "Kayıt sırasında hata oluştu: " . $db->error;
}

// Veritabanı bağlantısını kapat
$db->close();
