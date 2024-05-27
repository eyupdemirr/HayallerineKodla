<?php
$dbHost = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "hayallerinekodla";

// Bağlantı kurma
$db = new mysqli($dbHost, $dbUsername, $dbPassword, $dbName);
mysqli_set_charset($db, "UTF8");

// Hata kontrolü
if ($db->connect_error) {
    die("Veritabanına bağlantı sağlanamadı: " . $db->connect_error);
} else {
    echo ("Başarıyla bağlandı \n");
}
