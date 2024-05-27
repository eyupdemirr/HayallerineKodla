<?php
include('db_connect.php');

// Kullanıcılar tablosundan tüm verileri seç
$sql = "SELECT * FROM kullanicilar";
$result = $db->query($sql);

// Hata kontrolü
if ($result->num_rows > 0) {
    echo "Kullanıcılar:\n";
    while ($row = $result->fetch_assoc()) {
        echo "ID: " . $row["id"] . " - Ad: " . $row["first_name"] . " Soyad: " . $row["last_name"] . "\n";
    }
} else {
    echo "Sonuç bulunamadı.";
}

// Eğitimler tablosundan tüm verileri seç
$sql = "SELECT * FROM egitimler";
$result = $db->query($sql);

// Hata kontrolü
if ($result->num_rows > 0) {
    echo "Eğitimler:\n";
    while ($row = $result->fetch_assoc()) {
        echo "ID: " . $row["id"] . " - Başlık: " . $row["baslik"] . "\n";
    }
} else {
    echo "Eğitimler tablosu boş.";
}

// Kullanıcı_Egitimleri tablosundan tüm verileri seç
$sql = "SELECT * FROM kullanici_egitimleri";
$result = $db->query($sql);

// Hata kontrolü
if ($result->num_rows > 0) {
    echo "Kullanıcı Eğitimleri:\n";
    while ($row = $result->fetch_assoc()) {
        echo "Kullanıcı ID: " . $row["kullanici_id"] . " - Eğitim ID: " . $row["egitim_id"] . "\n";
    }
} else {
    echo "Kullanıcı Eğitimleri tablosu boş.";
}

$db->close(); // Veritabanı bağlantısını kapat
