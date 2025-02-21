<?php
require 'vendor/autoload.php';

use PhpOffice\PhpSpreadsheet\IOFactory;

try {
    $archivo = 'C:\\xampp\\htdocs\\tesis\\dataset\\usuarios.xlsx';
    $documento = IOFactory::load($archivo);
    echo "PHPSpreadsheet está funcionando correctamente.";
} catch (Exception $e) {
    echo "Error: " . $e->getMessage();
}
?>