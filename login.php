<?php
require 'vendor/autoload.php'; // Asegúrate de que la ruta sea correcta

use PhpOffice\PhpSpreadsheet\IOFactory;

// Ruta del archivo Excel que contiene las credenciales
$archivo = "C:\\xampp\\htdocs\\tesis\\dataset\\usuarios.xlsx";

try {
    // Cargar el archivo Excel
    $documento = IOFactory::load($archivo);
    $hoja = $documento->getActiveSheet();
    $highestRow = $hoja->getHighestRow();

    // Obtener las credenciales ingresadas por el usuario
    $username = $_POST['username'];
    $password = $_POST['password'];
    $loginSuccessful = false;

    // Iterar sobre las filas del archivo Excel para verificar las credenciales
    for ($row = 2; $row <= $highestRow; $row++) {
        // Suponiendo que el usuario está en la columna A y la contraseña en la columna B
        $storedUsername = $hoja->getCell("A" . $row)->getValue();
        $storedPassword = $hoja->getCell("B" . $row)->getValue();

        // Comparar las credenciales ingresadas con las almacenadas
        if ($username === $storedUsername && $password === $storedPassword) {
            $loginSuccessful = true;
            break;
        }
    }

    // Verificar si el inicio de sesión fue exitoso
    if ($loginSuccessful) {
        // Redirigir a la página deseada si las credenciales son correctas
        header("Location: xd.php");
        exit();
    } else {
        // Manejar el error de inicio de sesión
        echo "Credenciales incorrectas.";
    }
} catch (Exception $e) {
    // Manejar cualquier error al cargar el archivo Excel
    echo "Error al cargar el archivo: " . $e->getMessage();
}
?>
