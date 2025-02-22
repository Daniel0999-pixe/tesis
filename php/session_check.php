<?php
// Iniciar la sesión
session_start(); // Asegúrate de que la sesión esté iniciada

// Verificar si el usuario está logueado
if (!isset($_SESSION['user_id'])) {
header("Location: ../html/index.html"); // Redirigir a la página de inicio de sesión si no está logueado
exit();
}

// Si el usuario está logueado, incluye el contenido principal
include '../html/menu.html';
?>