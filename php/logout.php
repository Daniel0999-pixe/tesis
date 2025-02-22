<?php
session_start(); // Iniciar la sesión

// Destruir todas las variables de sesión y destruir la sesión.
session_unset(); 
session_destroy(); 

// Redirigir al login.
header("Location: ../html/index.html"); // Ruta actualizada aquí.
exit();
?>