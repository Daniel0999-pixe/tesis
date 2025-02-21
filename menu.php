<?php
// secod_v3.php

// Verificar el estado de la sesión antes de iniciar
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}
?>
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SISTEMA DE EVALUACION DE MODELOS DE M.L. PARA PLATAFORMA SECOED V.3</title>
    <link rel="icon" href="../imagenes/ug.png">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-image: url('../imagenes/LogoUGcolor.png'); /* Ajusta la ruta para acceder a la imagen */
            background-size: 30%; /* Ajusta el ancho al 30% */
            background-repeat: no-repeat; /* Evita que la imagen se repita */
            background-position: center center;
            background-attachment: fixed;
            background-color: white; /* Cambiar fondo a blanco */
            margin: 0;
            padding: 0;
            height: 100vh; /* Altura completa para ver el menú */
            overflow: hidden; /* Evita el scroll */
        }
        
        #menu {
            width: 200px;
            background-color: #113C70; /* Color del menú */
            color: white;
            position: fixed;
            left: -200px; /* Oculto inicialmente */
            top: 0; /* Fijo a la parte superior */
            height: 100%; /* Altura completa del menú */
            transition: left 0.3s; /* Animación suave */
        }
        
        #menu.visible {
            left: 0; /* Muestra el menú cuando tiene la clase 'visible' */
        }

        #menu ul {
            list-style-type: none;
            padding: 0;
        }

        #menu ul li {
            padding: 10px;
            font-size: 13px; /* Reducir el tamaño de la fuente */
            border-bottom: 1px solid #015DA2; /* Color de borde */
            cursor: pointer; /* Cambia el cursor al pasar sobre los elementos */
        }

        #menu ul li:hover {
            background-color: white; /* Color de fondo al pasar el cursor sobre la opción principal */
            color: #113C70; /* Cambiar texto a azul oscuro al pasar el mouse */
        }

        .submenu {
            display: none; /* Oculta los submenús inicialmente */
            padding-left: 10px; /* Reducir la indentación para submenús */
            background-color: white; /* Fondo blanco para submenús */
        }

        .submenu li {
            color: black; /* Texto negro para submenús */
            font-size: 13px; /* Reducir el tamaño de la fuente para submenús */
        }

        .submenu li:hover {
            background-color: #428ca1; /* Color de fondo al pasar sobre los elementos del submenú */
        }

        #content {
            margin-left: 220px; /* Espacio para el menú */
            padding: 20px;
            color: #113C70; /* Color del texto del contenido */
            height: calc(100vh - 40px); /* Altura ajustada para evitar scroll */
            overflow-y: auto; /* Permite scroll si es necesario */
        }

        h1 {
            text-align: center;
            color: #113C70; /* Color del título principal */
            font-size: 36px; /* Tamaño de fuente del título principal */
        }
    </style>
</head>
<body>

<h1>SISTEMA DE EVALUACION DE MODELOS DE M.L.</h1>

<!-- Incluir el menú HTML -->
<?php include 'menu.html'; ?>

<div id="content">
    <!-- Aquí puedes agregar contenido adicional -->
</div>

<script src="functions.js"></script> <!-- Archivo JavaScript externo -->
</body>
</html>