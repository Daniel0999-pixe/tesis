// functions.js

// Función para redirigir a otra página
function handleSubMenuClick(page) {
    window.scrollTo(0, 0);
    setTimeout(function() {
        window.location.href = page; // Redirigir a la nueva página
    }, 100);
}

function logout() {
    localStorage.removeItem('loggedIn'); // Eliminar el estado de inicio de sesión
    window.location.replace('index.html');}


// Mostrar el menú al pasar el cursor por la izquierda de la pantalla
document.addEventListener('mousemove', (event) => {
    const menu = document.getElementById('menu');
    if (event.clientX < 50) { // Si el cursor está cerca del borde izquierdo
        menu.classList.add('visible');
    } else { 
        menu.classList.remove('visible');
    }
});

document.addEventListener("DOMContentLoaded", function () {
    fetch("menu.html") // Carga el archivo del menú
        .then(response => response.text()) // Convierte la respuesta en texto
        .then(data => {
            document.getElementById("menu-container").innerHTML = data; // Inserta el menú
            attachMenuEvents(); // Llama a la función que maneja los eventos del menú
        })
        .catch(error => console.error("Error cargando el menú:", error));
});

function attachMenuEvents() {
    document.querySelectorAll("#menu > ul > li").forEach(item => {
        item.addEventListener("mouseenter", () => {
            const submenu = item.querySelector(".submenu");
            if (submenu) submenu.style.display = "block";
        });

        item.addEventListener("mouseleave", () => {
            const submenu = item.querySelector(".submenu");
            if (submenu) submenu.style.display = "none";
        });
    });
}

// Mostrar/ocultar submenús al pasar el mouse sobre los elementos principales
document.querySelectorAll('#menu > ul > li').forEach(item => {
    item.addEventListener('mouseenter', () => { 
        const submenu = item.querySelector('.submenu');
        if (submenu && submenu.style.display !== 'block') {
            submenu.style.display = 'block'; // Muestra el submenú al pasar el mouse sobre la opción principal
        }
    });

    item.addEventListener('mouseleave', () => { 
        const submenu = item.querySelector('.submenu');
        if (submenu) {
            submenu.style.display = 'none'; // Oculta el submenú al salir del mouse del elemento principal
        }
    });
});