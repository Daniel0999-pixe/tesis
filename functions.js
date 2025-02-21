// functions.js

// Función para redirigir a otra página
function handleSubMenuClick(page) {
    // Desplazar hacia arriba
    window.scrollTo(0, 0);
    // Esperar un momento antes de redirigir para permitir el desplazamiento
    setTimeout(function() {
        window.location.href = page; // Redirigir a la nueva página
    }, 100); // Ajusta el tiempo según sea necesario
}

function logout() {
    window.location.href = 'logout.php'; // Redirigir al script de cierre de sesión
}

// Mostrar el menú al pasar el cursor por la izquierda de la pantalla
document.addEventListener('mousemove', (event) => {
    const menu = document.getElementById('menu');
    if (event.clientX < 50) { // Si el cursor está cerca del borde izquierdo
        menu.classList.add('visible'); // Agrega la clase 'visible'
    } else {
        menu.classList.remove('visible'); // Elimina la clase 'visible'
    }
});

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
