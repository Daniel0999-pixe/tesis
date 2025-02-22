import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Rutas de los archivos CSV
rutas_archivos = [
    r"C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\resultados_metricas_modelosM1.csv",
    r"C:\xampp\htdocs\tesis2\dataframe\modelo2\entrenamiento\resultados_metricas_modelosM2.csv",
    r"C:\xampp\htdocs\tesis2\dataframe\modelo3\entrenamiento\resultados_metricas_modelosM3.csv",
    r"C:\xampp\htdocs\tesis2\dataframe\modelo4\entrenamiento\resultados_metricas_modelosM4.csv"
]

# Nombres de los modelos
nombres_modelos = ['Escenario 1', 'Escenario 2', 'Escenario 3', 'Escenario 4']

# Lista para almacenar los datos de Accuracy y Algoritmos
data = []

# Cargar datos y extraer la columna "Model" y "Accuracy (%)"
for ruta in rutas_archivos:
    df = pd.read_csv(ruta)
    data.append(df[['Model', 'Accuracy (%)']])  # Guardamos solo las columnas necesarias

# Crear un DataFrame combinado
df_combinado = pd.concat(data, keys=nombres_modelos, names=['Modelo', 'Index']).reset_index()

# Crear el gráfico
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar las posiciones para las barras
x = np.arange(len(nombres_modelos))  # Posiciones base para cada modelo
width = 0.20  # Ancho de cada barra

# Obtener los nombres únicos de los algoritmos (columna Model)
algoritmos = df_combinado['Model'].unique()

# Colores para las barras
colores = ['skyblue', 'salmon', 'lightgreen']

# Crear barras para cada algoritmo
for i, algoritmo in enumerate(algoritmos):
    accuracies = []
    for modelo in nombres_modelos:
        # Filtrar por modelo y algoritmo específico
        accuracy = df_combinado[(df_combinado['Modelo'] == modelo) & (df_combinado['Model'] == algoritmo)]['Accuracy (%)']
        accuracies.append(accuracy.values[0] if not accuracy.empty else 0)  # Si no hay datos, usar 0
    
    # Crear las barras
    bars = ax.bar(x + i * width, accuracies, width, label=algoritmo, color=colores[i % len(colores)])

    # Añadir etiquetas sobre las barras
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

# Añadir etiquetas y título
ax.set_xlabel('Escenarios')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparación de Accuracy entre Escenarios')
ax.set_xticks(x + width)
ax.set_xticklabels(nombres_modelos)
ax.legend(title='Algoritmos')

# Ajustar límites del eje Y
ax.set_ylim(0, 100)

# Ajustar diseño y mostrar el gráfico
plt.tight_layout()
plt.savefig(r'C:\xampp\htdocs\tesis2\imagenes\comparacionaccuracy.png')  
plt.show()

