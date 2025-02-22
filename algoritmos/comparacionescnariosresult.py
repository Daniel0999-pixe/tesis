import pandas as pd
import matplotlib.pyplot as plt

# Cargar los DataFrames
df_reales = pd.read_excel(r"C:\xampp\htdocs\tesis2\dataframe\datosbase\conjunto_prueba.xlsx")
df_escenario1 = pd.read_excel(r"C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_prediccionesM1.xlsx")
df_escenario2 = pd.read_excel(r"C:\xampp\htdocs\tesis2\dataframe\modelo2\prueba\resultados_prediccionesM2.xlsx")
df_escenario3 = pd.read_excel(r"C:\xampp\htdocs\tesis2\dataframe\modelo3\prueba\resultados_prediccionesM3.xlsx")
df_escenario4 = pd.read_excel(r"C:\xampp\htdocs\tesis2\dataframe\modelo4\prueba\resultados_prediccionesM4.xlsx")

# Función para calcular el porcentaje de similitud
def calcular_similitud(df_reales, df_predicciones):
    comparacion = df_reales['Curso_recomendado'] == df_predicciones['Curso_recomendado_prediccion']
    return comparacion.mean() * 100  # Porcentaje de similitud

# Calcular similitudes
similitudes = {
    'Escenario 1': calcular_similitud(df_reales, df_escenario1),
    'Escenario 2': calcular_similitud(df_reales, df_escenario2),
    'Escenario 3': calcular_similitud(df_reales, df_escenario3),
    'Escenario 4': calcular_similitud(df_reales, df_escenario4)
}

# Colores especificados
colores = ['skyblue', 'lightgreen', 'salmon', 'orange']

# Crear gráfico de pastel
plt.figure(figsize=(8, 6))
plt.pie(similitudes.values(), colors=colores, autopct='%1.1f%%', startangle=90)  # Mostrar porcentajes
plt.axis('equal')  # Para que el gráfico sea un círculo
plt.title('Porcentaje de Similitud de resultados entre Escenarios')

# Agregar leyenda
plt.legend(similitudes.keys(), title="Escenarios", loc="upper right")
plt.savefig(r'C:\xampp\htdocs\tesis2\imagenes\Similitud_resultados.png')  
plt.show()