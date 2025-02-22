import pandas as pd
import matplotlib.pyplot as plt

# Cargar los DataFrames desde los archivos Excel
df_recomendado = pd.read_excel(r'C:\xampp\htdocs\tesis2\dataframe\modelo4\conjunto_pruebaM4.xlsx')
df_prediccion = pd.read_excel(r'C:\xampp\htdocs\tesis2\dataframe\modelo4\resultados_prediccionesM4.xlsx')

# Asegurarse de que las columnas a comparar existen
if 'Curso_recomendado' not in df_recomendado.columns or 'Curso_recomendado_prediccion' not in df_prediccion.columns:
    raise ValueError("Una o ambas columnas no existen en los DataFrames.")

# Contar la frecuencia de cada valor en ambas columnas
frecuencia_recomendado = df_recomendado['Curso_recomendado'].value_counts()
frecuencia_prediccion = df_prediccion['Curso_recomendado_prediccion'].value_counts()

# Crear un DataFrame para facilitar la comparación
comparacion_df = pd.DataFrame({
    'Curso Recomendado': frecuencia_recomendado,
    'Curso Recomendado Predicción': frecuencia_prediccion
}).fillna(0)  # Rellenar NaN con 0 para valores que no están presentes en ambos

# Graficar la comparación
ax = comparacion_df.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'salmon'])
plt.title('Comparación de Cursos Recomendados vs Cursos Recomendados por Predicción, Modelo #3')
plt.ylabel('Frecuencia')
plt.xticks(rotation=0)

# Ajustar el eje y para que vaya de 0 a 50
plt.ylim(0, 50)

# Ubicar la leyenda fuera del gráfico a la derecha
plt.legend(title='Campos comparados', bbox_to_anchor=(1.05, 1), loc='upper left')

# Cambiar las etiquetas del eje x a números asignados
plt.xticks(ticks=range(len(comparacion_df)), labels=[str(i + 1) for i in range(len(comparacion_df))])

# Crear una segunda leyenda con los cursos enumerados
curso_numeros = {i + 1: curso for i, curso in enumerate(comparacion_df.index)}
curso_numeros_str = '\n'.join([f"{num}: {curso}" for num, curso in curso_numeros.items()])
plt.figtext(0.73, 0.45, curso_numeros_str, fontsize=6, bbox=dict(facecolor='white', alpha=0.5))

# Ajustar el diseño para que todo se vea bien
plt.tight_layout()

# Guardar el gráfico como PNG
plt.savefig(r'C:\xampp\htdocs\tesis2\comparacion\grafico_comparacionM4.png')

# Mostrar el gráfico
plt.show()

# Exportar los datos de comparación a un archivo CSV
comparacion_df.to_csv(r'C:\xampp\htdocs\tesis2\comparacion\comparacion_cursosM4.csv', index=True)



