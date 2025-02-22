import pandas as pd
import matplotlib.pyplot as plt


# Cargar el DataFrame de prueba
file_path_prueba = r'C:\xampp\htdocs\tesis2\dataframe\modelo4\entrenamiento\conjunto_pruebaM4.xlsx'
data_prueba = pd.read_excel(file_path_prueba)

# Cargar el DataFrame de resultados de predicciones
file_path_predicciones = r'C:\xampp\htdocs\tesis2\dataframe\modelo4\prueba\resultados_predicciones_todos_modelosM4.xlsx'
data_predicciones = pd.read_excel(file_path_predicciones)

# Renombrar columnas en data_prueba para evitar conflictos
data_prueba.rename(columns={
    'DOCENTE': 'DOCENTE_orig',
    'MATERIA': 'MATERIA_orig',
    'AREA': 'AREA_orig',
    'CARRERA': 'CARRERA_orig',
    'CICLO': 'CICLO_orig',
    'resultado_tic': 'resultado_tic_orig',
    'resultado_dic': 'resultado_dic_orig',
    'resultado_ped': 'resultado_ped_orig',
    'total_proceso': 'total_proceso_orig',
    'ESTADO': 'ESTADO_orig',
    'SEMAFORO': 'SEMAFORO_orig'
}, inplace=True)

# Inicializar una lista para almacenar los resultados finales
resultados_comparacion = []

# Iterar sobre cada registro en data_prueba
for index, row in data_prueba.iterrows():
    # Obtener el ID del registro actual
    id_actual = row['ID']
    
    # Iterar sobre cada modelo y agregar las predicciones correspondientes
    for model_name in data_predicciones['ALGORITMO'].unique():
        # Filtrar las predicciones para el modelo actual
        predicciones_modelo = data_predicciones[data_predicciones['ALGORITMO'] == model_name]
        
        # Obtener la predicción correspondiente al ID actual
        prediccion_row = predicciones_modelo[predicciones_modelo['ID'] == id_actual]
        
        if not prediccion_row.empty:
            # Si hay una predicción para este ID, construir el resultado
            resultado_final = {
                'ID': row['ID'],
                'DOCENTE': row['DOCENTE_orig'],
                'MATERIA': row['MATERIA_orig'],
                'AREA': row['AREA_orig'],
                'CARRERA': row['CARRERA_orig'],
                'CICLO': row['CICLO_orig'],
                'resultado_tic': row['resultado_tic_orig'],
                'agrtic': prediccion_row.iloc[0].get('agrtic', None),  
                'resultado_dic': row['resultado_dic_orig'],
                'agrdic': prediccion_row.iloc[0].get('agrdic', None),                  
                'resultado_ped': row['resultado_ped_orig'],
                'agrped': prediccion_row.iloc[0].get('agrped', None),  
                'total_proceso': row['total_proceso_orig'],
                'ESTADO': row['ESTADO_orig'],
                'SEMAFORO': row['SEMAFORO_orig'],
                'Curso_recomendado': row.get('Curso_recomendado', None),  # Suponiendo que esta columna existe en data_prueba
                'tipo': 1,  # Establecer tipo en 1
                'algoritmo': model_name,
                'Curso_recomendado_prediccion': prediccion_row.iloc[0]['Curso_recomendado_prediccion'],  # Obtener la predicción del curso
                'resultado': int(row.get('Curso_recomendado', None) == prediccion_row.iloc[0]['Curso_recomendado_prediccion'])  # Comparar y asignar resultado
            }
            resultados_comparacion.append(resultado_final)

# Convertir la lista a un DataFrame
resultados_comparacion_df = pd.DataFrame(resultados_comparacion)

# Guardar el DataFrame final en un archivo Excel
output_file_path_comparacion = r'C:\xampp\htdocs\tesis2\dataframe\modelo4\prueba\resultados_comparacionM4.xlsx'
resultados_comparacion_df.to_excel(output_file_path_comparacion, index=False)

# Calcular porcentajes de similitud por modelo
similitudes = resultados_comparacion_df.groupby('algoritmo')['resultado'].mean() * 100
# Crear gráfico de pastel
colores = ['skyblue', 'lightgreen', 'salmon', 'orange']
plt.figure(figsize=(8, 6))
plt.pie(similitudes, labels=None, colors=colores, autopct='%1.2f%%', startangle=90)  # Ocultar etiquetas de algoritmos
plt.axis('equal')  # Para que el gráfico sea un círculo
plt.title('Porcentaje de Similitud entre Resultados Predichos y Reales Escenario 4')

# Agregar leyenda con los nombres de los algoritmos
plt.legend(similitudes.index, title="Algoritmos", loc="upper right")

# Guardar el gráfico como imagen
plt.savefig(r'C:\xampp\htdocs\tesis2\imagenes\modelo4\prueba\grafico_similitud_M4.png')
plt.show()