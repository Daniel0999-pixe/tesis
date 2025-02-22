import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Importar joblib para guardar modelos
from sklearn.metrics import classification_report

# Cargar el DataFrame desde el archivo Excel de prueba
file_path_prueba = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\pruebaM1.xlsx'
data_prueba = pd.read_excel(file_path_prueba)

# Cargar los LabelEncoders utilizados para codificar las etiquetas
label_encoder_curso = joblib.load(r'C:\xampp\htdocs\tesis2\algoritmos\entrenamiento1\label\label_encoder_cursoM1.joblib')


# Definir características para la predicción
X_prueba = data_prueba[['resultado_tic',	'resultado_dic',	'resultado_ped']]

# Cargar los modelos entrenados
model_knn = joblib.load(r'C:\xampp\htdocs\tesis2\algoritmos\entrenamiento1\modelos\K-Nearest_NeighborsM1.joblib')
model_rf = joblib.load(r'C:\xampp\htdocs\tesis2\algoritmos\entrenamiento1\modelos\Random_ForestM1.joblib')
model_dt = joblib.load(r'C:\xampp\htdocs\tesis2\algoritmos\entrenamiento1\modelos\Decision_TreeM1.joblib')

# Almacenar los resultados de rendimiento
results = {
    'K-Nearest Neighbors': model_knn,
    'Random Forest': model_rf,
    'Decision Tree': model_dt
}

# Cargar las métricas desde el archivo CSV anterior
metrics_file_path = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\resultados_metricas_modelosM1.csv'
metrics_df = pd.read_csv(metrics_file_path)

# Determinar el mejor modelo basado en la precisión (Accuracy) del CSV
best_model_index = metrics_df['Accuracy (%)'].idxmax()
best_model_name = metrics_df.iloc[best_model_index]['Model']
best_model_accuracy = metrics_df['Accuracy (%)'].max()

# Realizar predicciones con cada modelo y almacenar en un DataFrame
predictions_df = pd.DataFrame()
#
for model_name, model in results.items():
   try:
        y_pred = model.predict(X_prueba)
        predictions_df[model_name] = label_encoder_curso.inverse_transform(y_pred)  # Guardar las predicciones como texto original
   except ValueError as e:
        print(f"Error al predecir con {model_name}: {e}")

# Crear un nuevo DataFrame para almacenar los resultados en el formato deseado
resultados_finales = []

# Reestructurar el DataFrame para incluir las predicciones de cada modelo
for index, row in data_prueba.iterrows():
    for model_name in results.keys():
        resultados_finales.append({
            'ID': row['ID'],  # Conservar la columna ID existente
            'DOCENTE': row['DOCENTE'],
            'MATERIA': row['MATERIA'],
            'AREA': row['AREA'],
            'CARRERA': row['CARRERA'],
            'CICLO': row['CICLO'],
            'resultado_tic': row['resultado_tic'],
            'agrtic': row['agrtic'],
            'resultado_dic': row['resultado_dic'],
            'agrdic': row['agrdic'],
            'resultado_ped': row['resultado_ped'], 
            'agrped': row['agrped'],
            'ESTADO': row['DOCENTE'],
            'SEMAFORO': row['DOCENTE'],
            'ALGORITMO': model_name,
            'Curso_recomendado_prediccion': predictions_df[model_name].iloc[index]
        })

# Convertir a DataFrame
resultados_finales_df = pd.DataFrame(resultados_finales)

# Guardar resultados en un nuevo archivo Excel con el formato deseado
output_file_path_predictions_all_models = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_predicciones_todos_modelosM1.xlsx'
resultados_finales_df.to_excel(output_file_path_predictions_all_models, index=False)

# Filtrar solo la predicción del mejor modelo para el segundo archivo
data_prueba_best_model = data_prueba.copy()  # Copiar todo el DataFrame original

# Agregar la columna de la predicción del mejor modelo con el nuevo nombre
y_pred_best_model = results[best_model_name].predict(X_prueba)  # Obtener las predicciones del mejor modelo

data_prueba_best_model['Curso_recomendado_prediccion'] = label_encoder_curso.inverse_transform(y_pred_best_model)  # Cambiar nombre a Curso_recomendado_prediccion

# Guardar resultados del mejor modelo en un nuevo archivo Excel
output_file_path_best_model_predictions = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_prediccionesM1.xlsx'
data_prueba_best_model.to_excel(output_file_path_best_model_predictions, index=False)

# Guardar las predicciones de cada modelo en archivos Excel separados
for model_name in results.keys():
    model_predictions_df = data_prueba.copy()
    y_pred = results[model_name].predict(X_prueba)
    model_predictions_df['Curso_recomendado_prediccion'] = label_encoder_curso.inverse_transform(y_pred)
    
    # Convertir columnas a texto original antes de guardar
    #model_predictions_df['MATERIA'] = label_encoder_materia.inverse_transform(model_predictions_df['MATERIA'])
   # model_predictions_df['CICLO'] = label_encoder_ciclo.inverse_transform(model_predictions_df['CICLO'])
    
    # Guardar resultados del modelo específico en un nuevo archivo Excel
    output_file_path_model_predictions = f'C:\\xampp\\htdocs\\tesis2\\dataframe\\modelo1\\prueba\\resultados_predicciones_{model_name}M1.xlsx'
    model_predictions_df.to_excel(output_file_path_model_predictions, index=False)

# Contar las recomendaciones de cursos para cada modelo y crear gráficas
for model_name in results.keys():
    curso_counts = predictions_df[model_name].value_counts()  # Contar recomendaciones por modelo
    
    # Definir una paleta de colores personalizada
    colors = plt.colormaps['tab20'](np.linspace(0, 1, len(curso_counts)))  # Usar colormap 'tab20'

    # Crear la gráfica de pastel para la distribución de cursos recomendados por modelo
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(curso_counts, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 7})  # Ajustar tamaño de fuente

    # Personalizar la leyenda
    plt.setp(autotexts, size=5, weight="bold", color="white")  # Estilo del texto de porcentaje
    plt.setp(texts, size=7)  # Estilo del texto de las etiquetas

    # Agregar leyenda a la derecha (ajustar bbox_to_anchor para moverla más a la izquierda)
    plt.legend(wedges, curso_counts.index, title="Cursos Recomendados", loc="center left", bbox_to_anchor=(0.75, 0, 0.5, 1))  # Mover más a la izquierda

    plt.title(f'Distribución de Cursos Recomendados - {model_name}, Escenario 1', loc='center')
    plt.axis('equal')  # Para asegurar que el pastel sea un círculo

    # Ajustar márgenes para mover todo hacia la izquierda
    plt.subplots_adjust(left=-0.2)  # Ajustar el margen izquierdo

    # Guardar la gráfica como imagen para distribución de cursos recomendados por modelo
    output_image_path_cursos = f'C:\\xampp\\htdocs\\tesis2\\imagenes\\modelo1\\prueba\\distribucion_cursos_recomendados_{model_name}M1.png'
    plt.savefig(output_image_path_cursos, format='png', bbox_inches='tight')  # Guardar la figura