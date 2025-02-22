import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
import joblib 
print(sklearn.__version__)
# Cargar el DataFrame desde el archivo Excel
file_path = r'C:\xampp\htdocs\tesis2\dataframe\datosbase\datosentreno2.xlsx'
data = pd.read_excel(file_path)

# Preprocesamiento de datos
label_encoder_curso = LabelEncoder()
data['Curso_recomendado'] = label_encoder_curso.fit_transform(data['Curso_recomendado'])
joblib.dump(label_encoder_curso, r'C:\xampp\htdocs\tesis2\algoritmos\entrenamiento1\label\label_encoder_cursoM1.joblib')

# Definir características y variable objetivo
X = data[['resultado_tic', 'resultado_dic', 'resultado_ped']]
y = data['Curso_recomendado']

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eliminar clases con pocos ejemplos
threshold = 5  # Umbral para eliminar clases
counts = y.value_counts()
to_remove = counts[counts < threshold].index

# Filtrar X e y para eliminar las clases con pocos ejemplos
y_filtered = y[~y.isin(to_remove)]
X_filtered = X[~y.isin(to_remove)]

# Dividir los datos en conjuntos de entrenamiento y prueba (estratificados)
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=42)

# Crear un DataFrame para el conjunto de prueba con todas las columnas necesarias
test_set = pd.DataFrame(X_test, columns=X.columns)  # Crear DataFrame con las características
# Crear un DataFrame para el conjunto de prueba con todas las columnas necesarias
test_set = pd.DataFrame(X_test, columns=X.columns)  # Crear DataFrame con las características
# Agregar la variable objetivo (Curso_recomendado) y las las demás columnas al DataFrame
test_set['Curso_recomendado'] = label_encoder_curso.inverse_transform(y_test.values)
test_set['ID'] = data.loc[X_test.index, 'ID'].values
test_set['DOCENTE'] = data.loc[X_test.index, 'DOCENTE'].values
test_set['MATERIA'] = data.loc[X_test.index, 'MATERIA'].values
test_set['AREA'] = data.loc[X_test.index, 'AREA'].values
test_set['CARRERA'] = data.loc[X_test.index, 'CARRERA'].values
test_set['CICLO'] = data.loc[X_test.index, 'CICLO'].values
test_set['ESTADO'] = data.loc[X_test.index, 'ESTADO'].values
test_set['SEMAFORO'] = data.loc[X_test.index, 'SEMAFORO'].values
test_set['agrtic'] = data.loc[X_test.index, 'agrtic'].values
test_set['agrdic'] = data.loc[X_test.index, 'agrdic'].values
test_set['agrped'] = data.loc[X_test.index, 'agrped'].values
test_set['total_proceso'] = data.loc[X_test.index, 'total_proceso'].values
# Reordenar las columnas según el orden deseado
column_order = [
    'ID', 'DOCENTE', 'MATERIA', 'AREA', 'CARRERA', 'CICLO',
    'resultado_tic','agrtic', 'resultado_dic','agrdic', 'resultado_ped', 'agrped','total_proceso',
    'ESTADO', 'SEMAFORO', 'Curso_recomendado'
]
test_set = test_set[column_order]
# Función para ordenar el DataFrame por ID antes de guardar
def save_sorted_dataframe(df, file_path):
    df_sorted = df.sort_values(by='ID')  # Ordenar por ID
    df_sorted.to_excel(file_path, index=False)  # Guardar como archivo Excel
# Guardar el conjunto de prueba en un archivo Excel con codificación UTF-8
test_set_output_file_path = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\conjunto_pruebaM1.xlsx'
save_sorted_dataframe(test_set, test_set_output_file_path)
# Inicializar un DataFrame para almacenar TP, TN, FP y FN totales de cada modelo.
confusion_totals_summary = []

# Modelos a utilizar
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}
# Inicializar resultados y matrices
results = {}
# Inicializar un DataFrame para almacenar TP, TN, FP y FN totales de cada modelo.
confusion_totals_summary = []
normal_results_summary = []
# Rutas para guardar los modelos
model_path = r'C:\xampp\htdocs\tesis2\algoritmos\entrenamiento1\modelos'

# Entrenar y evaluar cada modelo
for model_name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Guardar el modelo entrenado
    model_filename = f'{model_path}/{model_name.replace(" ", "_")}M1.joblib'
    joblib.dump(model, model_filename)
    print(f'Modelo {model_name} guardado en {model_filename}')
    
    # Predecir las clases y las probabilidades
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Obtener probabilidades para todas las clases
    
    # Calcular la matriz de confusión individual
    cm = confusion_matrix(y_test, y_pred)

    # Calcular TP, TN, FP y FN totales por clase
    TP_total = np.diag(cm).astype(float)  # Verdaderos positivos por clase
    FP_total = cm.sum(axis=0).astype(float) - TP_total  # Falsos positivos por clase
    FN_total = cm.sum(axis=1).astype(float) - TP_total  # Falsos negativos por clase
    TN_total = cm.sum().astype(float) - (FP_total + FN_total + TP_total)  # Verdaderos negativos por clase

    # Almacenar los resultados normales en el resumen
    for i in range(len(TP_total)):
        normal_results_summary.append({
            'Modelo': model_name,
            'Clase': label_encoder_curso.inverse_transform([i])[0],  # Decodificar el índice a su nombre original
            'TP': TP_total[i],
            'TN': TN_total[i],
            'FP': FP_total[i],
            'FN': FN_total[i]
        })

    # Almacenar los totales en el resumen
    confusion_totals_summary.append({
        'Modelo': model_name,
        'TP Total': TP_total.sum(),
        'TN Total': TN_total.sum(),
        'FP Total': FP_total.sum(),
        'FN Total': FN_total.sum()
    })

    # Calcular métricas con zero_division para evitar advertencias
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    accuracy = report['accuracy'] * 100  # Convertir a porcentaje
    f1_score = report['weighted avg']['f1-score'] * 100  # Convertir a porcentaje
    recall = report['weighted avg']['recall'] * 100  # Convertir a porcentaje
    
    confidence = accuracy / 100  # Valor de confianza entre 0 y 1.
    error_rate = 1 - confidence     # Tasa de error.

    results[model_name] = {
        'Accuracy (%)': accuracy,
        'F1 Score (%)': f1_score,
        'Recall (%)': recall,
        'CV Score (%)': np.mean(cross_val_score(model, X_filtered, y_filtered, cv=5)) * 100,
        'Confianza (%)': confidence * 100,
        'Error (%)': error_rate * 100
    }

# Crear un DataFrame con los resultados normales.
normal_results_df = pd.DataFrame(normal_results_summary)

# Guardar los resultados normales en un archivo Excel.
normal_results_excel_path = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\resultados_normales_MC_M1.xlsx'
with pd.ExcelWriter(normal_results_excel_path) as writer:
    normal_results_df.to_excel(writer, sheet_name='Resultados Normales', index=False)

print("Resultados normales guardados exitosamente en formato XLSX.")

# Crear un DataFrame con el resumen de la matriz de confusión total.
confusion_totals_summary_df = pd.DataFrame(confusion_totals_summary)

# Dividir cada total por el número de clases.
num_classes = len(label_encoder_curso.classes_)
confusion_totals_summary_df['TP Total'] /= num_classes
confusion_totals_summary_df['TN Total'] /= num_classes
confusion_totals_summary_df['FP Total'] /= num_classes
confusion_totals_summary_df['FN Total'] /= num_classes

# Guardar los promedios en un solo archivo Excel.
excel_file_path = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\totales_matriz_confusion_resumen_M1.xlsx'
with pd.ExcelWriter(excel_file_path) as writer:
    confusion_totals_summary_df.to_excel(writer, sheet_name='Resumen Totales', index=False)

print("Resumen de totales guardado exitosamente en formato XLSX.")

# Graficar la matriz de totales para cada modelo.
for model in confusion_totals_summary_df['Modelo']:
    model_data = confusion_totals_summary_df[confusion_totals_summary_df['Modelo'] == model]
    
    plt.figure(figsize=(6, 4))
    
    # Crear una matriz de confusión ficticia para la visualización (solo para ilustración)
    cm_display_data = np.array([
        [model_data['TP Total'].values[0], model_data['FP Total'].values[0]],
        [model_data['FN Total'].values[0], model_data['TN Total'].values[0]]
    ])
    
    sns.heatmap(cm_display_data, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Total'})
    
    plt.title(f'Matriz de Confusión - {model}')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    
    plt.savefig(f'C:\\xampp\\htdocs\\tesis2\\imagenes\\modelo1\\entrenamiento\\matriz_confusion_{model.replace(" ", "_")}M1.png', bbox_inches='tight')
    plt.close()

print("Matrices de confusión guardadas exitosamente como imágenes PNG.")

roc_metrics = []  # Lista para almacenar los resultados de las curvas ROC

plt.figure(figsize=(12, 6))  # Configurar el tamaño del gráfico

for model_name, model in models.items():
    model.fit(X_train, y_train)  # Entrenar el modelo
    y_pred_proba = model.predict_proba(X_test)  # Obtener probabilidades para todas las clases

    all_fpr = np.linspace(0, 1, 100)  # Crear un rango común de FPR
    mean_tpr = np.zeros_like(all_fpr)  # Inicializar TPR promedio

    for i in range(len(label_encoder_curso.classes_)):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])  # Calcular FPR y TPR para cada clase
        mean_tpr += np.interp(all_fpr, fpr, tpr)  # Interpolar TPR al rango común de FPR

    mean_tpr /= len(label_encoder_curso.classes_)  # Promediar TPR
    mean_auc = auc(all_fpr, mean_tpr)  # Calcular AUC promedio

    # Graficar la curva ROC promedio del modelo actual
    plt.plot(all_fpr, mean_tpr, lw=2, label=f'{model_name} (AUC = {mean_auc:.2f})')

    # Expandir los datos: Cada combinación de FPR y TPR será una fila separada
    for fpr_value, tpr_value in zip(all_fpr, mean_tpr):
        roc_metrics.append({
            'Modelo': model_name,
            'FPR': fpr_value,
            'TPR': tpr_value,
            'AUC': mean_auc
        })

# Agregar línea diagonal (random guess)
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC Promediadas por Modelo, Escenario 1')
plt.legend(loc='lower right')    
plt.savefig(r'C:\xampp\htdocs\tesis2\imagenes\modelo1\entrenamiento\curva_roc_M1.png', bbox_inches='tight')  # Guardar la imagen
plt.close()

# Crear un DataFrame con los resultados para exportar a Excel
roc_metrics_df = pd.DataFrame(roc_metrics)

# Guardar los resultados en un archivo XLSX con formato en filas detalladas
roc_metrics_output_file_path = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\resultados_roc_train_M1.xlsx'
with pd.ExcelWriter(roc_metrics_output_file_path, engine='openpyxl') as writer:
    roc_metrics_df.to_excel(writer, sheet_name='Resultados ROC', index=False)

    # Ajustar el ancho de las columnas automáticamente
    worksheet = writer.sheets['Resultados ROC']
    for column in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column if cell.value is not None)
        column_letter = column[0].column_letter
        worksheet.column_dimensions[column_letter].width = max_length + 2

print("Resultados guardados exitosamente en formato XLSX.")

# Calcular distancias usando KNN con distancia Manhattan
knn_model_manhattan = KNeighborsClassifier(metric='manhattan')

# Ajustar el modelo a los datos de entrenamiento
knn_model_manhattan.fit(X_train, y_train)

# Calcular las distancias entre los puntos de prueba y los puntos de entrenamiento
distances_manhattan, indices_manhattan = knn_model_manhattan.kneighbors(X_test)

# Crear un DataFrame para almacenar las distancias con Curso_recomendado correspondiente antes que las distancias
distances_df_manhattan = pd.DataFrame(distances_manhattan,
                                       columns=[f'Distancia_{i+1}' for i in range(distances_manhattan.shape[1])])
distances_df_manhattan['Curso_recomendado'] = label_encoder_curso.inverse_transform(y_test.values)

# Reordenar las columnas para que Curso_recomendado esté primero
column_order = ['Curso_recomendado'] + [f'Distancia_{i+1}' for i in range(distances_manhattan.shape[1])]
distances_df_manhattan = distances_df_manhattan[column_order]

# Guardar las distancias en un archivo Excel (.xlsx)
distances_output_file_path_manhattan_xlsx = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\distancias_knn_manhattan.xlsx'
with pd.ExcelWriter(distances_output_file_path_manhattan_xlsx) as writer:
    distances_df_manhattan.to_excel(writer, sheet_name='Distancias KNN', index=False)

print("Distancias calculadas y guardadas exitosamente en formato XLSX.")

# Crear un DataFrame con las métricas para guardarlas en un archivo CSV
metrics_df = pd.DataFrame(results).T  # Transponer para que los modelos sean filas
# Agregar una columna con los nombres de los modelos (opcional)
metrics_df.reset_index(inplace=True)  # Restablecer el índice para convertirlo en una columna normal
metrics_df.rename(columns={'index': 'Model'}, inplace=True)  # Renombrar la columna del índice a "Model"
# Guardar las métricas en un archivo CSV
metrics_output_file_path = r'C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\resultados_metricas_modelosM1.csv'
metrics_df.to_csv(metrics_output_file_path, index=False)  # index=False ya que ahora "Model" es una columna normal

# Crear gráfico de barras para las métricas incluyendo CV Score
model_names = metrics_df['Model'].tolist()
x = np.arange(len(model_names))  # Posiciones en el eje x
plt.figure(figsize=(9, 4)) 
bar_width = 0.35  # Ancho de las barras
num_models = len(metrics_df)  # Número total de modelos
x_positions_with_spacing = np.arange(num_models) * (bar_width * 5 + 0.2)  
plt.bar(x_positions_with_spacing - bar_width * 2.0, metrics_df['Accuracy (%)'], width=bar_width,
        label='Precisión en Test', color='skyblue')
plt.bar(x_positions_with_spacing - bar_width, metrics_df['CV Score (%)'], width=bar_width,
        label='Precisión CV', color='lightgreen')
plt.bar(x_positions_with_spacing, metrics_df['Recall (%)'], width=bar_width,
        label='Recall (%)', color='salmon')
plt.bar(x_positions_with_spacing + bar_width, metrics_df['F1 Score (%)'], width=bar_width,
        label='F1-Score (%)', color='orange')
plt.ylabel('Métricas (%)')
plt.title('Comparación de Métricas por Modelo, Escenario 1')
plt.xticks(x_positions_with_spacing, model_names)  

for i in range(len(metrics_df)):
    plt.text(x_positions_with_spacing[i] - bar_width * 2.0,
             metrics_df['Accuracy (%)'].iloc[i] + 1,
             f'{metrics_df["Accuracy (%)"].iloc[i]:.2f}%', ha='center', va='bottom', fontsize=6)
    plt.text(x_positions_with_spacing[i] - bar_width,
             metrics_df['CV Score (%)'].iloc[i] + 1,
             f'{metrics_df["CV Score (%)"].iloc[i]:.2f}%', ha='center', va='bottom', fontsize=6)
    plt.text(x_positions_with_spacing[i],
             metrics_df['Recall (%)'].iloc[i] + 1,
             f'{metrics_df["Recall (%)"].iloc[i]:.1f}%', ha='center', va='bottom', fontsize=6)
    plt.text(x_positions_with_spacing[i] + bar_width,
             metrics_df['F1 Score (%)'].iloc[i] + 1,
             f'{metrics_df["F1 Score (%)"].iloc[i]:.1f}%', ha='center', va='bottom', fontsize=6)
plt.legend(title='Métricas', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  
plt.ylim(0, 100)
# Guardar el gráfico como imagen en la ruta especificada 
plt.savefig(r'C:\xampp\htdocs\tesis2\imagenes\modelo1\entrenamiento\metricas_modelosM1.png')  
plt.close()