# Importar bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Cargar los datos reales desde el archivo Excel
datos_reales = pd.read_excel(r"C:\xampp\htdocs\tesis2\dataframe\modelo1\entrenamiento\conjunto_pruebaM1.xlsx")
y_true = datos_reales['Curso_recomendado']  # Etiquetas reales

# Binarizar las etiquetas reales (convertir a formato "uno contra el resto")
unique_classes = y_true.unique()
y_true_bin = label_binarize(y_true, classes=unique_classes)
n_classes = y_true_bin.shape[1]

# Inicializar la figura para graficar con un tamaño mayor
plt.figure(figsize=(12, 8))  # Ajusta el tamaño de la figura aquí

# Lista de archivos de predicciones para cada algoritmo
archivos_predicciones = [
    r"C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_predicciones_K-Nearest NeighborsM1.xlsx",
    r"C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_predicciones_Random ForestM1.xlsx",
    r"C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_predicciones_Decision TreeM1.xlsx"
]

# Colores para cada algoritmo
colors = ['darkorange', 'blue', 'green']
model_names = ['K-Nearest Neighbors', 'Random Forest', 'Decision Tree']

# Crear una lista para almacenar los resultados en formato expandido
resultados_expandido = []

# Calcular y graficar la curva ROC para cada algoritmo
for idx, archivo in enumerate(archivos_predicciones):
    # Cargar las predicciones del algoritmo actual
    predicciones = pd.read_excel(archivo)
    
    # Verificar las columnas disponibles en el archivo
    print(f"Columnas disponibles en {model_names[idx]}: {predicciones.columns}")
    
    # Extraer las predicciones del modelo (asegúrate de que la columna es correcta)
    if 'Curso_recomendado_probabilidad' in predicciones.columns:
        y_scores = predicciones['Curso_recomendado_probabilidad']  # Probabilidades si están disponibles
    elif 'Curso_recomendado_prediccion' in predicciones.columns:
        y_scores = predicciones['Curso_recomendado_prediccion']  # Predicciones de clase
    else:
        raise ValueError(f"No se encontró una columna válida de predicción en {model_names[idx]}")
    
    # Binarizar las predicciones (esto puede variar según cómo estén formateadas tus predicciones)
    y_scores_bin = label_binarize(y_scores, classes=unique_classes)

    # Imprimir rango de valores de las puntuaciones para depuración
    print(f"Modelo: {model_names[idx]}, Rango de scores: {y_scores.min()} - {y_scores.max()}")

    # Inicializar listas para almacenar los valores de TPR y FPR
    all_fpr = np.linspace(0, 1, 100)  # Valores de FPR para interpolación
    mean_tpr = 0.0  # Tasa de verdaderos positivos acumulada

    # Calcular la curva ROC para cada clase y acumular TPR
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores_bin[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)  # Interpolar TPR en función de FPR

    # Promediar la TPR y calcular el AUC promedio
    mean_tpr /= n_classes
    mean_auc = auc(all_fpr, mean_tpr)

    # Guardar resultados expandidos en la lista (una fila por punto en la curva ROC)
    for fpr_value, tpr_value in zip(all_fpr, mean_tpr):
        resultados_expandido.append({
            'Modelo': model_names[idx],
            'FPR': fpr_value,
            'TPR': tpr_value,
            'AUC': mean_auc
        })

    # Graficar la curva ROC promediada para el algoritmo actual
    plt.plot(all_fpr, mean_tpr, color=colors[idx], lw=2,
             label='Curva ROC {0} (AUC = {1:0.2f})'.format(model_names[idx], mean_auc))

# Graficar la línea diagonal (clasificador aleatorio)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Clasificador aleatorio')

# Configurar el gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para Algoritmos (Resultados Reales vs Resultados Recomendación), Escenario 1')

plt.legend(loc='lower right')  
plt.savefig(r'C:\xampp\htdocs\tesis2\imagenes\modelo1\prueba\curva_roc_realvspredic_M1.png', bbox_inches='tight')

# Mostrar el gráfico
plt.show()

# Crear un DataFrame con los resultados expandidos
df_resultados_expandido = pd.DataFrame(resultados_expandido)

# Guardar los resultados expandidos en un archivo Excel
df_resultados_expandido.to_excel(r'C:\xampp\htdocs\tesis2\dataframe\modelo1\prueba\resultados_curvas_roc_M1.xlsx', index=False)

print("Los resultados se han guardado en resultados_curvas_roc_M1.xlsx")