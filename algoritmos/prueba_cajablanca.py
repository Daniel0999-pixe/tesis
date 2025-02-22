import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Función para cargar y preprocesar datos
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    label_encoder_curso = LabelEncoder()
    data['Curso_recomendado'] = label_encoder_curso.fit_transform(data['Curso_recomendado'])
    
    # Definir características y variable objetivo
    X = data[['resultado_tic', 'resultado_dic', 'resultado_ped']]
    y = data['Curso_recomendado']
    
    # Estandarizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, label_encoder_curso

# Función para entrenar modelos
def train_models(X_train, y_train):
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    trained_models = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
    
    return trained_models

# Función para hacer predicciones
def make_predictions(models, X_test):
    predictions = {}
    
    for model_name, model in models.items():
        preds = model.predict(X_test)
        predictions[model_name] = preds
    
    return predictions

# Función para evaluar modelos
def evaluate_models(predictions, y_test, label_encoder):
    results = {}
    
    for model_name, preds in predictions.items():
        report = classification_report(y_test, preds, output_dict=True)
        results[model_name] = {
            'accuracy': report['accuracy'],
            'f1_score': report['weighted avg']['f1-score'],
            'recall': report['weighted avg']['recall']
        }
    
    return results
# Mostrar resultados
    for model_name, metrics in results.items():
        print(f"Modelo: {model_name}, Precisión: {metrics['accuracy']:.2f}, F1 Score: {metrics['f1_score']:.2f}, Recall: {metrics['recall']:.2f}")


# Simulación de ejecución con datos de prueba "quemados"
def main():
    # Simular la carga de datos (aquí deberías usar la ruta correcta)
    file_path = r'C:\xampp\htdocs\tesis2\dataframe\datosbase\datosentreno2.xlsx'
    
    # Cargar y preprocesar datos
    X_scaled, y, label_encoder_curso = load_and_preprocess_data(file_path)

    # Dividir los datos en conjuntos de entrenamiento y prueba (estratificados)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Entrenar modelos
    models = train_models(X_train, y_train)

    # Hacer predicciones
    predictions = make_predictions(models, X_test)

    # Evaluar modelos
    results = evaluate_models(predictions, y_test, label_encoder_curso)

    # Mostrar resultados
    for model_name, metrics in results.items():
        print(f"Modelo: {model_name}, Precisión: {metrics['accuracy']:.2f}, F1 Score: {metrics['f1_score']:.2f}, Recall: {metrics['recall']:.2f}")

if __name__ == "__main__":
    main()
