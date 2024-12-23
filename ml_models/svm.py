import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from pre_traitement.clean import detect_target_column

# Fonction pour encoder les données catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

# Fonction pour normaliser les données
def normalize_data(data):
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

# Fonction pour appliquer SVM avec tous les noyaux
def svm_all_kernels(X_train, y_train, X_test, y_test, problem_type):
    kernels = ["linear", "poly", "rbf", "sigmoid"]  # Liste des noyaux SVM
    
    results = []  # Pour stocker les résultats pour chaque noyau

    for kernel in kernels:
        print(f"\n--- SVM avec le noyau : {kernel} ---")

        if problem_type == "classification":
            # SVM pour classification
            model = SVC(kernel=kernel, random_state=42)
        elif problem_type == "regression":
            # SVM pour régression
            model = SVR(kernel=kernel)
        else:
            print("Type de problème inconnu.")
            return None

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Évaluation
        if problem_type == "classification":
            train_metric = accuracy_score(y_train, y_pred_train)
            test_metric = accuracy_score(y_test, y_pred_test)
            print(f"Accuracy (Train) : {train_metric:.2f}")
            print(f"Accuracy (Test) : {test_metric:.2f}")
        else:
            train_metric = mean_squared_error(y_train, y_pred_train)
            test_metric = mean_squared_error(y_test, y_pred_test)
            print(f"Mean Squared Error (Train) : {train_metric:.2f}")
            print(f"Mean Squared Error (Test) : {test_metric:.2f}")
        
        # Ajouter les résultats
        results.append({"kernel": kernel, "train_metric": train_metric, "test_metric": test_metric})

    return results

# Fonction principale pour nettoyer et prétraiter les données
def preprocess_data(data):
    # Nettoyage des données (encodage des valeurs catégorielles)
    data = encode_data(data)

    # Normalisation des données
    data = normalize_data(data)

    # Détecter la colonne cible et le type de problème
    target_column, problem_type = detect_target_column(data)
    
    if not target_column or not problem_type:
        print("Impossible de détecter une colonne cible ou un type de problème.")
        return None, None, None, None

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y, target_column, problem_type

def SVM(data):
    # Prétraiter les données
    X, y, target_column, problem_type = preprocess_data(data)

    if X is None or y is None or problem_type is None:
        return None, {"error": "Data not suitable for SVM"}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Run model with all kernels
    results = svm_all_kernels(X_train, y_train, X_test, y_test, problem_type)
    
    # Return best model results
    best_result = max(results, key=lambda x: x['test_metric'])
    return best_result['kernel'], {
        'train_metric': best_result['train_metric'],
        'test_metric': best_result['test_metric']
    }
