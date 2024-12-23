import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score, mean_squared_error
from pre_traitement.clean import detect_target_column

# Fonction pour encoder les données catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

# Fonction pour appliquer Random Forest et évaluer les résultats
def random_forest_model(X_train, y_train, X_test, y_test, problem_type, n_estimators=100):
    if problem_type == "classification":
        # Appliquer Random Forest pour classification
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Évaluation de la classification
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"Accuracy (Train) : {train_accuracy:.2f}")
        print(f"Accuracy (Test) : {test_accuracy:.2f}")
        return model, train_accuracy, test_accuracy
    
    elif problem_type == "regression":
        # Appliquer Random Forest pour régression
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Évaluation de la régression
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        print(f"Mean Squared Error (Train) : {train_mse:.2f}")
        print(f"Mean Squared Error (Test) : {test_mse:.2f}")
        return model, train_mse, test_mse
    
    else:
        print("Type de problème inconnu, aucun modèle entraîné.")
        return None, None

# Fonction principale pour nettoyer et prétraiter les données
def preprocess_data(data):
    # Nettoyage des données (encodage des valeurs catégorielles)
    data = encode_data(data)

    # Détecter la colonne cible et le type de problème
    target_column, problem_type = detect_target_column(data)
    
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data
        y = None

    return X, y, target_column, problem_type

###############################################################################
def RF(data, n_estimators=100):
    # Prétraiter les données
    X, y, target_column, problem_type = preprocess_data(data)

    if X is None or y is None or problem_type is None:
        print("Les données ne sont pas adaptées pour le modèle Random Forest.")
        return

    print(f"Colonne cible détectée : {target_column}")
    print(f"Type de problème détecté : {problem_type}")

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Exécuter le modèle Random Forest
    print("Exécution du modèle Random Forest...")
    model, train_metric, test_metric = random_forest_model(X_train, y_train, X_test, y_test, problem_type, n_estimators)

    return model, train_metric, test_metric