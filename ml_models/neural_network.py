import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from pre_traitement.clean import detect_target_column
# Fonction pour encoder les données catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

# Fonction pour appliquer les réseaux de neurones (MLP) et évaluer les résultats
def mlp_model(X_train, y_train, X_test, y_test, problem_type, hidden_layer_sizes=(100,)):
    if problem_type == "classification":
        # Appliquer MLP pour classification
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Évaluation de la classification
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\nMLP - Hidden layers : {hidden_layer_sizes}")
        print(f"Accuracy (Train) : {train_accuracy:.2f}")
        print(f"Accuracy (Test) : {test_accuracy:.2f}")
        return model, train_accuracy, test_accuracy
    
    elif problem_type == "regression":
        # Appliquer MLP pour régression
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Évaluation de la régression
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        print(f"\nMLP - Hidden layers : {hidden_layer_sizes}")
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
def MLP_model(data, hidden_layer_sizes=(100,)):
    # Prétraiter les données
    X, y, target_column, problem_type = preprocess_data(data)
    
    # Diviser les données en ensembles d'entraînement et de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Appliquer le modèle de MLP en fonction du type de problème
    model, *metrics = mlp_model(X_train, y_train, X_test, y_test, problem_type, hidden_layer_sizes)

    # Retourner le modèle et les métriques pour l'affichage ou le traitement ultérieur
    return model, metrics