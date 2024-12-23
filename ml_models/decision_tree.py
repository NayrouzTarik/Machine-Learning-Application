import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pre_traitement.clean import detect_target_column


# Fonction pour encoder les données catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

# Fonction pour équilibrer les classes en cas de déséquilibre (SMOTE)
def balance_data(X, y, data):
    # Détecter la colonne cible et le type de problème
    target_column, problem_type = detect_target_column(data)

    if problem_type == "classification":
        # Appliquer SMOTE pour équilibrer les données
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=4)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("Les données de classification ont été équilibrées.")
        return X_balanced, y_balanced
    else:
        print("Le problème n'est pas une classification. Aucun équilibrage effectué.")
        return X, y

# Fonction pour détecter et appliquer le modèle de décision
def decision_tree_model(X_train, X_test, y_train, y_test, problem_type):
    if problem_type == "classification":
        # Arbre de décision pour classification
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        predictions = model.predict(X_test)
        
        # Évaluation
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        confusion = confusion_matrix(y_test, predictions)
        
        # Rapport de classification
        report = classification_report(y_test, predictions)
        
        print(f"Précision : {accuracy * 100:.2f}%")
        print(f"Précision (Precision) : {precision:.2f}")
        print(f"Rappel (Recall) : {recall:.2f}")
        print(f"F1-Score : {f1:.2f}")
        print("Matrice de confusion :")
        print(confusion)
        print("\nRapport de classification :")
        print(report)
        
        return model, accuracy, precision, recall, f1
    
    elif problem_type == "regression":
        # Arbre de décision pour régression
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        predictions = model.predict(X_test)
        
        # Évaluation
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
        print(f"Erreur absolue moyenne (MAE) : {mae:.2f}")
        print(f"Coefficient de détermination (R²) : {r2:.2f}")
        
        return model, mse, mae, r2

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
    X, y = balance_data(X, y, data)
    return X, y, target_column, problem_type

###############################################################################FCT########################################

def DT(data):
    # Prétraiter les données
    X, y, target_column, problem_type = preprocess_data(data)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Appliquer le modèle de décision en fonction du type de problème
    model, *metrics = decision_tree_model(X_train, X_test, y_train, y_test, problem_type)

    # Retourner le modèle et les métriques pour l'affichage ou le traitement ultérieur
    return model, metrics
