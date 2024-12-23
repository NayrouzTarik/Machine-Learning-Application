import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from pre_traitement.clean import detect_target_column

def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

# Fonction pour appliquer KMeans et évaluer les résultats
def kmeans_model(X, y, problem_type, n_clusters=3):
    if problem_type == "classification":
        print("K-Means n'est pas adapté aux problèmes de classification.")
        return None, None
    
    elif problem_type == "clustering":
        # Appliquer K-Means
        model = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = model.fit_predict(X)
        
        # Évaluation du clustering
        silhouette = silhouette_score(X, y_pred)
        homogeneity = homogeneity_score(y, y_pred)
        completeness = completeness_score(y, y_pred)
        v_measure = v_measure_score(y, y_pred)
        
        print(f"Silhouette Score : {silhouette:.2f}")
        print(f"Homogeneity Score : {homogeneity:.2f}")
        print(f"Completeness Score : {completeness:.2f}")
        print(f"V-Measure Score : {v_measure:.2f}")
        
        return model, silhouette, homogeneity, completeness, v_measure
    
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
def KMeans(data):
    try:
        X, y, target_column, problem_type = preprocess_data(data)
        
        if X is None:
            return None, (0.0, 0.0, 0.0, 0.0)

        # Initialize and fit KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = float(silhouette_score(X, clusters) if len(set(clusters)) > 1 else 0.0)
        homogeneity = float(homogeneity_score(y, clusters) if y is not None else 0.0)
        completeness = float(completeness_score(y, clusters) if y is not None else 0.0)
        v_measure = float(v_measure_score(y, clusters) if y is not None else 0.0)
        
        return kmeans, (silhouette, homogeneity, completeness, v_measure)
        
    except Exception as e:
        print(f"Error in KMeans: {e}")
        return None, (0.0, 0.0, 0.0, 0.0)