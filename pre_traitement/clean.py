import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def detect_target_column(data):
    target_column = data.columns[-1]  # On suppose que la dernière colonne est la cible
    target_data = data[target_column]

    # Détection de la classification
    if target_data.dtype == 'object' or len(target_data.unique()) < 10:
        X = data.drop(columns=[target_column])
        y = target_data
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        try:
            model = DecisionTreeClassifier()
            model.fit(X, y)
            predictions = model.predict(X)
            acc = accuracy_score(y, predictions)
            if acc > 0.5:  # Si la précision est suffisante
                return target_column, "classification"
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle de classification : {e}")
    
    # Détection de la régression
    if pd.api.types.is_numeric_dtype(target_data):
        X = data.drop(columns=[target_column])
        y = target_data

        try:
            model = DecisionTreeRegressor()  # Utilisation de DecisionTreeRegressor pour la régression
            model.fit(X, y)
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            if mse < 10:  # Si l'erreur est assez faible
                return target_column, "regression"
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle de régression : {e}")

    # Si ce n'est ni classification ni régression, c'est probablement un problème de clustering
    # (donc pas de cible à prédire)
    return target_column, "clustering"  # On retourne 'clustering' si aucune autre détection


def clean_dataset(data):
    data_cleaned = data.drop_duplicates()

    numeric_columns = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data_cleaned.select_dtypes(include=['object', 'category']).columns
    boolean_columns = data_cleaned.select_dtypes(include=['bool']).columns

    for col in numeric_columns:
        if data_cleaned[col].isnull().sum() > 0:
            mean_value = data_cleaned[col].mean()
            data_cleaned[col].fillna(mean_value, inplace=True)

    for col in categorical_columns:
        if data_cleaned[col].isnull().sum() > 0:
            data_cleaned = data_cleaned[data_cleaned[col].notnull()]

    
        if data_cleaned[col].isnull().sum() > 0:
            mode_value = data_cleaned[col].mode()[0]  # Mode pour les colonnes binaires
            data_cleaned[col].fillna(mode_value, inplace=True)

   
    data_cleaned.reset_index(drop=True, inplace=True)

    return data_cleaned
