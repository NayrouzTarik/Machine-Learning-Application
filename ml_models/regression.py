import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from pre_traitement.clean import detect_target_column

def encode_data(data):
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])
    return data

def preprocess_data(data):
    data = encode_data(data)
    target_column, problem_type = detect_target_column(data)
    
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data
        y = None

    return X, y, target_column, problem_type

def regression_model(data):
    try:
        X, y, target_column, problem_type = preprocess_data(data)
        
        if X is None or y is None:
            return {"error": "Invalid data format", "r2": 0.0}
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if problem_type == "regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
        else:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                "accuracy": float(accuracy_score(y_test, y_pred))
            }
            
    except Exception as e:
        print(f"Error in regression_model: {e}")
        return {"error": str(e), "r2": 0.0}