from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from pre_traitement.clean import detect_target_column

def encode_data(data):
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])
    return data

def naive_bayes_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report

def preprocess_data(data):
    # Encode categorical data
    data = encode_data(data)

    # Get target column and problem type
    target_column, problem_type = detect_target_column(data)
    
    if target_column is None or problem_type is None:
        return None, None, None, None

    if problem_type != "classification":
        return None, None, None, None

    # Prepare X and y
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y, target_column, problem_type

def naive_bayes(data):
    try:
        # Preprocess data
        X, y, target_column, problem_type = preprocess_data(data)
        
        if X is None or y is None:
            return None, 0, "Error: Invalid data format"

        # Run model
        model, accuracy, report = naive_bayes_model(X, y)
        
        return model, accuracy, report
        
    except Exception as e:
        print(f"Error in naive_bayes: {e}")
        return None, 0, str(e)