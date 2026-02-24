from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and train model globally to persist state
try:
    df = pd.read_csv("heart.csv")
    
    categorical_columns = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "rest_ecg",
        "exercise_induced_angina",
        "slope",
        "vessels_colored_by_flourosopy",
        "thalassemia"
    ]
    
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model trained successfully backend ✅")

except Exception as e:
    print(f"Error training model: {e}")
    model = None
    le_dict = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained'}), 500
        
    try:
        data = request.json
        
        # Prepare input data structure matching columns
        features = [
            data['age'],
            data['sex'],
            data['chest_pain_type'],
            data['resting_blood_pressure'],
            data['cholesterol'],
            data['fasting_blood_sugar'],
            data['rest_ecg'],
            data['max_heart_rate'],
            data['exercise_induced_angina'],
            data['oldpeak'],
            data['slope'],
            data['vessels_colored_by_flourosopy'],
            data['thalassemia']
        ]
        
        input_df = pd.DataFrame([features], columns=X.columns)
        
        # Encode categorical columns
        for col in categorical_columns:
            if col in le_dict:
                # Handle unseen labels or exact matching
                # For robustness, we might want to handle this better, but for now simple transform
                # Assuming frontend sends valid categories matching training data
                try:
                    input_df[col] = le_dict[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({'error': f"Invalid value for {col}: {input_df[col].values[0]}"}), 400
                    
        prediction = model.predict(input_df)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        
        return jsonify({'prediction': result, 'class': int(prediction[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
