from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==========================================================
# GLOBAL VARIABLES
# ==========================================================

model = None
le_dict = {}
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

# ==========================================================
# TRAIN MODEL ON STARTUP
# ==========================================================

try:
    df = pd.read_csv("heart.csv")

    # Encode categorical columns
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

    print("✅ Model trained successfully")

except Exception as e:
    print(f"❌ Error training model: {e}")
    model = None


# ==========================================================
# ROUTES
# ==========================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({'error': 'Model not trained'}), 500

    try:
        data = request.json

        # Create dataframe with correct column order
        input_data = pd.DataFrame([{
            "age": data["age"],
            "sex": data["sex"],
            "chest_pain_type": data["chest_pain_type"],
            "resting_blood_pressure": data["resting_blood_pressure"],
            "cholesterol": data["cholesterol"],
            "fasting_blood_sugar": data["fasting_blood_sugar"],
            "rest_ecg": data["rest_ecg"],
            "max_heart_rate": data["max_heart_rate"],
            "exercise_induced_angina": data["exercise_induced_angina"],
            "oldpeak": data["oldpeak"],
            "slope": data["slope"],
            "vessels_colored_by_flourosopy": data["vessels_colored_by_flourosopy"],
            "thalassemia": data["thalassemia"]
        }])

        # Encode categorical inputs safely
        for col in categorical_columns:
            if col in le_dict:
                if input_data[col].values[0] not in le_dict[col].classes_:
                    return jsonify({
                        "error": f"Invalid value for {col}: {input_data[col].values[0]}"
                    }), 400
                input_data[col] = le_dict[col].transform(input_data[col])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100

        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "class": int(prediction[0]),
            "risk_probability_percent": round(probability, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ==========================================================
# RUN APP (FIXED THREAD ISSUE)
# ==========================================================

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)