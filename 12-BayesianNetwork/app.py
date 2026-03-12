# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

st.title("Heart Disease Diagnosis using Bayesian Network")

# Upload dataset
uploaded_file = st.file_uploader("Upload Heart Disease Dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Discretize continuous features
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    data[continuous_features] = discretizer.fit_transform(data[continuous_features])

    # Define Bayesian Network structure
    model = DiscreteBayesianNetwork([
        ("age", "target"),
        ("sex", "target"),
        ("cp", "target"),
        ("trestbps", "target"),
        ("chol", "target"),
        ("fbs", "target"),
        ("restecg", "target"),
        ("thalach", "target"),
        ("exang", "target"),
        ("oldpeak", "target"),
        ("slope", "target"),
        ("ca", "target"),
        ("thal", "target")
    ])

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Train Bayesian Network
    model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu')

    # Show training outputs
    if st.button("Show Training Outputs"):
        st.subheader("Training Complete!")

        # Inference
        infer = VariableElimination(model)
        y_true = test_data['target'].values
        y_pred = []

        for _, row in test_data.iterrows():
            evidence = row.drop('target').to_dict()
            q = infer.map_query(variables=['target'], evidence=evidence)
            y_pred.append(q['target'])

        # Evaluation Metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)

        st.write(f"**Accuracy:** {acc:.4f}")
        st.write("**Confusion Matrix:**")
        st.write(cm)
        st.write("**Classification Report:**")
        st.text(cr)

    # -------------------------
    # Predict for manual input
    # -------------------------
    st.subheader("Predict Heart Disease for New Patient")

    with st.form("patient_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        sex = st.selectbox("Sex", [0, 1], help="0 = female, 1 = male")
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
        ca = st.selectbox("Number of major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Discretize continuous inputs same as training
            input_data = pd.DataFrame([{
                'age': discretizer.transform([[age,0,0,0,0]])[0][0],  # only first column matters
                'trestbps': discretizer.transform([[0, trestbps,0,0,0]])[0][1],
                'chol': discretizer.transform([[0,0,chol,0,0]])[0][2],
                'thalach': discretizer.transform([[0,0,0,thalach,0]])[0][3],
                'oldpeak': discretizer.transform([[0,0,0,0,oldpeak]])[0][4],
                'sex': sex,
                'cp': cp,
                'fbs': fbs,
                'restecg': restecg,
                'exang': exang,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }])

            # Perform inference
            infer = VariableElimination(model)
            evidence = input_data.iloc[0].to_dict()
            q = infer.map_query(variables=['target'], evidence=evidence)
            result = q['target']

            st.success(f"Predicted Heart Disease Target: {result}  (0 = No, 1 = Yes)")