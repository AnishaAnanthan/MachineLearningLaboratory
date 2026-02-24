import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("heart.csv")

# ===============================
# 2. ENCODE CATEGORICAL COLUMNS
# ===============================
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

# ===============================
# 3. SPLIT DATA
# ===============================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. TRAIN MODEL (ANTI-OVERFITTING)
# ===============================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 5. MODEL EVALUATION
# ===============================
y_pred = model.predict(X_test)

print("\nModel Training Complete ✅")
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ---- SAVE CONFUSION MATRIX ----
cm = confusion_matrix(y_test, y_pred)

plt.figure()
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Confusion Matrix saved as confusion_matrix.png")

# ===============================
# 6. VISUALIZE & SAVE TWO TREES
# ===============================

# ---- TREE 1 ----
plt.figure(figsize=(20,10))
plot_tree(model.estimators_[0],
          feature_names=X.columns,
          class_names=["No Disease", "Disease"],
          filled=True,
          max_depth=3)

plt.title("Decision Tree 1")
plt.savefig("decision_tree_1.png", dpi=300, bbox_inches="tight")
plt.close()

print("Decision Tree 1 saved as decision_tree_1.png")

# ---- TREE 2 ----
plt.figure(figsize=(20,10))
plot_tree(model.estimators_[1],
          feature_names=X.columns,
          class_names=["No Disease", "Disease"],
          filled=True,
          max_depth=3)

plt.title("Decision Tree 2")
plt.savefig("decision_tree_2.png", dpi=300, bbox_inches="tight")
plt.close()

print("Decision Tree 2 saved as decision_tree_2.png")

# ===============================
# 7. TAKE USER INPUT
# ===============================
print("\nEnter Patient Details:\n")

age = int(input("Age: "))
sex = input("Sex (Male/Female): ")
chest_pain_type = input("Chest Pain (Typical angina/Atypical angina/Non-anginal pain/Asymptomatic): ")
resting_blood_pressure = int(input("Resting Blood Pressure: "))
cholesterol = int(input("Cholesterol: "))

print("\nFasting Blood Sugar:")
print("1. Lower than 120 mg/ml")
print("2. Greater than 120 mg/ml")
fbs_choice = int(input("Enter choice (1 or 2): "))

fasting_blood_sugar = "Lower than 120 mg/ml" if fbs_choice == 1 else "Greater than 120 mg/ml"

rest_ecg = input("Rest ECG (Normal/ST-T wave abnormality/Left ventricular hypertrophy): ")
max_heart_rate = int(input("Max Heart Rate: "))
exercise_induced_angina = input("Exercise Induced Angina (Yes/No): ")
oldpeak = float(input("Oldpeak: "))
slope = input("Slope (Upsloping/Flat/Downsloping): ")
vessels_colored_by_flourosopy = input("Vessels (Zero/One/Two/Three): ")
thalassemia = input("Thalassemia (Normal/Fixed Defect/Reversable Defect): ")

# ===============================
# 8. PREPARE INPUT DATA
# ===============================
input_data = pd.DataFrame([[ 
    age,
    sex,
    chest_pain_type,
    resting_blood_pressure,
    cholesterol,
    fasting_blood_sugar,
    rest_ecg,
    max_heart_rate,
    exercise_induced_angina,
    oldpeak,
    slope,
    vessels_colored_by_flourosopy,
    thalassemia
]], columns=X.columns)

# Encode categorical input safely
for col in categorical_columns:
    try:
        input_data[col] = le_dict[col].transform(input_data[col])
    except:
        print(f"Invalid input for {col}. Please match dataset values exactly.")
        exit()

# ===============================
# 9. PREDICT
# ===============================
prediction = model.predict(input_data)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("⚠ Patient HAS Heart Disease")
else:
    print("✅ Patient does NOT have Heart Disease")