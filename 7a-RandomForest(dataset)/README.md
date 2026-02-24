# Heart Disease Prediction System

A machine learning-powered web application that predicts the likelihood of heart disease based on medical parameters. The system uses a **Random Forest Classifier** trained on the **Heart Disease Dataset UCI** (sourced from **Kaggle**) to provide instant risk assessments through a modern, user-friendly interface.

##  Dataset

The model is trained on the **Heart Disease Dataset UCI**, which is publicly available on **Kaggle**. This dataset contains patient records with various medical attributes used to predict the presence of heart disease.


## How the Prediction Works

The core of this application is a **Random Forest Classifier**, a robust ensemble learning method. Here's a breakdown of the process:

1.  **Data Preprocessing**:
    -   The raw dataset (`heart.csv`) contains categorical variables like "Sex", "Chest Pain Type", and "Thalassemia".
    -   These are converted into numerical keys using **Label Encoding** (e.g., "Male" -> 1, "Female" -> 0) so the model can understand them.
    -   The dataset is split into training and testing sets to evaluate performance.

2.  **Model Training (Random Forest)**:
    -   The application initializes a Random Forest model with **100 decision trees** (`n_estimators=100`).
    -   **Ensemble Learning**: Instead of relying on a single decision tree, the Random Forest creates multiple trees, each trained on a random subset of the data and features.
    -   **Voting Mechanism**: When you submit patient data, each of the 100 trees makes its own prediction (Disease or No Disease). The Random Forest takes a majority vote to determine the final result.
    -   **Anti-Overfitting**: The model is configured with constraints (e.g., `max_depth=5`) to ensure it generalizes well to new data rather than just memorizing the training set.

3.  **Real-Time Inference**:
    -   When you click "Analyze Risk", your input data is sent to the Flask backend.
    -   The backend processes the input, encodes it using the same encoders as the training phase, and feeds it into the trained model.
    -   The model returns a prediction (0 or 1), which the UI displays as "Low Risk" or "High Risk".

##  Features

-   **Real-time Prediction**: Instant analysis of patient data using a trained ML model.
-   **Interactive UI**: Clean, responsive web interface built with HTML5, CSS3, and JavaScript.
-   **Robust Backend**: Flask-based API that handles data processing and model inference.
-   **Visual Feedback**: Clear, color-coded results indicating risk levels.

##  Prerequisites

Ensure you have **Python 3.7+** installed on your system.

##  Installation & Setup

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2.  **Install dependencies**:
    Open your terminal/command prompt in the project folder and run:
    ```bash
    pip install flask pandas numpy scikit-learn matplotlib
    ```

##  Steps to Run the Application

Follow these simple steps to launch the prediction system:

1.  **Start the Server**:
    Run the following command in your terminal:
    ```bash
    python app.py
    ```
    You should see output similar to:
    > * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

2.  **Open the Application**:
    Open your web browser (Chrome, Edge, Firefox, etc.) and go to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

3.  **Make a Prediction**:
    -   **Fill out the form**: Enter the patient's Age, Sex, Chest Pain Type, Blood Pressure, etc.
    -   **Submit**: Click the **"Analyze Risk"** button.
    -   **View Results**: The application will instantly display whether the risk is **Low** (No Disease) or **High** (Disease Detected).

##  Project Structure

-   `app.py`: The main Flask application file. Handles server logic, model training, and API requests.
-   `templates/index.html`: The HTML file for the user interface.
-   `static/style.css`: CSS file for styling the application.
-   `static/script.js`: JavaScript file for handling form submissions and updating the UI.
-   `heart.csv`: The dataset used to train the machine learning model.
