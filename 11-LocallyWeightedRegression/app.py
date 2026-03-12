import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Locally Weighted Regression (LWR) - Fish Weight Prediction")
st.write("Upload Fish.csv dataset and select numeric columns for regression.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Fish.csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Automatically list numeric columns
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Dataset must have at least two numeric columns.")
    else:
        x_col = st.selectbox("Independent variable (X)", numeric_cols, index=numeric_cols.index("Length3"))
        y_col = st.selectbox("Dependent variable (y)", numeric_cols, index=numeric_cols.index("Weight"))

        X_raw = data[x_col].values
        y = data[y_col].values

        # Add bias term
        X = np.vstack((np.ones(len(X_raw)), X_raw)).T

        # Tau slider
        tau = st.slider("Select Bandwidth (Tau)", 0.1, 10.0, 1.0, step=0.1)

        # LWR function
        def lwlr(query_pt, X, y, tau):
            m = X.shape[0]
            W = np.eye(m)
            for i in range(m):
                diff = query_pt - X[i]
                W[i, i] = np.exp(diff @ diff.T / (-2 * tau**2))
            theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
            return query_pt @ theta

        # Prediction points
        x_vals = np.linspace(X_raw.min(), X_raw.max(), 200)
        preds = [lwlr(np.array([1, x]), X, y, tau) for x in x_vals]

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(X_raw, y, label="Actual Data")
        ax.plot(x_vals, preds, color="red", label="LWR Fit")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("LWR Fit - Fish Weight Prediction")
        ax.legend()
        st.pyplot(fig)

        # Training metrics
        if st.button("Show Metrics"):
            train_preds = np.array([lwlr(np.array([1, xi]), X, y, tau) for xi in X_raw])
            mse = np.mean((y - train_preds)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - train_preds))
            ss_tot = np.sum((y - np.mean(y))**2)
            ss_res = np.sum((y - train_preds)**2)
            r2 = 1 - (ss_res / ss_tot)

            st.subheader("Evaluation Metrics")
            st.write("**MSE:**", round(mse,3))
            st.write("**RMSE:**", round(rmse,3))
            st.write("**MAE:**", round(mae,3))
            st.write("**R² Score:**", round(r2,3))
else:
    st.info("Upload a CSV file to begin.")