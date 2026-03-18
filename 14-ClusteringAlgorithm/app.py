import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="K-Means Clustering", layout="centered")

# -----------------------------------
# Title
# -----------------------------------
st.title("🧠 K-Means Clustering (Kaggle Dataset)")
st.write("Clustering Mall Customers using K-Means Algorithm")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("mall_customers.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# OPTIONAL: Encode Gender
# -----------------------------------
if "Gender" in df.columns:
    df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

# -----------------------------------
# Select Numeric Features Only
# -----------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remove ID column (not useful for clustering)
if "CustomerID" in numeric_cols:
    numeric_cols.remove("CustomerID")

st.subheader("🎯 Select Features")

features = st.multiselect(
    "Choose exactly 2 features",
    numeric_cols,
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

if len(features) != 2:
    st.warning("⚠️ Please select exactly 2 numeric features")
    st.stop()

X = df[features]

# -----------------------------------
# Select K
# -----------------------------------
st.subheader("🔢 Select Number of Clusters")

k = st.slider("K value", 1, 10, 3)

# -----------------------------------
# Apply K-Means
# -----------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)

df["Cluster"] = kmeans.labels_
centroids = kmeans.cluster_centers_

# -----------------------------------
# Output
# -----------------------------------
st.subheader("📌 Clustered Data")
st.dataframe(df.head())

st.subheader("📍 Centroids")
centroid_df = pd.DataFrame(centroids, columns=features)
st.write(centroid_df)

# -----------------------------------
# Visualization
# -----------------------------------
st.subheader("📈 Visualization")

fig, ax = plt.subplots()

scatter = ax.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=df["Cluster"]
)

ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200
)

ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_title("K-Means Clustering")

st.pyplot(fig)