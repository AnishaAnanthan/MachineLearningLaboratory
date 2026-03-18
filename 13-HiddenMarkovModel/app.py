import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="HMM Predictor", layout="centered")

st.title("🌦 Hidden Markov Model - Weather Predictor")
st.write("Model learns from dataset ")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("hmm_weather.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data)

# -----------------------------
# States & Observations
# -----------------------------
states = list(data['HiddenState'].unique())
observations = list(data['Observation'].unique())

state_map = {s: i for i, s in enumerate(states)}
obs_map = {o: i for i, o in enumerate(observations)}

# -----------------------------
# Initial Probability π
# -----------------------------
pi = np.zeros(len(states))

for s in data['HiddenState']:
    pi[state_map[s]] += 1

pi = pi / np.sum(pi)

# -----------------------------
# Transition Matrix A
# -----------------------------
A = np.zeros((len(states), len(states)))

for i in range(len(data)-1):
    curr = state_map[data['HiddenState'][i]]
    next_ = state_map[data['HiddenState'][i+1]]
    A[curr][next_] += 1

# Normalize
for i in range(len(states)):
    total = np.sum(A[i])
    if total != 0:
        A[i] /= total

# -----------------------------
# Emission Matrix B
# -----------------------------
B = np.zeros((len(states), len(observations)))

for i in range(len(data)):
    s = state_map[data['HiddenState'][i]]
    o = obs_map[data['Observation'][i]]
    B[s][o] += 1

# Normalize
for i in range(len(states)):
    total = np.sum(B[i])
    if total != 0:
        B[i] /= total

# -----------------------------
# Show Matrices
# -----------------------------
st.subheader("📌 Initial Probability (π)")
st.write(pd.Series(pi, index=states))

st.subheader("📌 Transition Matrix (A)")
st.write(pd.DataFrame(A, index=states, columns=states))

st.subheader("📌 Emission Matrix (B)")
st.write(pd.DataFrame(B, index=states, columns=observations))

# -----------------------------
# Viterbi Algorithm
# -----------------------------
def viterbi(obs_seq):
    T = len(obs_seq)
    N = len(states)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    delta[0] = pi * B[:, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            prob = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(prob)
            delta[t, j] = np.max(prob) * B[j, obs_seq[t]]

    # Backtracking
    path = []
    last = np.argmax(delta[T-1])
    path.append(last)

    for t in range(T-1, 0, -1):
        last = psi[t][last]
        path.insert(0, last)

    return [states[i] for i in path]

# -----------------------------
# User Input
# -----------------------------
st.subheader("🧪 Enter Observations")

user_input = st.text_input(
    "Enter sequence (comma separated):",
    "Umbrella, NoUmbrella, Umbrella"
)

if st.button("🔍 Predict Hidden States"):
    try:
        obs_list = [x.strip() for x in user_input.split(",")]
        obs_seq = [obs_map[o] for o in obs_list]

        result = viterbi(obs_seq)

        st.success("✅ Predicted Hidden States:")
        st.write(result)

    except:
        st.error("❌ Invalid input! Use only available observations")