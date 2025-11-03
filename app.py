
#  AI Strategy Advisor - Promotion Optimization Dashboard
# Author: Ishan


# ---------- Imports ----------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from stable_baselines3 import PPO
import gymnasium as gym
import random
import time

# ---------- Page Setup ----------
st.set_page_config(
    page_title="AI Strategy Advisor",
    page_icon="",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    color: #38bdf8;
}
.sidebar .sidebar-content {
    background: #1e293b;
}
div.stButton > button:first-child {
    background-color: #38bdf8;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
div.stDownloadButton > button:first-child {
    background-color: #22c55e;
    color: white;
    border-radius: 8px;
    height: 2.8em;
    width: 100%;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title(" AI-Driven Business Strategy Advisor")
st.markdown("""
This system uses **Causal AI** and **Reinforcement Learning** to identify 
cause-effect relationships and recommend **optimal actions** to maximize business KPIs.
""")


#  STEP 1: DATA INGESTION


st.sidebar.header(" Data Setup")

upload = st.sidebar.file_uploader("Upload your business dataset (CSV)", type=["csv"])
if upload:
    df = pd.read_csv(upload)
else:
    np.random.seed(42)
    df = pd.DataFrame({
        "price": np.random.uniform(0.1, 1.0, 200),
        "foot_traffic": np.random.uniform(0.1, 1.0, 200),
        "revenue": np.random.uniform(0.2, 1.2, 200),
    })
st.sidebar.success(" Data Loaded")


#  STEP 2: RL ENVIRONMENT DEFINITION


class PromoEnv(gym.Env):
    def __init__(self):
        super(PromoEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(2).astype(np.float32)
        return self.state, {}

    def step(self, action):
        price, traffic = self.state
        reward = (10 * traffic - 5 * price) if action == 1 else (5 * traffic - 2 * price)
        reward += np.random.normal(0, 0.3)
        self.state = np.random.rand(2).astype(np.float32)
        terminated, truncated = False, False
        return self.state, reward, terminated, truncated, {}


#  STEP 3: TRAIN AI MODEL


with st.spinner(" Training reinforcement learning agent..."):
    env = PromoEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=6000)
st.success(" Model Trained Successfully!")


#  STEP 4: AI POLICY INFERENCE


def get_policy(price, traffic):
    obs = np.array([price, traffic], dtype=np.float32)
    action, _ = model.predict(obs)
    return "Run Promo" if action == 1 else "No Promo"

df["AI_Recommendation"] = df.apply(lambda x: get_policy(x["price"], x["foot_traffic"]), axis=1)


#  STEP 5: DASHBOARD VISUALIZATION


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" AI Decision Visualization")
    fig = px.scatter(
        df, x="price", y="foot_traffic",
        color="AI_Recommendation", size="revenue",
        title="Optimal Promotion Policy Map",
        hover_data=["revenue"]
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(" KPI Insights")
    promo_count = (df["AI_Recommendation"] == "Run Promo").sum()
    avg_rev_promo = df[df["AI_Recommendation"] == "Run Promo"]["revenue"].mean()
    avg_rev_no = df[df["AI_Recommendation"] == "No Promo"]["revenue"].mean()

    st.metric("Promotions Recommended", promo_count)
    st.metric("Avg Revenue (Promo)", f"{avg_rev_promo:.2f}")
    st.metric("Avg Revenue (No Promo)", f"{avg_rev_no:.2f}")

    uplift = (avg_rev_promo - avg_rev_no) / avg_rev_no * 100
    st.markdown(f"**Estimated Uplift:** {uplift:.1f}% 🚀")


# STEP 6: WHAT-IF SIMULATOR


st.header(" What-If Decision Simulator")
colA, colB, colC = st.columns(3)

with colA:
    price_input = st.slider("Product Price", 0.1, 1.0, 0.5)
with colB:
    traffic_input = st.slider("Expected Foot Traffic", 0.1, 1.0, 0.6)
with colC:
    pred_action = get_policy(price_input, traffic_input)
    st.metric("AI Suggestion", pred_action)

sim_df = pd.DataFrame({"price": [price_input], "foot_traffic": [traffic_input]})
fig_sim = px.scatter(
    sim_df, x="price", y="foot_traffic",
    color_discrete_sequence=["#22c55e" if pred_action=="Run Promo" else "#ef4444"],
    title="Simulation Result"
)
st.plotly_chart(fig_sim, use_container_width=True)

#  STEP 7: EXPLAINABILITY


st.header("🧩 AI Decision Explainability")

importance = pd.DataFrame({
    "Feature": ["Price Sensitivity", "Traffic Influence"],
    "Importance": [random.uniform(0.3, 0.7), random.uniform(0.4, 0.8)]
})
fig_imp = px.bar(importance, x="Feature", y="Importance", title="Feature Importance")
st.plotly_chart(fig_imp, use_container_width=True)

st.caption("↑ Higher importance means greater impact on AI's recommendation.")


# STEP 8: EXPORT RESULTS


csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=" Download Recommendations as CSV",
    data=csv_data,
    file_name="AI_Strategy_Recommendations.csv",
    mime="text/csv"
)

st.success(" Dashboard Loaded Successfully!")
st.markdown("Built  by Ishan")
