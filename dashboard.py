import streamlit as st
import pandas as pd
from self_healing import load_model, remedial_action, log_action, predict_message, remediate_and_log
from datetime import datetime
import os
import time

st.set_page_config(page_title="AIOps Self-Healing", layout="wide")
st.title("🤖 AIOps Self-Healing Simulator")
st.markdown("Simulates logs, detects anomalies, and applies remedial actions automatically.")

vectorizer, model = load_model()

# Sidebar controls
st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Simulate Stream", "Manual Test"])
n_msgs = st.sidebar.slider("Messages per run", 1, 100, 20)
sleep = st.sidebar.slider("Delay (ms)", 0, 2000, 200)

def predict_and_render(message, index, placeholder):
    pred, prob = predict_message(message, vectorizer, model)
    ts = datetime.utcnow().isoformat()
    if pred == 1:
        action = remedial_action(message)
        log_action(ts, message, pred, action)
        with placeholder.container():
            st.markdown(f"**#{index+1} — {ts} — :red[ANOMALY]**")
            st.write(message)
            st.warning(f"Action: {action} | Confidence: {prob:.2f}" if prob else f"Action: {action}")
    else:
        with placeholder.container():
            st.markdown(f"**#{index+1} — {ts} — :green[NORMAL]**")
            st.write(message)
            if prob:
                st.write(f"Confidence: {prob:.2f}")

if mode == "Simulate Stream":
    if st.sidebar.button("Start Simulation"):
        placeholder = st.empty()
        df_logs = pd.read_csv("logs.csv")
        for i in range(n_msgs):
            msg = df_logs.sample(1).iloc[0]['message']
            predict_and_render(msg, i, placeholder)
            time.sleep(sleep/1000.0)
        st.success("Simulation finished.")
else:
    user_msg = st.sidebar.text_area("Log message", value="ERROR - Database connection failed")
    if st.sidebar.button("Test Message"):
        placeholder = st.empty()
        predict_and_render(user_msg, 0, placeholder)

st.header("Remediation Actions Log")
if os.path.exists("actions_log.csv"):
    df_actions = pd.read_csv("actions_log.csv")
    st.dataframe(df_actions.sort_values("timestamp", ascending=False).head(50))
else:
    st.info("No actions logged yet.")
