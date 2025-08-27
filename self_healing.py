import os
import time
from datetime import datetime
import pandas as pd
import joblib

MODEL_FILE = "model.joblib"
ACTIONS_LOG = "actions_log.csv"

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    data = joblib.load(MODEL_FILE)
    return data['vectorizer'], data['model']

def remedial_action(message):
    msg = message.lower()
    if "database" in msg:
        return "Restart Database Service"
    elif "memory" in msg or "outofmemory" in msg:
        return "Clear cache / Restart app"
    elif "cpu" in msg:
        return "Reduce load / Restart heavy worker"
    elif "disk" in msg:
        return "Clean temp files / Alert storage admin"
    elif "too many open files" in msg:
        return "Increase file descriptor limit"
    else:
        return "Restart service (generic)"

def log_action(timestamp, message, prediction, action):
    row = {'timestamp': timestamp, 'message': message, 'prediction': int(prediction), 'action': action}
    file_exists = os.path.exists(ACTIONS_LOG)
    pd.DataFrame([row]).to_csv(ACTIONS_LOG, index=False, header=not file_exists, mode='a')

def predict_message(message, vectorizer, model):
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max() if hasattr(model, 'predict_proba') else None
    return pred, prob

def remediate_and_log(message, vectorizer, model):
    pred, prob = predict_message(message, vectorizer, model)
    action = None
    timestamp = datetime.utcnow().isoformat()
    if pred == 1:
        action = remedial_action(message)
        print(f"[{timestamp}] 🚨 Anomaly: '{message}' -> Action: {action} (p={prob})")
        log_action(timestamp, message, pred, action)
    else:
        print(f"[{timestamp}] ✅ Normal: '{message}'")
    return pred, prob, action

def simulate_stream(n_messages=20, sleep_sec=0.5):
    vectorizer, model = load_model()
    df = pd.read_csv("logs.csv")
    for _ in range(n_messages):
        msg = df.sample(1).iloc[0]['message']
        remediate_and_log(msg, vectorizer, model)
        time.sleep(sleep_sec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", type=int, help="Simulate N logs in CLI")
    args = parser.parse_args()
    if args.simulate:
        simulate_stream(args.simulate)
