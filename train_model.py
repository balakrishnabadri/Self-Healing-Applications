import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

LOGS_CSV = "logs.csv"
MODEL_FILE = "model.joblib"

def train_model():
    df = pd.read_csv(LOGS_CSV)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Model trained. Test results:")
    print(classification_report(y_test, preds))
    joblib.dump({'vectorizer': vectorizer, 'model': clf}, MODEL_FILE)
    print(f"Model saved -> {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
