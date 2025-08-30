import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger
import pandas as pd

if __name__ == "__main__":
    combined_data = pd.read_csv("data/for_model/combined_safety_data.csv")
    model = joblib.load("models/safety-model-test.pkl")
    logger.info("Loaded model from local directory...")

    X, y = combined_data["text"], combined_data["unsafe_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logger.info("Now evaluating, please do not interrupt...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(report)
    print(f"Accuracy: {accuracy:.4f}")
