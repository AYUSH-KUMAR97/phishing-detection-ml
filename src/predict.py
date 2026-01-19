import joblib
import numpy as np

MODEL_PATH = "model/phishing_classifier.pkl"

model = joblib.load(MODEL_PATH)


def predict_email(text):
    prob = model.predict_proba([text])[0]
    label = model.predict([text])[0]
    confidence = np.max(prob) * 100

    result = "🚨 Phishing" if label == 1 else "✅ Legitimate"
    return result, round(confidence, 2)


if __name__ == "__main__":
    email = input("Enter email text: ")
    result, confidence = predict_email(email)
    print("Prediction:", result)
    print("Confidence:", confidence, "%")
