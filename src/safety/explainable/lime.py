import joblib

from lime.lime_text import LimeTextExplainer


def lime(example: str, num_features: int = 10):
    safety_pipeline = joblib.load("models/safety-model-test.pkl")

    def predict_proba(texts):
        return safety_pipeline.predict_proba(texts)

    explainer = LimeTextExplainer(class_names=["Safe", "Unsafe"])

    probs = predict_proba([example])[0]
    pred_label = probs.argmax()
    pred_class_name = explainer.class_names[pred_label]  # type: ignore
    pred_confidence = probs[pred_label]

    print(f"Prediction: {pred_class_name} ({pred_label})")
    print(f"Confidence: {pred_confidence:.4f}")

    exp = explainer.explain_instance(
        example, predict_proba, num_features=num_features, labels=(0, 1)
    )

    print("\nExplanation for class 'Unsafe':\n")
    print(exp.as_list(label=0))

    print("\nExplanation for class 'Safe':\n")
    print(exp.as_list(label=1))


if __name__ == "__main__":
    example = "I really hated him, I wanted to beat him up!"

    lime(example=example)
