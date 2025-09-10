import joblib
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def plot_lda():
    combined_data = pd.read_csv("data/for_model/combined_safety_data.csv")
    combined_data.dropna(inplace=True)
    combined_data = combined_data.sample(frac=0.02)

    safety_pipeline = joblib.load("models/safety-model-test.pkl")
    tfidf_vec = safety_pipeline.named_steps["features"]["tfidf"]

    reviews, labels = combined_data["text"], combined_data["unsafe_label"]
    logger.info("Preparing TF-IDF vector...")
    X_tfidf = tfidf_vec.transform(reviews)

    lda = LinearDiscriminantAnalysis(n_components=1)

    logger.info("Preparing LDA plot...")
    X_lda = lda.fit_transform(X_tfidf.to_array(), labels)

    plt.hist(X_lda[labels == 0], alpha=0.6, label="Safe", bins=30)
    plt.hist(X_lda[labels == 1], alpha=0.6, label="Unsafe", bins=30)
    plt.title("LDA Projection of Reviews")
    plt.legend()
    plt.xlabel("LDA Component")
    plt.show()


if __name__ == "__main__":
    plot_lda()
