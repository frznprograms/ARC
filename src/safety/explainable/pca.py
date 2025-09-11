import joblib
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger

from sklearn.decomposition import PCA


def plot_pca():
    combined_data = pd.read_csv("data/for_model/combined_safety_data.csv")
    combined_data.dropna(inplace=True)
    combined_data = combined_data.sample(frac=0.02)

    safety_pipeline = joblib.load("models/safety-model-test.pkl")
    tfidf_vec = safety_pipeline.named_steps["features"]["tfidf"]

    reviews, labels = combined_data["text"], combined_data["unsafe_label"]
    logger.info("Preparing TF-IDF vector...")
    X_tfidf = tfidf_vec.transform(reviews)

    logger.info("Preparing PCA plot...")
    X_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm", alpha=0.6)
    plt.title("PCA Projection of TF-IDF Vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Label (0=Safe, 1=Unsafe)")
    plt.show()


if __name__ == "__main__":
    plot_pca()
