import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils.base_helpers import read_json_safety_config, timed_execution
from src.utils.lexicon_features import LexiconFeatureExtractor
from src.utils.toxic_lexicon import toxic_lexicon


@dataclass
class SafetyPipeline:
    """
    A pipeline for training and evaluating a safety model to classify text as safe or unsafe.

    Attributes:
        data (pd.DataFrame): The input data containing text and labels.
        data_prepared (bool): A flag indicating whether the data has been split into training and testing sets.
        seed (int): The random seed for reproducibility.
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_prepared: bool = False
    seed: int = 42

    def create_pipeline(self):
        """
        Creates a machine learning pipeline with TF-IDF features and lexicon-based features,
        followed by a logistic regression classifier.

        The pipeline includes:
        - TF-IDF vectorization of text data.
        - Lexicon-based feature extraction using a predefined toxic lexicon.
        - Logistic regression for classification.
        """
        tfidf = TfidfVectorizer()
        features = FeatureUnion(
            [
                ("tfidf", tfidf),  # type: ignore
                ("lexicon", LexiconFeatureExtractor(toxic_lexicon)),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(random_state=self.seed)),
            ]
        )

    @timed_execution
    @logger.catch(message="Unable to finish training pipeline.", reraise=True)
    def train_pipeline(
        self,
        param_grid: dict,
        save_name: str = "experiment_default.csv",
        cv: int = 3,
        test_size: float = 0.25,
    ):
        """
        Trains the pipeline using GridSearchCV to find the best hyperparameters.

        Args:
            param_grid (dict): The hyperparameter grid for GridSearchCV.
            save_name (str): The name of the file to save the trained model. Defaults to "experiment_default.csv".
            cv (int): The number of cross-validation folds. Defaults to 3.
            test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.25.

        Returns:
            GridSearchCV: The fitted GridSearchCV object.

        Raises:
            Exception: If the training process fails.
        """
        if not self.data_prepared:
            self.prepare_data(test_size=test_size)

        logger.info("Now training, please do not interrupt...")
        search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)
        self.pipeline = search.best_estimator_
        best_params = search.best_params_

        save_path = Path("models")  # type: ignore
        save_path.mkdir(parents=True, exist_ok=True)  # type:ignore
        model_save_path = save_path / save_name
        joblib.dump(self.pipeline, model_save_path)
        with open(f"models/best-params-{save_name}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        logger.success(f"Best params: {search.best_params_}")
        return search

    @timed_execution
    @logger.catch(message="Unable to finish evaluating pipeline.", reraise=True)
    def eval_pipeline(self, threshold: float = 0.5):
        """
        Evaluates the trained pipeline on the test data.

        Args:
            threshold (float): The decision threshold for classification. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the classification report (str) and accuracy (float).

        Raises:
            Exception: If the evaluation process fails.
        """
        logger.info("Now evaluating, please do not interrupt...")
        y_pred_prob = self.pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)

        report = classification_report(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(report)
        print(f"Accuracy: {accuracy:.4f}")

        return report, accuracy

    @logger.catch(message="Unable to prepare train-test split.")
    def prepare_data(self, test_size: float = 0.25):
        """
        Prepares the data by splitting it into training and testing sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.25.

        Raises:
            Exception: If the data preparation process fails.
        """
        X, y = self.data["text"], self.data["unsafe_label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        self.data_prepared = True


if __name__ == "__main__":
    param_grid = read_json_safety_config(
        json_config_path="src/configs/safety_config.json"
    )

    combined_data = pd.read_csv("data/for_model/combined_safety_data.csv")
    combined_data.dropna(inplace=True)
    sp = SafetyPipeline(data=combined_data)
    sp.create_pipeline()
    sp.train_pipeline(param_grid=param_grid, save_name="safety-model-test.pkl")
    sp.eval_pipeline()
