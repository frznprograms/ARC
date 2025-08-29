from pathlib import Path
import json
import joblib
from dataclasses import dataclass, field
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils.base_helpers import read_json_safety_config, timed_execution
from src.utils.toxic_lexicon import toxic_lexicon


class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def fit(self, X, y=None):
        # do nothing
        return self

    def transform(self, X):
        return [
            [1 if any(word in text.lower() for word in self.lexicon) else 0]
            for text in X
        ]


@dataclass
class SafetyPipeline:
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_prepared: bool = False
    seed: int = 42

    def create_pipeline(self):
        tfidf = TfidfVectorizer()
        features = FeatureUnion(
            [
                ("tfidf", tfidf),
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
        if not self.data_prepared:
            self.prepare_data(test_size=test_size)

        logger.info("Now training, please do not interrupt...")
        search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="recall",
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
    def eval_pipeline(self):
        logger.info("Now evaluating, please do not interrupt...")
        y_pred = self.pipeline.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(report)
        print(f"Accuracy: {accuracy:.4f}")
        return report, accuracy

    @logger.catch(message="Unable to prepare train-test split.")
    def prepare_data(self, test_size: float = 0.25):
        X, y = self.data["text"], self.data["unsafe_label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        self.data_prepared = True


if __name__ == "__main__":
    # combined_data = prepare_safety_datasets(
    #     jigsaw_path="data/raw/toxicity/jigsaw.csv",
    #     jigsaw_save_path="data/cleaned/trial/jigsaw_clean.csv",
    #     toxigen_save_path="data/cleaned/trial/toxiegn_clean.csv",
    #     twitter_path="data/raw/toxicity/twitter.csv",
    #     twitter_save_path="data/cleaned/trial/twitter.csv",
    # )
    # combined_data.to_csv("data/for_model/combined_safety_data.csv", index=False)
    #

    param_grid = read_json_safety_config(
        json_config_path="src/configs/safety_config.json"
    )

    combined_data = pd.read_csv("data/for_model/combined_safety_data.csv")
    combined_data.dropna(inplace=True)
    sp = SafetyPipeline(data=combined_data)
    sp.create_pipeline()
    sp.train_pipeline(param_grid=param_grid, save_name="safety-model-test.pkl")
    sp.eval_pipeline()
