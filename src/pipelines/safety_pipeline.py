from dataclasses import dataclass, field
import pandas as pd
from typing import Optional
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils.base_helpers import timed_execution, prepare_safety_datasets
from src.utils.toxic_lexicon import toxic_lexicon
from src.utils.config_helpers import load_config


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

    def create_pipeline(self, config: dict):
        tfidf_cfg = config["safety_pipeline"]["tfidf"]
        clf_cfg = config["safety_pipeline"]["clf"]
        lexicon_enabled = config["safety_pipeline"]["lexicon"]["enabled"]

        if "ngram_range" in tfidf_cfg and isinstance(tfidf_cfg["ngram_range"], list):
            tfidf_cfg["ngram_range"] = tuple(tfidf_cfg["ngram_range"])

        tfidf = TfidfVectorizer(**tfidf_cfg)

        features = []
        features.append(("tfidf", tfidf))
        if lexicon_enabled:
            features.append(("lexicon", LexiconFeatureExtractor(toxic_lexicon)))

        clf = LogisticRegression(**clf_cfg)

        self.pipeline = Pipeline(
            [
                ("features", FeatureUnion(features)),
                ("clf", clf),
            ]
        )

    @timed_execution
    @logger.catch(message="Unable to finish training pipeline.", reraise=True)
    def train_pipeline(self, test_size: float = 0.25, config: Optional[dict] = None):
        if not self.data_prepared:
            self.prepare_data(test_size=test_size)

        if not hasattr(self, "pipeline"):
            if config is None:
                logger.error(
                    "Pipeline not created. Call create_pipeline() and pass a config."
                )
            self.create_pipeline(config=config)  # type: ignore

        logger.info("Now training, please do not interrupt...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.success("Finished training pipeline.")

    @timed_execution
    @logger.catch(message="Unable to finish evaluating pipeline.", reraise=True)
    def eval_pipeline(self):
        logger.info("Now evaluating, please do not interrupt...")
        y_pred = self.pipeline.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")

    @logger.catch(message="Unable to prepare train-test split.")
    def prepare_data(self, test_size: float = 0.25):
        X, y = self.data["text"], self.data["unsafe_label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )
        self.data_prepared = True


if __name__ == "__main__":
    combined_data = prepare_safety_datasets(
        jigsaw_path="data/raw/toxicity/jigsaw.csv",
        jigsaw_save_path="data/cleaned/trial/jigsaw_clean.csv",
        toxigen_save_path="data/cleaned/trial/toxiegn_clean.csv",
        twitter_path="data/raw/toxicity/twitter.csv",
        twitter_save_path="data/cleaned/trial/twitter.csv",
    )
    # print(combined_data.head())

    sp = SafetyPipeline(data=combined_data)
    test_config = load_config(path="src/configs/safety_config.yaml")
    sp.create_pipeline(config=test_config)
    sp.train_pipeline()
    sp.eval_pipeline()
