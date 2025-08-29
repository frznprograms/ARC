import yaml
import itertools


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def expand_config(config: dict):
    tfidf = config["safety_pipeline"]["tfidf"]
    clf = config["safety_pipeline"]["clf"]
    lexicon = config["safety_pipeline"]["lexicon"]

    for (
        min_df,
        max_features,
        ngram_range,
        max_iter,
        lexicon_enabled,
    ) in itertools.product(
        tfidf["min_df"],
        tfidf["max_features"],
        [tuple(tfidf["ngram_range"])],  # convert to tuple here
        clf["max_iter"],
        lexicon["enabled"],
    ):
        yield {
            "pipeline": {
                "tfidf": {
                    "min_df": min_df,
                    "max_features": max_features,
                    "ngram_range": ngram_range,
                },
                "clf": {
                    "class_weight": clf["class_weight"],
                    "max_iter": max_iter,
                },
                "lexicon": {
                    "enabled": lexicon_enabled,
                },
            }
        }
