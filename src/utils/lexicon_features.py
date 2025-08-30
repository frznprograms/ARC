from sklearn.base import BaseEstimator, TransformerMixin


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
