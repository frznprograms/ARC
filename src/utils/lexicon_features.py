from sklearn.base import BaseEstimator, TransformerMixin


class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom feature extractor that uses a lexicon to identify the presence of specific words in text data.

    This transformer checks whether any word from the provided lexicon appears in each text sample and outputs
    a binary feature (1 if any word is found, 0 otherwise).

    Attributes:
        lexicon (list[str]): A list of words to check for in the text data.
    """

    def __init__(self, lexicon):
        """
        Initializes the LexiconFeatureExtractor with a given lexicon.

        Args:
            lexicon (list[str]): A list of words to use for feature extraction.
        """
        self.lexicon = lexicon

    def fit(self, X, y=None):
        """
        Fits the transformer to the data. This method does nothing as the transformer is stateless.

        Args:
            X (iterable): The input data (not used in this method).
            y (iterable, optional): The target labels (not used in this method).

        Returns:
            LexiconFeatureExtractor: The fitted transformer.
        """
        # do nothing
        return self

    def transform(self, X):
        """
        Transforms the input data into binary features based on the presence of lexicon words.

        Args:
            X (iterable): The input text data.

        Returns:
            list[list[int]]: A list of binary features for each text sample. Each feature is 1 if any word
            from the lexicon is found in the text, and 0 otherwise.
        """
        return [
            [1 if any(word in text.lower() for word in self.lexicon) else 0]
            for text in X
        ]
