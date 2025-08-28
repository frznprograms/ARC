import tempfile
import os
from .quality_classifier import BaseClassifier


class FastTextClassifier(BaseClassifier):
    def __init__(self, lr=1.0, epoch=25, wordNgrams=2):
        self.lr = lr
        self.epoch = epoch
        self.wordNgrams = wordNgrams
        self.model = None
        self.label_prefix = "__label__"

    def _prepare_fasttext_format(self, X, y, file_path):
        # Write data in FastText supervised format
        with open(file_path, "w", encoding="utf-8") as f:
            for text, label in zip(X, y):
                f.write(f"{self.label_prefix}{label} {text.replace(chr(10), ' ')}\n")

    def train(self, X, y):
        import fasttext

        # Prepare data in FastText format
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as tmpfile:
            self._prepare_fasttext_format(X, y, tmpfile.name)
            train_file = tmpfile.name

        # Train FastText supervised model
        self.model = fasttext.train_supervised(
            input=train_file,
            lr=self.lr,
            epoch=self.epoch,
            wordNgrams=self.wordNgrams,
            verbose=0,
            thread=1,  # Use 1 thread for reproducibility
        )

        # Clean up
        os.unlink(train_file)

    def predict(self, X):
        # Ensure X is a list of strings
        if isinstance(X, str):
            X = [X]
        elif hasattr(X, "tolist"):
            X = X.tolist()
        labels, _ = self.model.predict(X)
        # FastText returns [['__label__2'], ...], so extract the number
        return [int(label[0].replace(self.label_prefix, "")) for label in labels]

    def save_model(self, file_path: str):
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save_model(file_path)

    def load_model(self, file_path: str):
        import fasttext

        self.model = fasttext.load_model(file_path)