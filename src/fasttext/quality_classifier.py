import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings


def stratified_sample_split(split_data, max_samples, label_idx=1, debug=True):
    """
    Performs balanced random sampling on a data split.

    Aims to create a subset where each class is represented by an equal
    number of samples. If a class has fewer samples than the target,
    all of its samples are taken.

    Args:
        split_data (list or tuple): A list/tuple of arrays or DataFrames,
            e.g., (X_data, y_data).
        max_samples (int): The total number of samples desired. The final
                           count may be lower if classes are scarce.
        label_idx (int): The index in `split_data` that holds the labels (y).
                         Defaults to 1 for a structure like (X, y).
        debug (bool): If True, prints detailed execution logs.

    Returns:
        list: A new list of sampled and class-balanced arrays/DataFrames.
    """
    if debug:
        print("\n" + "=" * 80)
        print(f"[DEBUG] Attempting BALANCED sampling with max_samples={max_samples}.")
        print(f"[DEBUG] Input split_data contains {len(split_data)} elements.")
        print("=" * 80)

    try:
        # 1. Check if sampling is necessary
        if not split_data or len(split_data[0]) <= max_samples:
            if debug:
                print(
                    "[DEBUG] No sampling needed: dataset is smaller than or equal to max_samples."
                )
            return split_data

        # 2. Identify labels and get class counts
        labels = split_data[label_idx]
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels.flatten())

        unique_classes = labels.unique()
        n_classes = len(unique_classes)
        if n_classes == 0:
            raise ValueError("No classes found in the label array.")

        if debug:
            print(
                f"[DEBUG] Found {n_classes} unique classes: {unique_classes.tolist()}"
            )
            print("[DEBUG] Original class counts:")
            print(labels.value_counts().to_string())

        # 3. Determine target samples per class for a balanced set
        target_per_class = max_samples // n_classes
        if debug:
            print(
                f"\n[DEBUG] Target samples per class: {max_samples} / {n_classes} = {target_per_class}"
            )

        # 4. Collect indices for each class, respecting availability
        final_indices = []
        if debug:
            print("\n[DEBUG] Sampling within each class to create a balanced set...")
        for class_label in unique_classes:
            # Get integer positions of all samples for the current class
            class_indices = np.where(labels.to_numpy() == class_label)[0]
            n_available = len(class_indices)

            # CRITICAL LOGIC: Take the target number, or all available if fewer
            n_to_sample = min(target_per_class, n_available)

            if debug:
                print(
                    f"  - Class '{class_label}': Target is {target_per_class}. Found {n_available} available. Taking {n_to_sample}."
                )

            if n_to_sample == 0:
                continue

            sampled_indices = np.random.choice(
                class_indices, size=n_to_sample, replace=False
            )
            final_indices.append(sampled_indices)

        # 5. Combine, shuffle, and apply final indices
        final_indices = np.concatenate(final_indices)
        np.random.shuffle(final_indices)

        if debug:
            print(
                f"\n[DEBUG] Total indices collected for balanced set: {len(final_indices)}"
            )
            if len(final_indices) < max_samples:
                print(
                    f"  - NOTE: Final sample count is less than {max_samples} due to scarcity in one or more classes."
                )

        # 6. Slice all arrays in the split
        sampled_split = []
        for arr in split_data:
            if hasattr(arr, "iloc"):
                sampled_split.append(arr.iloc[final_indices].reset_index(drop=True))
            else:
                sampled_split.append(arr[final_indices])

        if debug:
            print("\n[DEBUG] Slicing complete. Verifying new class distribution:")
            new_labels = sampled_split[label_idx]
            if not isinstance(new_labels, pd.Series):
                new_labels = pd.Series(new_labels.flatten())
            print(new_labels.value_counts().to_string())

        print("\n[DEBUG] Balanced sampling successful.")
        print("=" * 80 + "\n")
        return sampled_split

    except Exception as e:
        # --- Fallback Logic ---
        warnings.warn(
            f"Balanced sampling failed with error: {e}. Falling back to simple random sampling.",
            UserWarning,
        )
        if debug:
            import traceback

            print("\n" + "!" * 80)
            print(
                "! [DEBUG] BALANCED SAMPLING FAILED. FALLING BACK TO SIMPLE SAMPLING."
            )
            print(f"! [DEBUG] Error was: {e}")
            traceback.print_exc()
            print("!" * 80 + "\n")

        n_samples_total = len(split_data[0])
        # Ensure we don't request more samples than available in the fallback
        n_to_sample_fallback = min(n_samples_total, max_samples)
        indices = np.random.choice(n_samples_total, n_to_sample_fallback, replace=False)

        fallback_split = []
        for arr in split_data:
            if hasattr(arr, "iloc"):
                fallback_split.append(arr.iloc[indices].reset_index(drop=True))
            else:
                fallback_split.append(arr[indices])
        return fallback_split


class BaseClassifier:
    def train(self, X, y):
        """Fit the model on features X and labels y."""
        raise NotImplementedError

    def predict(self, X):
        """Return predicted labels for features X."""
        raise NotImplementedError

    def save_model(self, file_path: str):
        """
        Serialize this classifier to disk at the given file path.

        Args:
            file_path: Path (including filename) where the model will be saved.
        """
        raise NotImplementedError


class DataLoader:
    def __init__(
        self,
        csv_path=None,
        df=None,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        with_val=True,
        stratify=None,
        max_n_samples_eval=None,
    ):
        """
        Args:
            csv_path: Path to CSV file with 'text' and 'label' columns (optional if df provided)
            df: Preloaded DataFrame with 'text' and 'label' columns (optional if csv_path provided)
            test_size: Proportion of data to use for test set (default 0.2)
            val_size: Proportion of remaining data to use for validation (default 0.2)
            random_state: Random seed for reproducibility (default 42)
            with_val: Whether to include a validation split (default True)
            stratify: Whether to split by language (default None)
            max_n_samples_eval: Limit for the number of samples used for validation (default None)
        """
        if csv_path is None and df is None:
            raise ValueError("Either csv_path or df must be provided")
        if csv_path is not None and df is not None:
            raise ValueError("Only one of csv_path or df should be provided")

        self.csv_path = csv_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.with_val = with_val
        self.stratify = stratify
        self.max_n_samples_eval = max_n_samples_eval

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_path)

        self.splits = self._split_data()

    def _split_data(self):
        X = self.df["text"].astype(str)
        y = self.df["label"].astype(str)

        # Always split out test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.stratify,
        )
        if self.with_val:
            val_ratio = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=self.stratify,
            )
            splits = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
                "df": self.df,
            }
        else:
            splits = {
                "train": (X_temp, y_temp),
                "test": (X_test, y_test),
                "df": self.df,
            }

        if self.max_n_samples_eval:
            if splits.get("val"):
                splits["val"] = stratified_sample_split(
                    splits["val"], self.max_n_samples_eval
                )

            if splits.get("test"):
                splits["test"] = stratified_sample_split(
                    splits["test"], self.max_n_samples_eval
                )

        return splits

    def get_split(self, split):
        return self.splits.get(split)

    def get_all(self):
        return self.splits