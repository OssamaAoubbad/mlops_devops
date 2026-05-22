import re

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from madewithml.config import STOPWORDS


def load_data(dataset_loc: str, num_samples: int | None = None) -> Dataset:
    """Load data from source into a Ray Dataset."""
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=1234)
    if num_samples:
        ds = ray.data.from_items(ds.take(num_samples))
    return ds


def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and test splits with equal amounts of data points from each class."""

    def _add_split(df: pd.DataFrame) -> pd.DataFrame:
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[df["_split"] == split].drop("_split", axis=1)

    # Train, test split with stratify
    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")
    
    # Modernized map_batches syntax
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas")

    # Shuffle each split
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


def clean_text(text: str, stopwords: list[str] = STOPWORDS) -> str:
    """Clean raw text string."""
    text = text.lower()

    # Remove stopwords using regex word boundaries
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)

    return text


def tokenize(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Tokenize the text input in our batch using a tokenizer."""
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    
    # Modern HuggingFace prefers returning dicts natively
    encoded_inputs = tokenizer(batch["text"].tolist(), padding="longest", return_tensors="np")
    
    return {
        "ids": encoded_inputs["input_ids"],
        "masks": encoded_inputs["attention_mask"],
        "targets": np.array(batch["tag"])
    }


def preprocess(df: pd.DataFrame, class_to_index: dict[str, int]) -> dict[str, np.ndarray]:
    """Preprocess the data in our dataframe."""
    df["text"] = df.title + " " + df.description
    df["text"] = df.text.apply(clean_text)
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")
    df = df[["text", "tag"]]
    df["tag"] = df["tag"].map(class_to_index)
    
    return tokenize(df)


class CustomPreprocessor:
    """Custom preprocessor class holding state for label encoding."""

    def __init__(self, class_to_index: dict[str, int] | None = None):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, ds: Dataset):
        tags = ds.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, ds: Dataset) -> Dataset:
        return ds.map_batches(
            preprocess, 
            fn_kwargs={"class_to_index": self.class_to_index}, 
            batch_format="pandas"
        )