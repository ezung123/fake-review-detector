from pathlib import Path
import re
import string
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "burstiness",
    "type_token_ratio",
    "avg_word_length",
    "punctuation_density",
    "text_length",
]

class StylometricExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts stylometric and linguistic features from review texts:
      1. burstiness: sentence-length variance (in words)
      2. type_token_ratio: unique words / total words
      3. avg_word_length: average length of words in characters
      4. punctuation_density: punctuation count / total character length
      5. text_length: character length
      6. perplexity (optional): GPT-2 perplexity merged from perplexity.csv if present
    """
    def __init__(self, perplexity_path: str | Path | None = None):
        self.perplexity_path = Path(perplexity_path) if perplexity_path else Path("data/processed/perplexity.csv")
        self.perplexity_map: dict[str, float] | None = None
        self._feature_names = list(FEATURE_NAMES)
        
    def fit(self, X, y=None):
        if self.perplexity_path.is_file():
            try:
                p_df = pd.read_csv(self.perplexity_path)
                if "row_id" in p_df.columns and "perplexity" in p_df.columns:
                    self.perplexity_map = dict(zip(p_df["row_id"].astype(str), p_df["perplexity"].astype(float)))
                    if "perplexity" not in self._feature_names:
                        self._feature_names.append("perplexity")
            except Exception:
                self.perplexity_map = None
        return self

    def _extract_single(self, text: str, row_id: str | None = None) -> list[float]:
        text_str = str(text) if text is not None else ""
        text_len = len(text_str)
        words = re.findall(r"\b\w+\b", text_str)
        n_words = len(words)
        
        avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
        ttr = len(set(w.lower() for w in words)) / max(n_words, 1)
        punct_count = sum(1 for c in text_str if c in string.punctuation)
        punct_density = punct_count / max(text_len, 1)
        
        sentences = [s.strip() for s in re.split(r"[.!?]+", text_str) if s.strip()]
        sent_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences if len(re.findall(r"\b\w+\b", s)) > 0]
        burstiness = float(np.var(sent_lens, ddof=1)) if len(sent_lens) > 1 else 0.0
        
        feats = [burstiness, ttr, avg_word_len, punct_density, float(text_len)]
        
        if self.perplexity_map is not None:
            ppl = self.perplexity_map.get(str(row_id), 0.0) if row_id else 0.0
            feats.append(float(ppl))
            
        return feats

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            texts = X["text"].tolist()
            row_ids = X["row_id"].astype(str).tolist() if "row_id" in X.columns else [None] * len(texts)
        elif isinstance(X, pd.Series):
            texts = X.tolist()
            row_ids = [None] * len(texts)
        else:
            texts = list(X)
            row_ids = [None] * len(texts)
            
        matrix = [self._extract_single(t, r_id) for t, r_id in zip(texts, row_ids)]
        return np.array(matrix, dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return list(self._feature_names)


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts only character text length as a trivial single-feature baseline.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            texts = X["text"].tolist()
        elif isinstance(X, pd.Series):
            texts = X.tolist()
        else:
            texts = list(X)
        lens = [[float(len(str(t)))] for t in texts]
        return np.array(lens, dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return ["text_length"]

