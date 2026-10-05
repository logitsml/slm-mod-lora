from __future__ import annotations

import pickle
from inspect import signature
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


def _as_labels(labels: Sequence[int]) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError("labels must be binary: 0=kept, 1=removed")
    if len(unique) < 2:
        raise ValueError("training requires at least one kept and one removed example")
    return y


def _fit_cv_head(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression | LogisticRegressionCV:
    minority_count = min(int((y_train == 1).sum()), int((y_train == 0).sum()))
    if minority_count < 2:
        return LogisticRegression(max_iter=2000, C=0.03, random_state=11).fit(x_train, y_train)
    kwargs = {
        "Cs": np.logspace(-4, 2, 16),
        "cv": min(5, minority_count),
        "scoring": "roc_auc",
        "max_iter": 2000,
        "random_state": 11,
        "l1_ratios": (0,),
    }
    if "use_legacy_attributes" in signature(LogisticRegressionCV).parameters:
        kwargs["use_legacy_attributes"] = True
    return LogisticRegressionCV(**kwargs).fit(x_train, y_train)


@dataclass
class WarmStartModerator:
    encoder_name: str = "intfloat/e5-large-v2"
    prefix: str = "query: "
    batch_size: int = 128
    device: str = "auto"
    scaler: StandardScaler | None = None
    head: LogisticRegression | LogisticRegressionCV | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    _encoder_model: object | None = field(default=None, init=False, repr=False, compare=False)

    def fit(self, comments: Sequence[str], labels: Sequence[int]) -> "WarmStartModerator":
        embeddings = self.embed(comments)
        return self.fit_from_embeddings(embeddings, labels)

    def fit_from_embeddings(self, embeddings: np.ndarray, labels: Sequence[int]) -> "WarmStartModerator":
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        y = _as_labels(labels)
        if len(x) != len(y):
            raise ValueError(f"embedding/label length mismatch: {len(x)} != {len(y)}")

        self.scaler = StandardScaler().fit(x)
        x_train = self.scaler.transform(x)
        self.head = _fit_cv_head(x_train, y)
        self.metadata = {
            "n_train": int(len(y)),
            "n_removed": int((y == 1).sum()),
            "n_kept": int((y == 0).sum()),
            "embedding_dim": int(x.shape[1]),
        }
        return self

    def embed(self, comments: Sequence[str]) -> np.ndarray:
        model = self._get_encoder_model()
        inputs = [self.prefix + comment for comment in comments]
        return model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _get_encoder_model(self) -> object:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Text embedding requires sentence-transformers and torch. "
                "Install the package with its dependencies: `pip install -e .` from the repository root."
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._encoder_model is None:
            self._encoder_model = SentenceTransformer(self.encoder_name, device=device)
        return self._encoder_model

    def predict_proba(self, comments: Sequence[str] | None = None, embeddings: np.ndarray | None = None) -> np.ndarray:
        if self.scaler is None or self.head is None:
            raise ValueError("model is not fitted")
        if embeddings is None:
            if comments is None:
                raise ValueError("provide comments or embeddings")
            embeddings = self.embed(comments)
        x = self.scaler.transform(np.asarray(embeddings, dtype=np.float32))
        return self.head.predict_proba(x)[:, 1]

    def predict(
        self,
        comments: Sequence[str] | None = None,
        embeddings: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        return (self.predict_proba(comments=comments, embeddings=embeddings) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_encoder_model"] = None
        return state

    @classmethod
    def load(cls, path: str | Path) -> "WarmStartModerator":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected WarmStartModerator artifact, got {type(obj)!r}")
        return obj
