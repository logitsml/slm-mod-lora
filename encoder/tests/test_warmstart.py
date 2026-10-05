from pathlib import Path

import numpy as np

from modmodmod_warmstart import WarmStartModerator
from modmodmod_warmstart.cli import main


def test_fit_predict_from_embeddings():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [1.1, 1.1],
        ],
        dtype=np.float32,
    )
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    model = WarmStartModerator().fit_from_embeddings(embeddings, labels)
    probabilities = model.predict_proba(embeddings=embeddings)

    assert probabilities.shape == (8,)
    assert probabilities[:4].mean() < probabilities[4:].mean()
    assert model.metadata["n_train"] == 8


def test_save_load_roundtrip(tmp_path: Path):
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=np.float32,
    )
    labels = [0, 0, 1, 1]
    model = WarmStartModerator().fit_from_embeddings(embeddings, labels)

    path = tmp_path / "warmstart.pkl"
    model.save(path)
    loaded = WarmStartModerator.load(path)

    np.testing.assert_allclose(
        model.predict_proba(embeddings=embeddings),
        loaded.predict_proba(embeddings=embeddings),
    )


def test_cli_score_uses_saved_model(tmp_path: Path, monkeypatch):
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=np.float32,
    )
    labels = [0, 0, 1, 1]
    model = WarmStartModerator().fit_from_embeddings(embeddings, labels)

    calls = {"count": 0}

    def fake_embed(comments):
        calls["count"] += 1
        return embeddings[: len(comments)]

    model_path = tmp_path / "warmstart.pkl"
    model.save(model_path)
    monkeypatch.setattr(WarmStartModerator, "embed", lambda self, comments: fake_embed(comments))

    csv_path = tmp_path / "comments.csv"
    csv_path.write_text("body\none\ntwo\nthree\nfour\n", encoding="utf-8")
    out_path = tmp_path / "scores.csv"

    main([
        "score",
        "--model",
        str(model_path),
        "--csv",
        str(csv_path),
        "--text-col",
        "body",
        "--output",
        str(out_path),
    ])

    assert calls["count"] == 1
    assert "removal_probability" in out_path.read_text(encoding="utf-8")
