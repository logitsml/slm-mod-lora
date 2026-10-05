# modmodmod-warmstart

Train a lightweight community-specific moderation head over frozen text embeddings.

`modmodmod-warmstart` is the warm-start counterpart to `modmodmod-coldstart`: it assumes a target community has historical moderation labels, embeds comments with a frozen text encoder, and trains a small per-community logistic head.

## What this releases

- Frozen encoder infrastructure using `intfloat/e5-large-v2` by default.
- Per-community `StandardScaler` + `LogisticRegressionCV` head.
- Train-only cross-validation for the logistic head's L2 regularization strength.
- Saved artifact containing only the scaler, logistic head, encoder metadata, and label metadata.

It does not include Reddit comments, usernames, subreddit modlogs, or paper benchmark data.

This repository implements the warm-start encoder/head method. It does not reproduce the full paper pipeline: dataset construction, seed splits, BAL-AUC evaluation, bootstrap intervals, and paper figure/table generation live outside this package.

## Install

```bash
pip install -e .
```

## Train

Input CSV must include a text column and a binary label column where `1` means removed and `0` means kept.

```bash
modmodmod-warmstart train \
  --csv examples/tiny_comments.csv \
  --text-col body \
  --label-col label \
  --output warmstart.pkl
```

## Score

```bash
modmodmod-warmstart score \
  --model warmstart.pkl \
  --csv examples/tiny_comments.csv \
  --text-col body \
  --output scores.csv
```

The output CSV adds:

- `removal_probability`
- `remove_decision`

## Python API

```python
from modmodmod_warmstart import WarmStartModerator

model = WarmStartModerator()
model.fit(
    comments=[
        "please stay on topic",
        "thanks for the helpful explanation",
        "you are an idiot",
        "go harass them somewhere else",
    ],
    labels=[0, 0, 1, 1],
)
model.save("warmstart.pkl")

loaded = WarmStartModerator.load("warmstart.pkl")
scores = loaded.predict_proba(["off-topic meme post"])
```

## Notes

- This is a supervised warm-start artifact: it uses target-community historical keep/remove labels.
- The head is `LogisticRegressionCV` with up to 5 folds; the fold count drops to the minority-class count when a community has fewer than 5 minority examples, and communities with fewer than 2 minority examples fall back to a fixed-regularization `LogisticRegression` (C=0.03).
- It is not initialized from `modmodmod-coldstart`; "warm-start" refers to the deployment setting where local moderation labels are available.
- The default encoder prompt prefix is `query: `, matching the paper's e5 setup.
- Saved artifacts are Python pickle files. Only load artifacts you trust.

## Exporting training data from your own subreddit

`examples/export_modlog_csv.py` builds a `body,label` CSV from a community
you moderate (PRAW; moderator account required). Removals and spam actions
from the mod log become label 1, explicit approvals and the recent comment
listing become label 0, the latest action per comment wins, and bodies are
de-duplicated last-label-wins. Run it on a schedule with `--state` to
accumulate labels incrementally, then retrain (head fits take under a
second per community):

```
python examples/export_modlog_csv.py --subreddit yoursub --out yoursub.csv --state yoursub.state.json
modmodmod-warmstart train --csv yoursub.csv --output yoursub.pkl
```

Continuous-learning practices that worked in our evaluation setting: retrain
on every export (it is cheap); keep the most recent week of labels as a
rolling holdout and watch its AUC for drift; treat explicit approvals as
stronger keep labels than untouched comments; once the model triages for
you, log its score at decision time and periodically audit a random sample
it did not flag, so moderator labels do not silently become model echoes;
and route near-threshold comments to moderators first, since those
decisions are the most informative labels you can collect.
