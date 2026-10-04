"""Shared L2-regularized logistic head with the L2 strength chosen by cross-validation.

Rather than relying on sklearn's default C=1.0, this helper cross-validates the L2
strength on the TRAIN fold (leakage-free) via a log-spaced grid over C, so the chosen
regularization adapts to each fit instead of using a fixed prior.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def cv_logreg_fit(Xtr, ytr, class_weight=None):
    """Fit a logistic head with L2 strength chosen by inner StratifiedKFold ROC-AUC on the TRAIN fold only.
    Falls back to a fixed strong C=0.03 when a class has <2 train examples (CV infeasible)."""
    ytr = np.asarray(ytr)
    # size of the rarer class -- caps how many CV folds are feasible (need >=1 of each class per fold)
    mc = min(int((ytr == 1).sum()), int((ytr == 0).sum()))
    if mc < 2:
        # too few minority examples to stratify a CV split; skip the search and use a strong fixed
        # prior (C=0.03, near the regularized end of the grid) rather than the under-regularizing default
        return LogisticRegression(max_iter=2000, C=0.03, class_weight=class_weight).fit(Xtr, ytr)
    # log-spaced 1e-4..1e2 grid, ROC-AUC-scored on inner stratified folds; cap folds at 5 but shrink to
    # mc when the minority class is thin so every fold still sees both labels
    return LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=min(5, mc), scoring="roc_auc",
                                max_iter=2000, class_weight=class_weight).fit(Xtr, ytr)
