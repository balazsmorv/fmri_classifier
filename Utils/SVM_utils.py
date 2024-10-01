from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
import numpy as np


def train_svm(X, y, kernel, C):
    model = SVC(
        kernel=kernel, C=C, class_weight="balanced", random_state=42, verbose=True
    )
    model.fit(X=X, y=y)
    return model


def test_svm(model, x, y, L=None, b=None):
    if L is not None and b is not None:
        test_predictions = model.predict(x @ L + b)
    else:
        test_predictions = model.predict(x)
    cm = confusion_matrix(y, test_predictions, labels=model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    prec, rec, fbeta, supp = precision_recall_fscore_support(y, test_predictions)
    metrics = {
        "accuracy": np.count_nonzero(test_predictions == y) / len(test_predictions),
        "recall": rec,
        "precision": prec,
        "fscore": fbeta,
    }
    return metrics, display
