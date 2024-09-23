from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    display
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": ((tp + tn) / (tp + tn + fp + fn + 1e-6)),
        "recall": (tp / (tp + fn + 1e-6)),
        "precision": (tp / (tp + fp + 1e-6)),
    }
    return metrics, display
