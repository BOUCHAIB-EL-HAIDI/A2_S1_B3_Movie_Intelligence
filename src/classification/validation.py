import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR / "src"))

INPUT_FILE = BASE_DIR / "data" / "processed" / "movies_features.json"

from classification import (
    load_data,
    create_target,
    prepare_data
)

from preprocessing.ml_preprocessing import (
    apply_multi_hot_encoding,
    create_preprocessor
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def create_logistic_model(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

def prepare_validation_data():
    df = load_data()
    df, _ = create_target(df)
    X, y = prepare_data(df)

    return X, y

def create_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )



#valitdation function

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def validate_model(X, y):
    cv = create_cv()

    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": []
    }

    for train_index, valid_index in cv.split(X, y):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]

        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        X_train, X_valid = apply_multi_hot_encoding(
            X_train,
            X_valid
        )

        preprocessor = create_preprocessor()

        model = create_logistic_model(preprocessor)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        y_score = model.predict_proba(X_valid)[:, 1]

        results["accuracy"].append(
            accuracy_score(y_valid, y_pred)
        )

        results["precision"].append(
            precision_score(y_valid, y_pred)
        )

        results["recall"].append(
            recall_score(y_valid, y_pred)
        )

        results["f1"].append(
            f1_score(y_valid, y_pred)
        )

        results["roc_auc"].append(
            roc_auc_score(y_valid, y_score)
        )

    for metric, scores in results.items():
        print(
            f"{metric}: "
            f"{sum(scores) / len(scores):.3f}"
        )


if __name__ == "__main__":
    X, y = prepare_validation_data()
    validate_model(X, y)



