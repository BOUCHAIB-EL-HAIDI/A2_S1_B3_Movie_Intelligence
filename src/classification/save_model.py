import sys
from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR / "src"))

from classification import (
    load_data,
    create_target,
    prepare_data
)

from preprocessing.ml_preprocessing import (
    apply_multi_hot_encoding,
    create_preprocessor
)


def train_final_model(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=0.1,
                max_iter=1000
            ))
        ]
    )

    model.fit(X_train, y_train)

    return model


def save_model(model):
    model_path = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


def main():
    df = load_data()

    df, threshold = create_target(df)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_test = apply_multi_hot_encoding(
        X_train,
        X_test
    )

    preprocessor = create_preprocessor()

    model = train_final_model(
        X_train,
        y_train,
        preprocessor
    )

    save_model(model)


if __name__ == "__main__":
    main()