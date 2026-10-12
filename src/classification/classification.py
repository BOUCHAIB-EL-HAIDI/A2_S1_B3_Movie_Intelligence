
import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))
INPUT_FILE = BASE_DIR / "data" / "processed" / "movies_features.json"

from preprocessing.ml_preprocessing import (
    apply_multi_hot_encoding,
    create_preprocessor
)



def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


def create_target(df):
    df = df.copy()
    threshold = df["vote_count"].quantile(0.60)
    df["high_engagement"] = (df["vote_count"] >= threshold).astype(int)

    return df, threshold


def prepare_data(df):
    y = df["high_engagement"]

    X = df.drop(
        columns=[
            "movie_id",
            "title",
            "release_date",
            "release_decade",
            "high_engagement",
            "vote_count",
        ]
    )

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

def prepare_features(X_train, X_test):
    X_train, X_test = apply_multi_hot_encoding(
        X_train,
        X_test
    )

    preprocessor = create_preprocessor()

    return X_train, X_test, preprocessor


def train_logistic_regression(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_linear_svm(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC())
        ]
    )

    model.fit(X_train, y_train)

    return model




def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_score):.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))




def analyze_target(df, threshold):
    print(f"Engagement threshold: {threshold:.0f} votes")
    print("\nTarget distribution:")
    print(df["high_engagement"].value_counts(normalize=True).sort_index())


def main():
    df = load_data()
    df, threshold = create_target(df)
    analyze_target(df, threshold)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, preprocessor = prepare_features(
        X_train,
        X_test
    )

    #logistic regression train and evaluate
    model = train_logistic_regression(
        X_train,
        y_train,
        preprocessor
    )

    evaluate_model(model, X_test, y_test)

    #random forest train and evaluate

    random_forest = train_random_forest(

        X_train,
        y_train,
        preprocessor
     )

    evaluate_model(
        random_forest,
        X_test,
        y_test
     )
    #linear svm train and evaluate

    linear_svm = train_linear_svm(
        X_train,
        y_train,
        preprocessor
    )

    evaluate_model(
        linear_svm,
        X_test,
        y_test
     )
    

    print(f"\nTraining set: {len(X_train)} movies")
    print(f"Test set: {len(X_test)} movies")
    print(f"Features: {X.shape[1]}")
    print(f"Target leakage check: {'vote_count' in X.columns}")

if __name__ == "__main__":
    main()
