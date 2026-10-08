import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from validation import create_logistic_model, create_cv


BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR / "src"))

from classification import (
    load_data,
    create_target,
    prepare_data
)


from preprocessing.ml_preprocessing import (
    create_gridsearch_preprocessor
)

def create_grid_search(model, cv):
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    }

    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

def main():
    df = load_data()
    df, _ = create_target(df)

    X, y = prepare_data(df)

    preprocessor = create_gridsearch_preprocessor()

    model = create_logistic_model(preprocessor)

    cv = create_cv()

    grid_search = create_grid_search(model, cv)

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best F1 score:", grid_search.best_score_)


if __name__ == "__main__":
    main()

