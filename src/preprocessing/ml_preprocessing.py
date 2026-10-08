import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

NUMERICAL_FEATURES = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "release_year",
    "release_month",
    "genre_count",
    "keyword_count"
]

CATEGORICAL_FEATURES = [
    "original_language",
    "runtime_category"
]

TEXT_FEATURE = "overview"





def create_numerical_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

def create_categorical_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

def create_text_pipeline():
    return TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english"
    )

def create_preprocessor():
    numerical_pipeline = create_numerical_pipeline()
    categorical_pipeline = create_categorical_pipeline()
    text_pipeline = create_text_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("text", text_pipeline, TEXT_FEATURE)
        ],
        remainder="passthrough"
    )

    return preprocessor


# this is for handling the genres and keywords because they are lists so we use multi-hotencoding


def apply_multi_hot_encoding(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    genre_encoder = MultiLabelBinarizer()
    keyword_encoder = MultiLabelBinarizer()

    train_genres = genre_encoder.fit_transform(X_train["genres"])

    genre_classes = set(genre_encoder.classes_)

    test_genres = genre_encoder.transform(
        X_test["genres"].apply(
            lambda items: [
                item for item in items
                if item in genre_classes
            ]
        )
    )

    train_keywords = keyword_encoder.fit_transform(X_train["keywords"])

    keyword_classes = set(keyword_encoder.classes_)

    test_keywords = keyword_encoder.transform(
        X_test["keywords"].apply(
            lambda items: [
                item for item in items
                if item in keyword_classes
            ]
        )
    )

    genre_columns = [
        f"genre_{genre}"
        for genre in genre_encoder.classes_
    ]

    keyword_columns = [
        f"kw_{keyword}"
        for keyword in keyword_encoder.classes_
    ]

    train_genres = pd.DataFrame(
        train_genres,
        columns=genre_columns,
        index=X_train.index
    )

    test_genres = pd.DataFrame(
        test_genres,
        columns=genre_columns,
        index=X_test.index
    )

    train_keywords = pd.DataFrame(
        train_keywords,
        columns=keyword_columns,
        index=X_train.index
    )

    test_keywords = pd.DataFrame(
        test_keywords,
        columns=keyword_columns,
        index=X_test.index
    )

    X_train = X_train.drop(columns=["genres", "keywords"])
    X_test = X_test.drop(columns=["genres", "keywords"])

    X_train = pd.concat(
        [X_train, train_genres, train_keywords],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_genres, test_keywords],
        axis=1
    )

    return X_train, X_test







#this is for hyperparameter tuning 

def list_to_text(X):
    return X.iloc[:, 0].apply(
        lambda items: " ".join(
            item.replace(" ", "_")
            for item in items
        )
    )


def create_gridsearch_preprocessor():
    numerical_pipeline = create_numerical_pipeline()
    categorical_pipeline = create_categorical_pipeline()

    overview_pipeline = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    genres_pipeline = Pipeline(
        steps=[
            ("to_text", FunctionTransformer(list_to_text, validate=False)),
            ("encoder", CountVectorizer(binary=True))
        ]
    )

    keywords_pipeline = Pipeline(
        steps=[
            ("to_text", FunctionTransformer(list_to_text, validate=False)),
            ("encoder", CountVectorizer(binary=True))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("overview", overview_pipeline, TEXT_FEATURE),
            ("genres", genres_pipeline, ["genres"]),
            ("keywords", keywords_pipeline, ["keywords"])
        ]
    )

    return preprocessor

