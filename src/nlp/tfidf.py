import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "movies_features.json"


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def prepare_overview(df):
    df = df.copy()
    df["overview"] = df["overview"].fillna("")
    df["overview"] = df["overview"].apply(clean_text)
    return df

def apply_tfidf(df, max_features=3000, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(df["overview"])

    print("max_features:", max_features)
    print("ngram_range:", ngram_range)
    print("Number of movies:", matrix.shape[0])
    print("Number of features:", matrix.shape[1])
    print("Matrix dimensions:", matrix.shape)

    return vectorizer, matrix



# TF-IDF: convert movie overviews into numerical features for machine learning.
# Different max_features and ngram_range values are tested.
# The best configuration will be selected later based on model performance.



    
def show_top_terms(vectorizer, matrix, n=20):
    terms = vectorizer.get_feature_names_out()
    scores = matrix.mean(axis=0).A1
    top_indices = scores.argsort()[::-1][:n]

    for index in top_indices:
        print(terms[index], scores[index])



# Display the terms with the highest average TF-IDF scores.



def main():
    df = load_data()
    df = prepare_overview(df)

    vectorizer, matrix = apply_tfidf(
        df,
        max_features=3000,
        ngram_range=(1, 2)
    )

    show_top_terms(vectorizer, matrix)


if __name__ == "__main__":
    main()