import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = BASE_DIR / "data" / "raw" / "movies.json"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "movies_clean.json"


def load_data():
    with open(RAW_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


def inspect_data(df):
    print("BEFORE CLEANING")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate movie IDs:", df["movie_id"].duplicated().sum())


def clean_text(value):
    if not isinstance(value, str):
        return ""

    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_names(value):
    if not isinstance(value, list):
        return []

    names = []

    for item in value:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"])
        elif isinstance(item, str):
            names.append(item)

    return names


def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates(subset=["movie_id"])

    text_columns = [
        "title",
        "overview",
        "original_language"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(extract_names)

    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(extract_names)

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(
            df["release_date"],
            errors="coerce"
        )

    numeric_columns = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    for column in ["runtime", "budget", "revenue"]: 
       if column in df.columns: 
        df[column] = df[column].replace(0, pd.NA)

    for column in ["popularity", "vote_average", "vote_count"]:
        if column in df.columns:
                df[column] = df[column].fillna(0)
 
 
    df["title"] = df["title"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["original_language"] = df["original_language"].fillna("")

    df["genres"] = df["genres"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    df["keywords"] = df["keywords"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return df


def save_data(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = df.to_dict(orient="records")

    for movie in data:
        if pd.notna(movie.get("release_date")):
            movie["release_date"] = movie["release_date"].strftime("%Y-%m-%d")
        else:
            movie["release_date"] = None

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")


def main():
    df = load_data()

    inspect_data(df)

    df = clean_data(df)

    print("\nAFTER CLEANING")
    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate movie IDs:", df["movie_id"].duplicated().sum())

    save_data(df)


if __name__ == "__main__":
    main()