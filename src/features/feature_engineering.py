import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "movies_clean.json"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "movies_features.json"

def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


def create_date_features(df):
    df = df.copy()

    df["release_date"]=pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = (df["release_year"] // 10) * 10

    return df


def create_count_features(df):
    df=df.copy()

    df["genre_count"] = df["genres"].apply(len)
    df["keyword_count"] = df["keywords"].apply(len)

    return df 



def create_runtime_category(df):
    df = df.copy()

    df["runtime_category"] = pd.cut(
        df["runtime"],
        bins=[0, 90, 120, 150 , float("inf")],
        labels=["Short", "Medium", "Long", "Very Long"],
        include_lowest=True

    )

    return df

#remove outliers

def remove_outliers(df):
    df = df.copy()

    columns = [
        "runtime",
        "budget",
        "revenue",
        "popularity"
    ]

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df = df[
            df[column].isna() |
            (
                (df[column] >= lower_bound) &
                (df[column] <= upper_bound)
            )
        ]

    return df



def main():
    df = load_data()
    print("Before outlier removal:", df.shape)
    df = remove_outliers(df)
    print("After outlier removal:", df.shape)
    df = create_date_features(df)
    df = create_count_features(df)
    df = create_runtime_category(df)

    df["release_date"] = df["release_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None
    )

    data = df.to_dict(orient="records")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print("Feature Engineering completed.")
    print("Shape:", df.shape)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


