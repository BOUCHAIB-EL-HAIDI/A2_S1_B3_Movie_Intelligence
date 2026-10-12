
import json
from pathlib import Path

from pymongo import MongoClient, UpdateOne

BASE_DIR = Path(__file__).resolve().parents[2]

CLEAN_FILE = BASE_DIR / "data" / "processed" / "movies_clean.json"
FEATURES_FILE = BASE_DIR / "data" / "processed" / "movies_features.json"

MONGODB_URI = "mongodb://localhost:27017"


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def insert_movies():
    movies = load_data(CLEAN_FILE)

    client = MongoClient(MONGODB_URI)

    try:
        collection = client["movie_database"]["movies"]
        collection.insert_many(movies)
        print(f"Inserted cleaned movies: {len(movies)}")
    finally:
        client.close()


def insert_features():
    movies = load_data(FEATURES_FILE)

    if not movies:
        print("No features to insert.")
        return

    client = MongoClient(MONGODB_URI)

    try:
        collection = client["movie_database"]["movies_features"]

        operations = []

        for movie in movies:
            movie_id = movie.get("movie_id")

            if movie_id is None:
                continue

            operations.append(
                UpdateOne(
                    {"movie_id": movie_id},
                    {"$set": movie},
                    upsert=True,
                )
            )

        if operations:
            result = collection.bulk_write(operations, ordered=False)

            print(f"Feature records processed: {len(operations)}")
            print(f"New feature documents inserted: {result.upserted_count}")
            print(f"Existing feature documents modified: {result.modified_count}")
            print(f"Total feature documents: {collection.count_documents({})}")

    finally:
        client.close()


if __name__ == "__main__":
    insert_features()
