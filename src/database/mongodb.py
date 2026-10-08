import json
from pathlib import Path
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "movies_clean.json"





def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def insert_movies():
     movies = load_data()

     client = MongoClient("mongodb://localhost:27017")

     db = client["movie_database"]

     collection = db["movies"]

     collection.insert_many(movies)

     print(f"Inserted movies : {len(movies)}")

     client.close()


if __name__ == "__main__":
    insert_movies()
