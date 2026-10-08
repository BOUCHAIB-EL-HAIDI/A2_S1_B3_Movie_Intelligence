import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_DIR = Path("data/raw")


if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is not defined in .env")


def get_movies(page):
    url = f"{BASE_URL}/discover/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "page": page,
        "language": "en-US",
        "sort_by": "popularity.desc"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        raise ValueError(f"No movies returned for page {page}")

    return data["results"]


def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "append_to_response": "keywords"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def extract_movies(number_of_pages=150):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    movie_ids = []

    for page in range(1, number_of_pages + 1):
        print(f"Getting movie IDs: page {page}/{number_of_pages}")

        try:
            movies = get_movies(page)
            movie_ids.extend(movie["id"] for movie in movies)
            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"HTTP error on page {page}: {e}")

        except ValueError as e:
            print(f"Data error on page {page}: {e}")

    movie_ids = list(dict.fromkeys(movie_ids))

    print(f"Unique movie IDs: {len(movie_ids)}")

    all_movies = []

    for index, movie_id in enumerate(movie_ids, start=1):
        print(f"Getting details: {index}/{len(movie_ids)}")

        try:
            movie = get_movie_details(movie_id)

            all_movies.append({
                "movie_id": movie.get("id"),
                "title": movie.get("title"),
                "overview": movie.get("overview"),
                "release_date": movie.get("release_date"),
                "runtime": movie.get("runtime"),
                "original_language": movie.get("original_language"),
                "genres": movie.get("genres", []),
                "keywords": movie.get("keywords", {}).get("keywords", []),
                "budget": movie.get("budget"),
                "revenue": movie.get("revenue"),
                "popularity": movie.get("popularity"),
                "vote_average": movie.get("vote_average"),
                "vote_count": movie.get("vote_count")
            })

            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"HTTP error for movie {movie_id}: {e}")

    output_file = OUTPUT_DIR / "movies.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_movies, file, ensure_ascii=False, indent=2)

    print(f"Extracted movies: {len(all_movies)}")
    print(f"Saved to: {output_file}")
    


if __name__ == "__main__":
    extract_movies(150)