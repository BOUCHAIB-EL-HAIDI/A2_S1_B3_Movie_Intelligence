from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def task_extract_data(**context):
    raw_file = PROJECT_ROOT / "data" / "raw" / "movies.json"
    pages = int(os.getenv("TMDB_PAGES_TO_EXTRACT", "5"))

    print(f"Starting data extraction (target pages: {pages})...")
    try:
        from src.extraction.tmdb_api import extract_movies
        extract_movies(number_of_pages=pages)
        print("Extraction completed successfully.")
    except Exception as e:
        if raw_file.exists():
            print(f"Extraction encountered an issue ({e}), using existing raw data at {raw_file}.")
        else:
            raise RuntimeError(f"Extraction failed and no existing raw data found: {e}") from e


def task_clean_data(**context):
    print("Starting data cleaning...")
    from src.preprocessing.cleaning import main as clean_main
    clean_main()
    print("Data cleaning completed successfully.")


def task_feature_engineering(**context):
    print("Starting feature engineering...")
    from src.features.feature_engineering import main as features_main
    features_main()
    print("Feature engineering completed successfully.")


def task_store_mongodb(**context):
    print("Starting MongoDB storage...")
    from src.database import mongodb

    mongo_host = os.getenv("MONGODB_HOST", "mongodb" if os.path.exists("/.dockerenv") else "localhost")
    mongo_port = os.getenv("MONGODB_PORT", "27017")
    mongodb.MONGODB_URI = os.getenv("MONGODB_URI", f"mongodb://{mongo_host}:{mongo_port}")
    print(f"Connecting to MongoDB at: {mongodb.MONGODB_URI}")

    try:
        mongodb.insert_movies()
        print("Cleaned movies successfully inserted into MongoDB.")
    except Exception as e:
        print(f"Notice on insert_movies: {e}")

    try:
        mongodb.insert_features()
        print("Movie features successfully upserted into MongoDB.")
    except Exception as e:
        print(f"Notice on insert_features: {e}")

    print("MongoDB storage task finished.")


def task_train_classification_model(**context):
    print("Training and saving classification pipeline...")
    from src.classification.save_model import main as train_model_main
    train_model_main()
    print("Classification model saved successfully.")


def task_run_clustering(**context):
    print("Running KMeans financial clustering...")
    from src.clustering.clustering import main as clustering_main
    clustering_main()
    print("Clustering completed successfully.")


def task_run_nlp_tfidf(**context):
    print("Running NLP overview TF-IDF analysis...")
    from src.nlp.tfidf import main as tfidf_main
    tfidf_main()
    print("TF-IDF analysis completed successfully.")


default_args = {
    "owner": "movie_intelligence",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="movie_intelligence_pipeline",
    default_args=default_args,
    description="Pipeline ETL & Machine Learning pour Movie Intelligence",
    schedule="@daily",
    catchup=False,
    tags=["movie_intelligence", "etl", "machine_learning", "tmdb"],
) as dag:

    extract_op = PythonOperator(
        task_id="extract_tmdb_data",
        python_callable=task_extract_data,
    )

    clean_op = PythonOperator(
        task_id="clean_data",
        python_callable=task_clean_data,
    )

    features_op = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
    )

    mongodb_op = PythonOperator(
        task_id="store_in_mongodb",
        python_callable=task_store_mongodb,
    )

    train_classification_op = PythonOperator(
        task_id="train_classification_model",
        python_callable=task_train_classification_model,
    )

    clustering_op = PythonOperator(
        task_id="run_clustering",
        python_callable=task_run_clustering,
    )

    nlp_tfidf_op = PythonOperator(
        task_id="run_nlp_tfidf",
        python_callable=task_run_nlp_tfidf,
    )

    extract_op >> clean_op >> features_op >> mongodb_op >> [train_classification_op, clustering_op, nlp_tfidf_op]
