
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="Movie Intelligence",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def load_movies():
    path = DATA / "movies_features.json"
    if not path.exists():
        path = DATA / "movies_clean.json"

    if not path.exists():
        raise FileNotFoundError("No movie dataset was found.")

    df = pd.read_json(path)

    for col in ["genres", "keywords"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ", ".join(map(str, x))
                if isinstance(x, list)
                else str(x or "")
            )

    return df


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.title("🎬 Movie Intelligence")

try:
    movies = load_movies()
except Exception as e:
    st.error(f"Cannot load movie data: {e}")
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Classification", "Financial Clusters"],
)


if page == "Overview":
    st.header("Movie Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Movies", len(movies))
    col2.metric("Features", len(movies.columns))

    if "original_language" in movies.columns:
        col3.metric(
            "Languages",
            movies["original_language"].nunique(),
        )
    else:
        col3.metric("Languages", "N/A")

    if "vote_average" in movies.columns:
        st.subheader("Movie Ratings")
        st.bar_chart(
            movies["vote_average"].dropna().value_counts().sort_index()
        )

    if "genres" in movies.columns:
        st.subheader("Genres")
        genres = (
            movies["genres"]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
        )
        st.bar_chart(genres[genres != ""].value_counts().head(10))

    st.subheader("Dataset Preview")
    st.dataframe(movies.head(20), use_container_width=True)




elif page == "Classification":
    st.header("🎬 Movie Engagement")

    if "title" not in movies.columns or "vote_count" not in movies.columns:
        st.error("The dataset must contain title and vote_count columns.")
        st.stop()

    threshold = movies["vote_count"].quantile(0.60)

    movie_titles = sorted(
        movies["title"].dropna().astype(str).unique()
    )

    selected_title = st.selectbox(
        "Choose a movie",
        movie_titles,
    )

    if st.button("Check Engagement", type="primary"):
        movie = movies[movies["title"] == selected_title].iloc[0]
        high_engagement = int(movie["vote_count"] >= threshold)

        if high_engagement == 1:
            st.success("✅ High Engagement")
        else:
            st.info("🎬 Not High Engagement")



elif page == "Financial Clusters":
    st.header("Financial Clusters")

    path = DATA / "financial_clusters.csv"

    if not path.exists():
        st.warning(
            "financial_clusters.csv was not found in data/processed."
        )
    else:
        clusters = pd.read_csv(path)
        st.dataframe(clusters.head(100), use_container_width=True)

        cluster_col = next(
            (
                col
                for col in clusters.columns
                if "cluster" in col.lower()
            ),
            None,
        )

        if cluster_col:
            st.subheader("Movies per Cluster")
            st.bar_chart(
                clusters[cluster_col].value_counts().sort_index()
            )

        numeric_cols = clusters.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols, index=0)

            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=1)

            st.subheader("Financial Comparison")
            st.scatter_chart(
                clusters,
                x=x_axis,
                y=y_axis,
                color=cluster_col if cluster_col else None,
            )
