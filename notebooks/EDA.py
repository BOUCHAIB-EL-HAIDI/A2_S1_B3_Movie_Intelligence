import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE ="data/processed/movies_clean.json"

with open(INPUT_FILE , "r", encoding="utf-8") as file:
    data=json.load(file)

df = pd.DataFrame(data)

#Distribution des notes

def price_distrupution(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["vote_average"],
        bins=20,
        kde=True
    )

    plt.title("Distribution des notes des films")
    plt.xlabel("Note moyenne")
    plt.ylabel("Nombre de films") 

    plt.tight_layout()
    plt.show()

# La distribution des notes est principalement concentrée entre 6 et 8,
# avec une moyenne de 6,77 et une médiane de 6,87.
# La majorité des films obtiennent donc des notes relativement élevées.
# On observe également des films avec une note de 0 et un nombre de votes nul
# (vote_count = 0), ce qui correspond à des films pour lesquels aucune
# évaluation n'est disponible.
# Ces valeurs doivent être distinguées des véritables notes faibles lors
# des analyses statistiques et de la modélisation.



#Films par genre


def plot_movies_by_genre(df):
    genres = df.explode("genres")

    genre_counts = genres["genres"].value_counts().head(15)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=genre_counts.values,
        y=genre_counts.index
    )
    
    plt.title("Nombre de films par genre")
    plt.xlabel("Nombre de films")
    plt.ylabel("Genre")

    plt.tight_layout()
    plt.show()


# Le genre Drama est le plus représenté dans notre échantillon,
# suivi notamment par Action, Comedy et Thriller.
# Les genres Adventure, Science Fiction, Fantasy et Horror occupent
# une position intermédiaire, tandis que History et War sont moins représentés.
#
# Un même film peut appartenir à plusieurs genres. L'utilisation de
# explode() permet donc de compter chaque association film-genre séparément.
# Cette observation sera utile lors du Feature Engineering, notamment
# pour créer le nombre de genres par film.



#films par année de sortie


def plot_movies_by_year(df):
    data = df.copy()

    data["release_date"] = pd.to_datetime(
        data["release_date"],
        errors="coerce"
    )

    data["release_year"] = data["release_date"].dt.year

    year_counts = data["release_year"].value_counts().sort_index()

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        x=year_counts.index,
        y=year_counts.values
    )

    plt.title("Nombre de films par année de sortie")
    plt.xlabel("Année")
    plt.ylabel("Nombre de films")

    plt.tight_layout()
    plt.show()


# Les films sont peu nombreux dans les années anciennes et plus nombreux
# à partir des années 1990, surtout dans les années récentes.
# La baisse des dernières années peut être liée à des données encore partielles.
# Cette distribution montre un déséquilibre temporel et justifie la création
# de variables comme l'année et la décennie.


#Durée des films

def plot_runtime_distribution(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["runtime"],
        bins=30,
        kde=True
    )

    plt.title("Distribution de la durée des films")
    plt.xlabel("Durée (minutes)")
    plt.ylabel("Nombre de films")

    plt.tight_layout()
    plt.show()



# La durée est principalement comprise entre 90 et 130 minutes,
# avec un pic autour de 100 minutes.
# Quelques valeurs proches de 0 doivent être vérifiées.
# On observe aussi quelques films avec des durées très élevées.
# Ces résultats justifient la création de catégories de durée
# lors du Feature Engineering.


#Budget et revenus

def plot_budget_revenue(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(
        df["budget"],
        bins=30,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title("Distribution du budget")
    axes[0].set_xlabel("Budget")
    axes[0].set_ylabel("Nombre de films")

    sns.histplot(
        df["revenue"],
        bins=30,
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title("Distribution des revenus")
    axes[1].set_xlabel("Revenus")
    axes[1].set_ylabel("Nombre de films")

    plt.tight_layout()
    plt.show()




# Le budget et les revenus sont fortement asymétriques avec quelques valeurs
# très élevées.
# La majorité des films a un budget et des revenus faibles ou égaux à 0.
# Ces zéros peuvent représenter de petits montants ou des données non renseignées.
# Une transformation comme log1p() pourra être envisagée pour réduire l'effet
# des valeurs extrêmes.



#Votes vs popularité

def plot_votes_vs_popularity(df):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="vote_count",
        y="popularity"
    )

    plt.title("Relation entre le nombre de votes et la popularité")
    plt.xlabel("Nombre de votes")
    plt.ylabel("Popularité")

    plt.tight_layout()
    plt.show()

# La majorité des films possède peu de votes et une faible popularité.
# Certains films ont une forte popularité malgré peu de votes.
# Cela montre que vote_count et popularity sont deux indicateurs différents.
# Cette analyse sera utile pour définir high_engagement sans utiliser
# vote_count comme variable explicative afin d'éviter le data leakage.



def plot_numeric_boxplots(df):
    group_1 = [
        "runtime",
        "popularity",
        "vote_average"
    ]

    group_2 = [
        "budget",
        "revenue",
        "vote_count"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(
        data=df[group_1],
        ax=axes[0]
    )

    axes[0].set_title("Boxplots : Runtime, Popularity et Vote Average")
    axes[0].set_xlabel("Variables")
    axes[0].set_ylabel("Valeurs")

    sns.boxplot(
        data=df[group_2],
        ax=axes[1]
    )

    axes[1].set_title("Boxplots : Budget, Revenue et Vote Count")
    axes[1].set_xlabel("Variables")
    axes[1].set_ylabel("Valeurs")

    plt.tight_layout()
    plt.show()

#Graph 1 — Runtime, Popularity, Vote Average

# Runtime est principalement compris entre 90 et 125 minutes,
# avec quelques valeurs extrêmes très faibles ou très élevées.
# Popularity est fortement asymétrique avec une majorité de valeurs faibles
# et quelques valeurs très élevées.
# Vote Average est plus concentré, principalement entre 6 et 8.


#Graph 2 — Budget, Revenue, Vote Count

# Budget et Revenue présentent une forte asymétrie avec plusieurs valeurs
# extrêmes très élevées.
# Revenue possède les valeurs les plus élevées, atteignant plusieurs milliards.
# Budget présente également des valeurs extrêmes de plusieurs centaines de millions.
# Vote Count est difficile à observer correctement à cause de la différence
# d'échelle avec Budget et Revenue. Une visualisation séparée ou logarithmique
# pourra être utilisée.



def plot_correlation_heatmap(df):
    numeric_columns = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    correlation = df[numeric_columns].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title("Matrice de corrélation des variables numériques")
    plt.tight_layout()
    plt.show()






# Budget et Revenue présentent une corrélation positive relativement forte (0,68).
# Revenue et Vote Count présentent également une corrélation positive (0,64).
# Budget et Vote Count ont une corrélation modérée (0,45).
# Popularity présente de faibles corrélations avec les autres variables.
# La corrélation mesure une relation linéaire et ne signifie pas causalité.


def main():
    price_distrupution(df)
    plot_movies_by_genre(df)
    plot_movies_by_year(df)
    plot_runtime_distribution(df)
    plot_budget_revenue(df)
    plot_votes_vs_popularity(df)
    plot_numeric_boxplots(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()



