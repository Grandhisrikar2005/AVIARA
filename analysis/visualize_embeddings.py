"""
V2 BIRDSONG EMBEDDING VISUALIZATION

Reduces the 128-dimensional V2 acoustic embeddings to 2D
using UMAP and creates a species-level acoustic-space plot.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "embeddings"
    / "test_embeddings.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "embeddings"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "umap_test_embeddings.png"
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42

N_NEIGHBORS = 15

MIN_DIST = 0.10


# ------------------------------------------------------------
# LOAD EMBEDDINGS
# ------------------------------------------------------------

def load_embeddings():

    print("\nLoading test embeddings...")

    if not EMBEDDING_FILE.exists():

        raise FileNotFoundError(
            f"Embedding file not found:\n{EMBEDDING_FILE}"
        )

    df = pd.read_csv(
        EMBEDDING_FILE
    )

    embedding_columns = [
        column
        for column in df.columns
        if column.startswith("embedding_")
    ]

    if len(embedding_columns) != 128:

        raise ValueError(
            f"Expected 128 embedding dimensions, "
            f"found {len(embedding_columns)}"
        )

    embeddings = df[
        embedding_columns
    ].values.astype(
        np.float32
    )

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Species: "
        f"{df['scientific_name'].nunique()}"
    )

    return df, embeddings


# ------------------------------------------------------------
# STANDARDIZE EMBEDDINGS
# ------------------------------------------------------------

def standardize_embeddings(embeddings):

    print("\nStandardizing embeddings...")

    scaler = StandardScaler()

    embeddings_scaled = scaler.fit_transform(
        embeddings
    )

    return embeddings_scaled


# ------------------------------------------------------------
# RUN UMAP
# ------------------------------------------------------------

def run_umap(embeddings):

    print("\nRunning UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE
    )

    embedding_2d = reducer.fit_transform(
        embeddings
    )

    print(
        f"UMAP output shape: "
        f"{embedding_2d.shape}"
    )

    return embedding_2d


# ------------------------------------------------------------
# CREATE VISUALIZATION
# ------------------------------------------------------------

def create_plot(df, embedding_2d):

    print("\nCreating UMAP visualization...")

    plt.figure(
        figsize=(12, 9)
    )

    species = df[
        "scientific_name"
    ].values

    unique_species = sorted(
        np.unique(species)
    )

    for species_name in unique_species:

        mask = (
            species == species_name
        )

        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            label=species_name,
            alpha=0.75,
            s=45
        )

    plt.title(
        "V2 CNN + Transformer Birdsong Acoustic Embedding Space",
        fontsize=15
    )

    plt.xlabel(
        "UMAP Dimension 1"
    )

    plt.ylabel(
        "UMAP Dimension 2"
    )

    plt.legend(
        title="Species",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=9
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved visualization:\n{OUTPUT_FILE}"
    )


# ------------------------------------------------------------
# SAVE UMAP COORDINATES
# ------------------------------------------------------------

def save_coordinates(df, embedding_2d):

    output_df = pd.DataFrame({
        "segment_id": df["segment_id"],
        "scientific_name": df["scientific_name"],
        "label": df["label"],
        "umap_1": embedding_2d[:, 0],
        "umap_2": embedding_2d[:, 1]
    })

    output_file = (
        OUTPUT_DIR
        / "test_umap_coordinates.csv"
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved coordinates:\n{output_file}"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("BIRDSONG ACOUSTIC EMBEDDING UMAP VISUALIZATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load
    df, embeddings = load_embeddings()

    # Standardize
    embeddings_scaled = standardize_embeddings(
        embeddings
    )

    # UMAP
    embedding_2d = run_umap(
        embeddings_scaled
    )

    # Plot
    create_plot(
        df,
        embedding_2d
    )

    # Save coordinates
    save_coordinates(
        df,
        embedding_2d
    )

    print("\n" + "=" * 70)
    print("UMAP ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated:"
    )

    print(
        "  umap_test_embeddings.png"
    )

    print(
        "  test_umap_coordinates.csv"
    )


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    main()