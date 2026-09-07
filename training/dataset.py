"""
AI BIO-ACOUSTIC BIRDSONG DATASET

Loads pre-generated mel spectrograms from
train / val / test folders.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BirdsongDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        spectrogram_dir,
        label_to_idx=None
    ):

        self.manifest_path = Path(manifest_path)
        self.spectrogram_dir = Path(spectrogram_dir)

        self.df = pd.read_csv(
            self.manifest_path
        )

        # ----------------------------------------------------
        # LABEL MAPPING
        # ----------------------------------------------------

        if label_to_idx is None:

            species_names = sorted(
                self.df["scientific_name"].unique()
            )

            self.label_to_idx = {
                species: index
                for index, species
                in enumerate(species_names)
            }

        else:

            self.label_to_idx = label_to_idx

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "segment_id",
            "scientific_name"
        ]

        for column in required_columns:

            if column not in self.df.columns:

                raise ValueError(
                    f"Required column missing: {column}"
                )

        # ----------------------------------------------------
        # REMOVE UNKNOWN SPECIES
        # ----------------------------------------------------

        self.df = self.df[
            self.df["scientific_name"].isin(
                self.label_to_idx
            )
        ].reset_index(drop=True)

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.df)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        row = self.df.iloc[index]

        segment_id = str(
            row["segment_id"]
        )

        # ----------------------------------------------------
        # Convert:
        #
        # segment_0000.wav
        #
        # into:
        #
        # segment_0000.npy
        # ----------------------------------------------------

        spectrogram_name = (
            Path(segment_id).stem + ".npy"
        )

        # ----------------------------------------------------
        # DETERMINE SPLIT
        # ----------------------------------------------------

        if "split" in row.index:

            split = str(
                row["split"]
            ).lower()

        else:

            split = None

        # ----------------------------------------------------
        # FIND SPECTROGRAM
        # ----------------------------------------------------

        possible_paths = []

        if split in {"train", "val", "test"}:

            possible_paths.append(
                self.spectrogram_dir
                / split
                / spectrogram_name
            )

        # Fallback to root directory
        possible_paths.append(
            self.spectrogram_dir
            / spectrogram_name
        )

        spectrogram_path = None

        for path in possible_paths:

            if path.exists():

                spectrogram_path = path
                break

        if spectrogram_path is None:

            raise FileNotFoundError(
                "Spectrogram not found.\n"
                f"Expected filename: "
                f"{spectrogram_name}\n"
                f"Split: {split}\n"
                f"Searched locations:\n"
                + "\n".join(
                    str(path)
                    for path in possible_paths
                )
            )

        # ----------------------------------------------------
        # LOAD SPECTROGRAM
        # ----------------------------------------------------

        spectrogram = np.load(
            spectrogram_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # VALIDATE SHAPE
        # ----------------------------------------------------

        if spectrogram.ndim != 2:

            raise ValueError(
                f"Expected 2D spectrogram, "
                f"got {spectrogram.shape}"
            )

        # [128, 501]
        spectrogram = torch.from_numpy(
            spectrogram
        )

        # [1, 128, 501]
        spectrogram = spectrogram.unsqueeze(0)

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        species = row[
            "scientific_name"
        ]

        label = torch.tensor(
            self.label_to_idx[species],
            dtype=torch.long
        )

        return spectrogram, label


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__).resolve().parent.parent
    )

    MANIFEST = (
        PROJECT_ROOT
        / "dataset"
        / "metadata"
        / "dataset_manifest.csv"
    )

    SPECTROGRAM_DIR = (
        PROJECT_ROOT
        / "dataset"
        / "spectrograms"
    )

    print("=" * 60)
    print("BIRDSONG DATASET TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    full_df = pd.read_csv(
        MANIFEST
    )

    train_df = full_df[
        full_df["split"] == "train"
    ].copy()

    val_df = full_df[
        full_df["split"] == "val"
    ].copy()

    test_df = full_df[
        full_df["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # GLOBAL LABEL MAPPING
    # --------------------------------------------------------

    species_names = sorted(
        full_df["scientific_name"].unique()
    )

    label_to_idx = {
        species: index
        for index, species
        in enumerate(species_names)
    }

    # --------------------------------------------------------
    # TEMPORARY MANIFESTS
    # --------------------------------------------------------

    temp_dir = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_manifest = (
        temp_dir
        / "_dataset_test_train.csv"
    )

    val_manifest = (
        temp_dir
        / "_dataset_test_val.csv"
    )

    test_manifest = (
        temp_dir
        / "_dataset_test_test.csv"
    )

    train_df.to_csv(
        train_manifest,
        index=False
    )

    val_df.to_csv(
        val_manifest,
        index=False
    )

    test_df.to_csv(
        test_manifest,
        index=False
    )

    # --------------------------------------------------------
    # CREATE DATASETS
    # --------------------------------------------------------

    train_dataset = BirdsongDataset(
        train_manifest,
        SPECTROGRAM_DIR,
        label_to_idx
    )

    val_dataset = BirdsongDataset(
        val_manifest,
        SPECTROGRAM_DIR,
        label_to_idx
    )

    test_dataset = BirdsongDataset(
        test_manifest,
        SPECTROGRAM_DIR,
        label_to_idx
    )

    print(
        f"Train samples : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples  : "
        f"{len(test_dataset)}"
    )

    print("\nSpecies mapping:")

    for species, index in label_to_idx.items():

        print(
            f"{index}: {species}"
        )

    # --------------------------------------------------------
    # TEST TRAIN SAMPLE
    # --------------------------------------------------------

    train_spec, train_label = (
        train_dataset[0]
    )

    print("\nTraining sample:")
    print(
        f"Spectrogram shape : "
        f"{train_spec.shape}"
    )
    print(
        f"Label             : "
        f"{train_label.item()}"
    )

    # --------------------------------------------------------
    # TEST VALIDATION SAMPLE
    # --------------------------------------------------------

    val_spec, val_label = (
        val_dataset[0]
    )

    print("\nValidation sample:")
    print(
        f"Spectrogram shape : "
        f"{val_spec.shape}"
    )
    print(
        f"Label             : "
        f"{val_label.item()}"
    )

    # --------------------------------------------------------
    # TEST TEST SAMPLE
    # --------------------------------------------------------

    test_spec, test_label = (
        test_dataset[0]
    )

    print("\nTest sample:")
    print(
        f"Spectrogram shape : "
        f"{test_spec.shape}"
    )
    print(
        f"Label             : "
        f"{test_label.item()}"
    )

    print(
        f"Data type         : "
        f"{test_spec.dtype}"
    )

    print(
        "\nDataset loading successful!"
    )