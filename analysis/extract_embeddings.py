"""
V2 BIRDSONG EMBEDDING EXTRACTION

Extracts 128-dimensional acoustic embeddings from the
trained CNN + Transformer V2 model.

Outputs:
    outputs/embeddings/
        train_embeddings.csv
        val_embeddings.csv
        test_embeddings.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from training.model import BirdsongCNNTransformer
from training.dataset import BirdsongDataset


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "v2_best_model.pth"
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "embeddings"
)

BATCH_SIZE = 16


# ------------------------------------------------------------
# V2 LABEL MAPPING
# IMPORTANT:
# This MUST match the mapping used during V2 training.
# ------------------------------------------------------------

LABEL_TO_IDX = {
    "Eudynamys scolopaceus": 0,   # Asian Koel
    "Milvus migrans": 1,          # Black Kite
    "Acridotheres tristis": 2,     # Common Myna
    "Orthotomus sutorius": 3,      # Common Tailorbird
    "Centropus sinensis": 4,       # Greater Coucal
    "Corvus splendens": 5,         # House Crow
    "Pavo cristatus": 6,           # Indian Peafowl
    "Pycnonotus cafer": 7,         # Red-vented Bulbul
    "Psittacula krameri": 8,       # Rose-ringed Parakeet
    "Halcyon smyrnensis": 9        # White-throated Kingfisher
}


# ------------------------------------------------------------
# REVERSE LABEL MAPPING
# ------------------------------------------------------------

IDX_TO_SPECIES = {
    index: species
    for species, index in LABEL_TO_IDX.items()
}


# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------

def create_model():

    print("\nLoading V2 model...")

    model = BirdsongCNNTransformer(
        num_classes=10,
        cnn_dim=256,
        transformer_dim=256,
        num_heads=4,
        num_layers=3,
        embedding_dim=128,
        dropout=0.30
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # Handle both normal state_dict checkpoints
    # and checkpoints containing additional information.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully.")
    print(f"Device: {DEVICE}")

    return model


# ------------------------------------------------------------
# CREATE DATASET
# ------------------------------------------------------------

def create_dataset(split):

    print(f"\nLoading {split} dataset...")

    full_df = pd.read_csv(MANIFEST)

    split_df = full_df[
        full_df["split"] == split
    ].copy()

    if len(split_df) == 0:
        raise ValueError(
            f"No samples found for split: {split}"
        )

    # Temporary manifest containing only this split.
    temp_manifest = (
        OUTPUT_DIR
        / f"_embedding_{split}_manifest.csv"
    )

    split_df.to_csv(
        temp_manifest,
        index=False
    )

    dataset = BirdsongDataset(
        temp_manifest,
        SPECTROGRAM_DIR,
        LABEL_TO_IDX
    )

    print(f"{split.capitalize()} samples: {len(dataset)}")

    return dataset


# ------------------------------------------------------------
# EXTRACT EMBEDDINGS
# ------------------------------------------------------------

def extract_split(model, split):

    dataset = create_dataset(split)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    embeddings = []
    labels = []
    species_names = []
    segment_ids = []

    print(f"\nExtracting {split} embeddings...")

    with torch.no_grad():

        for batch_index, (spectrograms, batch_labels) in enumerate(loader):

            spectrograms = spectrograms.to(DEVICE)

            logits, batch_embeddings = model(
                spectrograms
            )

            batch_embeddings = (
                batch_embeddings
                .cpu()
                .numpy()
            )

            batch_labels = (
                batch_labels
                .cpu()
                .numpy()
            )

            embeddings.append(
                batch_embeddings
            )

            labels.append(
                batch_labels
            )

            # Recover metadata for this batch.
            start = batch_index * BATCH_SIZE
            end = start + len(batch_labels)

            batch_df = dataset.df.iloc[start:end]

            species_names.extend(
                batch_df["scientific_name"].tolist()
            )

            segment_ids.extend(
                batch_df["segment_id"].tolist()
            )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    # --------------------------------------------------------
    # BUILD OUTPUT DATAFRAME
    # --------------------------------------------------------

    data = {
        "segment_id": segment_ids,
        "scientific_name": species_names,
        "label": labels
    }

    # Add embedding_000 ... embedding_127
    for dimension in range(embeddings.shape[1]):

        data[
            f"embedding_{dimension:03d}"
        ] = embeddings[:, dimension]

    result = pd.DataFrame(data)

    output_file = (
        OUTPUT_DIR
        / f"{split}_embeddings.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return result


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("V2 BIRDSONG ACOUSTIC EMBEDDING EXTRACTION")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nCheckpoint:")
    print(CHECKPOINT)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"V2 checkpoint not found:\n{CHECKPOINT}"
        )

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Dataset manifest not found:\n{MANIFEST}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # EXTRACT ALL SPLITS
    # --------------------------------------------------------

    results = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        results[split] = extract_split(
            model,
            split
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 70)

    for split, dataframe in results.items():

        print(
            f"{split.capitalize():8s}: "
            f"{len(dataframe):4d} samples × "
            f"128 dimensions"
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(
            f"  {split}_embeddings.csv"
        )

    # --------------------------------------------------------
    # DELETE TEMPORARY MANIFESTS
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        temp_file = (
            OUTPUT_DIR
            / f"_embedding_{split}_manifest.csv"
        )

        if temp_file.exists():
            temp_file.unlink()

    print("\nTemporary files cleaned up.")

    print("=" * 70)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    main()