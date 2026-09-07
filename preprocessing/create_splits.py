from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "segments_manifest.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "dataset_manifest.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING RECORDING-LEVEL DATASET SPLIT")
    print("=" * 70)

    print("\nReading manifest:")
    print(MANIFEST_FILE)

    df = pd.read_csv(MANIFEST_FILE)

    print(f"\nTotal segments: {len(df)}")

    # --------------------------------------------------------
    # Get unique original recordings
    # --------------------------------------------------------

    recordings = (
        df[
            [
                "recording_id",
                "scientific_name"
            ]
        ]
        .drop_duplicates()
    )

    print(
        f"Unique recordings: "
        f"{len(recordings)}"
    )

    # --------------------------------------------------------
    # First split: 70% train, 30% temporary
    # --------------------------------------------------------

    train_recordings, temp_recordings = train_test_split(
        recordings,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=recordings["scientific_name"]
    )

    # --------------------------------------------------------
    # Second split: temporary → validation + test
    # --------------------------------------------------------

    relative_test_ratio = (
        TEST_RATIO
        / (VAL_RATIO + TEST_RATIO)
    )

    val_recordings, test_recordings = train_test_split(
        temp_recordings,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temp_recordings["scientific_name"]
    )

    # --------------------------------------------------------
    # Create recording → split mapping
    # --------------------------------------------------------

    split_mapping = {}

    for recording_id in train_recordings["recording_id"]:
        split_mapping[recording_id] = "train"

    for recording_id in val_recordings["recording_id"]:
        split_mapping[recording_id] = "val"

    for recording_id in test_recordings["recording_id"]:
        split_mapping[recording_id] = "test"

    # --------------------------------------------------------
    # Assign split to every segment
    # --------------------------------------------------------

    df["split"] = df["recording_id"].map(split_mapping)

    # Safety check
    if df["split"].isna().any():
        raise ValueError(
            "Some recordings were not assigned to a split."
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print("\nRecording distribution:")

    print(
        f"Train      : "
        f"{len(train_recordings)} recordings"
    )

    print(
        f"Validation : "
        f"{len(val_recordings)} recordings"
    )

    print(
        f"Test       : "
        f"{len(test_recordings)} recordings"
    )

    print("\nSegment distribution:")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    print("\nSpecies distribution:")

    print(
        pd.crosstab(
            df["scientific_name"],
            df["split"]
        )
        .to_string()
    )

    print("\nDataset manifest saved to:")

    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()