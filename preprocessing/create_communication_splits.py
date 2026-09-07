from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_manifest.csv"
)

TRAIN_FILE = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_train_manifest.csv"
)

VAL_FILE = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_val_manifest.csv"
)

TEST_FILE = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_test_manifest.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MIN_RECORDINGS_FOR_STRATIFICATION = 3


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING COMMUNICATION DATASET SPLITS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Communication manifest not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Total segments loaded : {len(df)}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = {
        "recording_id",
        "species",
        "scientific_name",
        "communication_type",
        "segment_file",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Verify every recording has exactly one label
    # --------------------------------------------------------

    recording_labels = (
        df.groupby("recording_id")
        ["communication_type"]
        .nunique()
    )

    inconsistent = recording_labels[
        recording_labels > 1
    ]

    if len(inconsistent) > 0:

        print()
        print(
            "ERROR: Some recordings contain "
            "multiple communication labels."
        )

        print(inconsistent.to_string())

        raise ValueError(
            "A recording has multiple communication labels."
        )

    # --------------------------------------------------------
    # One row per original recording
    # --------------------------------------------------------

    recordings = (
        df[
            [
                "recording_id",
                "species",
                "scientific_name",
                "communication_type",
            ]
        ]
        .drop_duplicates(
            subset=["recording_id"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique recordings      : {len(recordings)}"
    )

    print()
    print("RECORDINGS BY COMMUNICATION TYPE")
    print("-" * 70)

    recording_class_counts = (
        recordings[
            "communication_type"
        ]
        .value_counts()
    )

    print(
        recording_class_counts.to_string()
    )

    # ========================================================
    # HANDLE RARE CLASSES
    # ========================================================

    rare_records = recordings[
        recordings["communication_type"]
        .map(recording_class_counts)
        < MIN_RECORDINGS_FOR_STRATIFICATION
    ].copy()

    stratifiable_records = recordings[
        recordings["communication_type"]
        .map(recording_class_counts)
        >= MIN_RECORDINGS_FOR_STRATIFICATION
    ].copy()

    print()
    print(
        f"Rare-class recordings kept in train: "
        f"{len(rare_records)}"
    )

    if not rare_records.empty:

        print(
            rare_records[
                [
                    "recording_id",
                    "species",
                    "communication_type",
                ]
            ].to_string(index=False)
        )

    # ========================================================
    # STRATIFIED SPLIT OF COMMON CLASSES
    # ========================================================

    train_records, temp_records = train_test_split(
        stratifiable_records,
        test_size=(
            VAL_RATIO + TEST_RATIO
        ),
        random_state=RANDOM_STATE,
        stratify=stratifiable_records[
            "communication_type"
        ],
    )

    # --------------------------------------------------------
    # Split temporary set into validation and test
    # --------------------------------------------------------

    relative_test_ratio = (
        TEST_RATIO
        / (VAL_RATIO + TEST_RATIO)
    )

    # Check that every class has enough samples
    # for the second stratified split.
    temp_counts = (
        temp_records[
            "communication_type"
        ]
        .value_counts()
    )

    can_stratify_second_split = (
        len(temp_counts) > 0
        and temp_counts.min() >= 2
    )

    if can_stratify_second_split:

        val_records, test_records = train_test_split(
            temp_records,
            test_size=relative_test_ratio,
            random_state=RANDOM_STATE,
            stratify=temp_records[
                "communication_type"
            ],
        )

    else:

        # Fallback for extremely small validation/test sets.
        val_records, test_records = train_test_split(
            temp_records,
            test_size=relative_test_ratio,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    # ========================================================
    # ADD RARE CLASSES TO TRAIN
    # ========================================================

    train_records = pd.concat(
        [
            train_records,
            rare_records,
        ],
        ignore_index=True,
    )

    # ========================================================
    # ASSIGN SEGMENTS
    # ========================================================

    def assign_split(
        recording_ids,
        split_name,
    ):

        result = df[
            df["recording_id"].isin(
                recording_ids
            )
        ].copy()

        result["split"] = split_name

        return result

    train_df = assign_split(
        train_records["recording_id"],
        "train",
    )

    val_df = assign_split(
        val_records["recording_id"],
        "val",
    )

    test_df = assign_split(
        test_records["recording_id"],
        "test",
    )

    # ========================================================
    # SAVE
    # ========================================================

    TRAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    val_df.to_csv(
        VAL_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("COMMUNICATION SPLIT CREATED")
    print("=" * 70)

    print()
    print("RECORDING COUNTS")
    print("-" * 70)

    print(
        f"Train recordings : {len(train_records)}"
    )

    print(
        f"Val recordings   : {len(val_records)}"
    )

    print(
        f"Test recordings  : {len(test_records)}"
    )

    print(
        f"Total recordings : {len(recordings)}"
    )

    print()
    print("SEGMENT COUNTS")
    print("-" * 70)

    print(
        f"Train segments : {len(train_df)}"
    )

    print(
        f"Val segments   : {len(val_df)}"
    )

    print(
        f"Test segments  : {len(test_df)}"
    )

    print(
        f"Total segments : {len(df)}"
    )

    # ========================================================
    # COMMUNICATION DISTRIBUTION
    # ========================================================

    print()
    print("COMMUNICATION DISTRIBUTION")
    print("-" * 70)

    distribution = pd.DataFrame(
        {
            "train": train_df[
                "communication_type"
            ].value_counts(),

            "val": val_df[
                "communication_type"
            ].value_counts(),

            "test": test_df[
                "communication_type"
            ].value_counts(),
        }
    ).fillna(0).astype(int)

    print(
        distribution.to_string()
    )

    # ========================================================
    # RECORDING-LEVEL DISTRIBUTION
    # ========================================================

    print()
    print("RECORDING-LEVEL DISTRIBUTION")
    print("-" * 70)

    recording_distribution = pd.DataFrame(
        {
            "train": train_records[
                "communication_type"
            ].value_counts(),

            "val": val_records[
                "communication_type"
            ].value_counts(),

            "test": test_records[
                "communication_type"
            ].value_counts(),
        }
    ).fillna(0).astype(int)

    print(
        recording_distribution.to_string()
    )

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    train_ids = set(
        train_df["recording_id"]
    )

    val_ids = set(
        val_df["recording_id"]
    )

    test_ids = set(
        test_df["recording_id"]
    )

    train_val_overlap = (
        train_ids & val_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    val_test_overlap = (
        val_ids & test_ids
    )

    print()
    print("LEAKAGE CHECK")
    print("-" * 70)

    print(
        f"Train ∩ Val  : {len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test : {len(train_test_overlap)}"
    )

    print(
        f"Val ∩ Test   : {len(val_test_overlap)}"
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):
        raise RuntimeError(
            "Recording leakage detected!"
        )

    print(
        "No recording-level leakage detected."
    )

    # ========================================================
    # RARE CLASS WARNING
    # ========================================================

    if not rare_records.empty:

        print()
        print("IMPORTANT DATASET NOTE")
        print("-" * 70)

        print(
            "Some communication classes have fewer than "
            f"{MIN_RECORDINGS_FOR_STRATIFICATION} "
            "original recordings."
        )

        print(
            "Those recordings were kept entirely in "
            "the training set."
        )

        print(
            "Their test performance cannot be considered "
            "a reliable held-out evaluation."
        )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("FILES CREATED")
    print("-" * 70)

    print(TRAIN_FILE)
    print(VAL_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()