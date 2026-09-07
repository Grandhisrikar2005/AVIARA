from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SPECTROGRAM_DIR = (
    BASE_DIR
    / "dataset"
    / "spectrograms"
)


# ============================================================
# COMMUNICATION LABEL MAPPING
# ============================================================

COMMUNICATION_TO_IDX = {
    "alarm_warning": 0,
    "begging_distress": 1,
    "contact_social": 2,
    "song": 3,
}

IDX_TO_COMMUNICATION = {
    value: key
    for key, value in COMMUNICATION_TO_IDX.items()
}


# ============================================================
# DATASET
# ============================================================

class CommunicationDataset(Dataset):
    """
    Dataset for communication/vocalization classification.

    Each sample consists of:
        Mel spectrogram -> communication class
    """

    def __init__(
        self,
        manifest_path,
        spectrogram_dir=None,
        communication_to_idx=None,
    ):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found:\n{self.manifest_path}"
            )

        self.df = pd.read_csv(
            self.manifest_path
        )

        if self.df.empty:
            raise ValueError(
                f"Manifest is empty:\n{self.manifest_path}"
            )

        self.spectrogram_dir = Path(
            spectrogram_dir
            if spectrogram_dir is not None
            else SPECTROGRAM_DIR
        )

        self.communication_to_idx = (
            communication_to_idx
            if communication_to_idx is not None
            else COMMUNICATION_TO_IDX
        )

        self._validate_manifest()

        self.samples = []

        self._build_samples()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_manifest(self):

        required_columns = {
            "recording_id",
            "species",
            "scientific_name",
            "communication_type",
            "segment_file",
        }

        missing = (
            required_columns
            - set(self.df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing)}"
            )

        invalid_labels = sorted(
            set(
                self.df["communication_type"]
            )
            - set(
                self.communication_to_idx.keys()
            )
        )

        if invalid_labels:
            raise ValueError(
                "Unknown communication labels: "
                f"{invalid_labels}"
            )

    # ========================================================
    # FIND SPECTROGRAM
    # ========================================================

    def _find_spectrogram(
        self,
        segment_file,
    ):
        """
        Resolve a segment WAV path into its corresponding
        precomputed spectrogram .npy file.
        """

        segment_path = Path(
            str(segment_file)
        )

        filename = segment_path.name

        # ----------------------------------------------------
        # Segment filenames are typically:
        # XC123456_0001.wav
        #
        # Spectrogram:
        # XC123456_0001.npy
        # ----------------------------------------------------

        candidates = []

        candidates.append(
            self.spectrogram_dir
            / f"{filename}.npy"
        )

        candidates.append(
            self.spectrogram_dir
            / f"{segment_path.stem}.npy"
        )

        # Search recursively
        if self.spectrogram_dir.exists():

            for candidate in candidates:

                if candidate.exists():
                    return candidate

            matches = list(
                self.spectrogram_dir.rglob(
                    f"{segment_path.stem}.npy"
                )
            )

            if matches:
                return matches[0]

        return None

    # ========================================================
    # BUILD SAMPLE INDEX
    # ========================================================

    def _build_samples(self):

        missing = []

        for _, row in self.df.iterrows():

            segment_file = row[
                "segment_file"
            ]

            spectrogram_path = (
                self._find_spectrogram(
                    segment_file
                )
            )

            if spectrogram_path is None:

                missing.append(
                    segment_file
                )

                continue

            label_name = row[
                "communication_type"
            ]

            label_index = (
                self.communication_to_idx[
                    label_name
                ]
            )

            self.samples.append(
                {
                    "spectrogram": spectrogram_path,
                    "label": label_index,
                    "label_name": label_name,
                    "recording_id": str(
                        row["recording_id"]
                    ),
                    "species": row["species"],
                    "scientific_name": row[
                        "scientific_name"
                    ],
                    "segment_file": segment_file,
                }
            )

        if missing:

            print(
                f"WARNING: {len(missing)} "
                f"spectrograms could not be found."
            )

            if len(missing) <= 10:

                for item in missing:
                    print(
                        f"  {item}"
                    )

        if not self.samples:

            raise RuntimeError(
                "No usable communication samples found."
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[
            index
        ]

        spectrogram = np.load(
            sample["spectrogram"]
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Expected shape:
        # [128, 501]
        # ----------------------------------------------------

        if spectrogram.ndim != 2:

            raise ValueError(
                f"Unexpected spectrogram shape: "
                f"{spectrogram.shape}"
            )

        # ----------------------------------------------------
        # Add channel dimension
        #
        # [128, 501]
        #      ↓
        # [1, 128, 501]
        # ----------------------------------------------------

        spectrogram = torch.from_numpy(
            spectrogram
        ).unsqueeze(0)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            spectrogram,
            label,
        )

    # ========================================================
    # SAMPLE METADATA
    # ========================================================

    def get_metadata(self, index):

        return self.samples[
            index
        ]


# ============================================================
# DATASET TEST
# ============================================================

def main():

    print("=" * 70)
    print("COMMUNICATION DATASET TEST")
    print("=" * 70)

    train_manifest = (
        BASE_DIR
        / "dataset"
        / "metadata"
        / "communication_train_manifest.csv"
    )

    val_manifest = (
        BASE_DIR
        / "dataset"
        / "metadata"
        / "communication_val_manifest.csv"
    )

    test_manifest = (
        BASE_DIR
        / "dataset"
        / "metadata"
        / "communication_test_manifest.csv"
    )

    train_dataset = CommunicationDataset(
        train_manifest
    )

    val_dataset = CommunicationDataset(
        val_manifest
    )

    test_dataset = CommunicationDataset(
        test_manifest
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    print()
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

    print(
        f"Total samples : "
        f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    print()
    print("Communication mapping:")

    for idx, label in sorted(
        IDX_TO_COMMUNICATION.items()
    ):
        print(
            f"{idx}: {label}"
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print("TRAIN DISTRIBUTION")

    train_labels = [
        sample["label_name"]
        for sample in train_dataset.samples
    ]

    print(
        pd.Series(
            train_labels
        )
        .value_counts()
        .to_string()
    )

    print()
    print("VALIDATION DISTRIBUTION")

    val_labels = [
        sample["label_name"]
        for sample in val_dataset.samples
    ]

    print(
        pd.Series(
            val_labels
        )
        .value_counts()
        .to_string()
    )

    print()
    print("TEST DISTRIBUTION")

    test_labels = [
        sample["label_name"]
        for sample in test_dataset.samples
    ]

    print(
        pd.Series(
            test_labels
        )
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Inspect one sample
    # --------------------------------------------------------

    spectrogram, label = (
        train_dataset[0]
    )

    print()
    print("SAMPLE TEST")
    print("-" * 70)

    print(
        f"Spectrogram shape : "
        f"{tuple(spectrogram.shape)}"
    )

    print(
        f"Label index       : "
        f"{label.item()}"
    )

    print(
        f"Label name        : "
        f"{IDX_TO_COMMUNICATION[label.item()]}"
    )

    print(
        f"Spectrogram dtype : "
        f"{spectrogram.dtype}"
    )

    print(
        f"Spectrogram min   : "
        f"{spectrogram.min().item():.6f}"
    )

    print(
        f"Spectrogram max   : "
        f"{spectrogram.max().item():.6f}"
    )

    # --------------------------------------------------------
    # DataLoader test
    # --------------------------------------------------------

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    batch_x, batch_y = next(
        iter(loader)
    )

    print()
    print("DATALOADER TEST")
    print("-" * 70)

    print(
        f"Batch spectrogram shape : "
        f"{tuple(batch_x.shape)}"
    )

    print(
        f"Batch label shape       : "
        f"{tuple(batch_y.shape)}"
    )

    print(
        f"Batch label values      : "
        f"{batch_y.tolist()}"
    )

    print()
    print("=" * 70)
    print("COMMUNICATION DATASET TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()