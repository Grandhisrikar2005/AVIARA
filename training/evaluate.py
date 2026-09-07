"""
AI BIO-ACOUSTIC BIRDSONG
V2 MODEL EVALUATION

Evaluates the best V2 CNN + Transformer model
on the completely held-out test set.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import BirdsongDataset
from model import BirdsongCNNTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "v2_best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMPORTANT:
# EXACT LABEL MAPPING USED DURING V2 TRAINING
# ============================================================

LABEL_TO_IDX = {
    "Eudynamys scolopaceus": 0,   # Asian Koel
    "Milvus migrans": 1,          # Black Kite
    "Acridotheres tristis": 2,    # Common Myna
    "Orthotomus sutorius": 3,     # Common Tailorbird
    "Centropus sinensis": 4,      # Greater Coucal
    "Corvus splendens": 5,        # House Crow
    "Pavo cristatus": 6,          # Indian Peafowl
    "Pycnonotus cafer": 7,        # Red-vented Bulbul
    "Psittacula krameri": 8,      # Rose-ringed Parakeet
    "Halcyon smyrnensis": 9       # White-throated Kingfisher
}


IDX_TO_COMMON_NAME = {
    0: "Asian Koel",
    1: "Black Kite",
    2: "Common Myna",
    3: "Common Tailorbird",
    4: "Greater Coucal",
    5: "House Crow",
    6: "Indian Peafowl",
    7: "Red-vented Bulbul",
    8: "Rose-ringed Parakeet",
    9: "White-throated Kingfisher"
}


IDX_TO_SCIENTIFIC_NAME = {
    index: species
    for species, index in LABEL_TO_IDX.items()
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI BIO-ACOUSTIC BIRDSONG V2 MODEL EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print("GPU available.")
    else:
        print("Running evaluation on CPU.")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    if not SPECTROGRAM_DIR.exists():
        raise FileNotFoundError(
            f"Spectrogram directory not found:\n"
            f"{SPECTROGRAM_DIR}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    print("\nLoading dataset manifest...")

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Total manifest rows: {len(manifest)}"
    )

    # --------------------------------------------------------
    # FILTER TEST SET
    # --------------------------------------------------------

    if "split" not in manifest.columns:
        raise ValueError(
            "The dataset manifest does not contain "
            "a 'split' column."
        )

    test_manifest = manifest[
        manifest["split"].astype(str).str.lower() == "test"
    ].copy()

    if len(test_manifest) == 0:
        raise ValueError(
            "No test samples found in dataset manifest."
        )

    temporary_test_manifest = (
        OUTPUT_DIR
        / "_v2_test_manifest.csv"
    )

    test_manifest.to_csv(
        temporary_test_manifest,
        index=False
    )

    print(
        f"Test samples: {len(test_manifest)}"
    )

    # --------------------------------------------------------
    # VERIFY SPECIES
    # --------------------------------------------------------

    test_species = set(
        test_manifest["scientific_name"]
    )

    unknown_species = (
        test_species
        - set(LABEL_TO_IDX.keys())
    )

    if unknown_species:

        raise ValueError(
            "Unknown species found in test set:\n"
            + "\n".join(
                sorted(unknown_species)
            )
        )

    # --------------------------------------------------------
    # CREATE TEST DATASET
    # --------------------------------------------------------

    print(
        "\nLoading held-out test dataset..."
    )

    test_dataset = BirdsongDataset(
        manifest_path=temporary_test_manifest,
        spectrogram_dir=SPECTROGRAM_DIR,
        label_to_idx=LABEL_TO_IDX
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"Dataset samples loaded: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # PRINT EXACT MAPPING
    # --------------------------------------------------------

    print("\nV2 training label mapping:")

    for index in range(10):

        print(
            f"{index}: "
            f"{IDX_TO_COMMON_NAME[index]} "
            f"({IDX_TO_SCIENTIFIC_NAME[index]})"
        )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print(
        "\nCreating V2 CNN + Transformer model..."
    )

    model = BirdsongCNNTransformer(
        num_classes=10,
        embedding_dim=128
    )

    model = model.to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # LOAD BEST V2 CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nLoading best V2 checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        best_epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        best_val_f1 = checkpoint.get(
            "best_val_f1",
            None
        )

        print(
            f"Checkpoint epoch: {best_epoch}"
        )

        if best_val_f1 is not None:

            print(
                f"Best validation Macro-F1: "
                f"{best_val_f1:.4f}"
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "V2 checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print(
        "\nRunning inference on held-out test set..."
    )

    print("-" * 70)

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for spectrograms, labels in test_loader:

            spectrograms = spectrograms.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            output = model(
                spectrograms
            )

            # V2 normally returns logits.
            # Also support (logits, embedding).

            if isinstance(output, tuple):

                logits = output[0]

            else:

                logits = output

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print(
        f"\nPredictions generated: "
        f"{len(y_pred)}"
    )

    print(
        f"Ground-truth labels: "
        f"{len(y_true)}"
    )

    print(
        "\nPredicted class distribution:"
    )

    unique_predictions, prediction_counts = np.unique(
        y_pred,
        return_counts=True
    )

    for index, count in zip(
        unique_predictions,
        prediction_counts
    ):

        print(
            f"  {index}: "
            f"{IDX_TO_COMMON_NAME[int(index)]} "
            f"-> {count}"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # MAIN RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("V2 TEST RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy              : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Balanced Accuracy     : "
        f"{balanced_accuracy:.4f} "
        f"({balanced_accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision       : "
        f"{macro_precision:.4f} "
        f"({macro_precision * 100:.2f}%)"
    )

    print(
        f"Macro Recall          : "
        f"{macro_recall:.4f} "
        f"({macro_recall * 100:.2f}%)"
    )

    print(
        f"Macro F1              : "
        f"{macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted F1           : "
        f"{weighted_f1:.4f} "
        f"({weighted_f1 * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(10)),
        target_names=[
            IDX_TO_COMMON_NAME[i]
            for i in range(10)
        ],
        zero_division=0,
        digits=4
    )

    print("\n")
    print("=" * 70)
    print("PER-SPECIES CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(10))
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            IDX_TO_COMMON_NAME[i]
            for i in range(10)
        ],
        columns=[
            IDX_TO_COMMON_NAME[i]
            for i in range(10)
        ]
    )

    confusion_df.index.name = "Actual"

    confusion_path = (
        OUTPUT_DIR
        / "v2_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "v2_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI BIO-ACOUSTIC BIRDSONG "
            "V2 MODEL EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Model: CNN + Transformer V2\n"
        )

        file.write(
            f"Checkpoint: {CHECKPOINT_PATH}\n"
        )

        file.write(
            f"Test samples: "
            f"{len(test_dataset)}\n\n"
        )

        file.write(
            "Metrics\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy:.4f} "
            f"({accuracy * 100:.2f}%)\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4f} "
            f"({balanced_accuracy * 100:.2f}%)\n"
        )

        file.write(
            f"Macro Precision: "
            f"{macro_precision:.4f} "
            f"({macro_precision * 100:.2f}%)\n"
        )

        file.write(
            f"Macro Recall: "
            f"{macro_recall:.4f} "
            f"({macro_recall * 100:.2f}%)\n"
        )

        file.write(
            f"Macro F1: "
            f"{macro_f1:.4f} "
            f"({macro_f1 * 100:.2f}%)\n"
        )

        file.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f} "
            f"({weighted_f1 * 100:.2f}%)\n\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "PER-SPECIES CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(report)

    # --------------------------------------------------------
    # SAVE JSON RESULTS
    # --------------------------------------------------------

    results = {
        "model": "CNN + Transformer V2",
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "best_epoch": (
            int(best_epoch)
            if isinstance(best_epoch, (int, np.integer))
            else str(best_epoch)
        ),
        "test_samples": int(
            len(test_dataset)
        ),
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "label_mapping": {
            str(index): IDX_TO_COMMON_NAME[index]
            for index in range(10)
        }
    }

    results_path = (
        OUTPUT_DIR
        / "v2_test_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("V2 EVALUATION COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print(
        f"\n  Classification report:"
        f"\n  {report_path}"
    )

    print(
        f"\n  Confusion matrix:"
        f"\n  {confusion_path}"
    )

    print(
        f"\n  JSON results:"
        f"\n  {results_path}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()