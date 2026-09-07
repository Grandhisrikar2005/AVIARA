from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from communication_model import (
    CommunicationHead,
    COMMUNICATION_TO_IDX,
    IDX_TO_COMMUNICATION,
)
from communication_dataset import CommunicationDataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_MANIFEST = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_train_manifest.csv"
)

VAL_MANIFEST = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_val_manifest.csv"
)

TEST_MANIFEST = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_test_manifest.csv"
)

CACHE_DIR = (
    BASE_DIR
    / "models"
    / "communication_checkpoints"
    / "embedding_cache"
)

BEST_CHECKPOINT = (
    BASE_DIR
    / "models"
    / "communication_checkpoints"
    / "recording_level_best_model.pth"
)

HISTORY_FILE = (
    BASE_DIR
    / "models"
    / "communication_checkpoints"
    / "recording_level_training_history.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "communication"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "recording_level_classification_report.txt"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR
    / "recording_level_confusion_matrix.csv"
)

TEST_RESULTS_FILE = (
    OUTPUT_DIR
    / "recording_level_test_results.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 16

NUM_EPOCHS = 50

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 10

NUM_WORKERS = 0

CACHE_VERSION = "v1"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CACHE LOADING
# ============================================================

def load_cached_embeddings(split_name):

    cache_path = (
        CACHE_DIR
        / f"{split_name}_embeddings_{CACHE_VERSION}.pt"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached embeddings not found:\n"
            f"{cache_path}\n\n"
            "Run the previous embedding extraction once "
            "before using this recording-level trainer."
        )

    cached = torch.load(
        cache_path,
        map_location="cpu",
    )

    embeddings = cached["embeddings"].float()
    labels = cached["labels"].long()

    return embeddings, labels, cache_path


# ============================================================
# RECORDING-LEVEL AGGREGATION
# ============================================================

def build_recording_level_dataset(
    dataset,
    embeddings,
    cached_labels,
    split_name,
):
    """
    Convert segment-level 128-D embeddings into one
    recording-level representation per original recording.

    For each recording:

        mean embedding  -> 128-D
        max embedding   -> 128-D
        concatenate      -> 256-D
    """

    print()
    print(
        f"[{split_name.upper()}] "
        "Building recording-level representations..."
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(dataset) != len(embeddings):
        raise ValueError(
            f"{split_name}: dataset/cache length mismatch. "
            f"Dataset={len(dataset)}, "
            f"Cache={len(embeddings)}"
        )

    if len(dataset) != len(cached_labels):
        raise ValueError(
            f"{split_name}: label/cache length mismatch. "
            f"Dataset={len(dataset)}, "
            f"Cache={len(cached_labels)}"
        )

    # --------------------------------------------------------
    # Get recording IDs in exactly the same order used
    # when the embeddings were cached.
    # --------------------------------------------------------

    recording_ids = [
        sample["recording_id"]
        for sample in dataset.samples
    ]

    dataset_labels = torch.tensor(
        [
            sample["label"]
            for sample in dataset.samples
        ],
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Verify cached labels still match dataset labels.
    # --------------------------------------------------------

    if not torch.equal(
        dataset_labels,
        cached_labels,
    ):
        raise ValueError(
            f"{split_name}: cached labels do not match "
            "the communication dataset order. "
            "Delete the cache and regenerate it."
        )

    # --------------------------------------------------------
    # Group segment embeddings by recording.
    # --------------------------------------------------------

    groups = {}

    for index, recording_id in enumerate(
        recording_ids
    ):

        groups.setdefault(
            recording_id,
            []
        ).append(index)

    recording_embeddings = []
    recording_labels = []
    recording_ids_final = []
    segment_counts = []

    for recording_id, indices in groups.items():

        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
        )

        segment_embeddings = embeddings[
            index_tensor
        ]

        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

        mean_embedding = (
            segment_embeddings.mean(
                dim=0
            )
        )

        # ----------------------------------------------------
        # Max pooling
        # ----------------------------------------------------

        max_embedding = (
            segment_embeddings.max(
                dim=0
            ).values
        )

        # ----------------------------------------------------
        # Combined 256-D representation
        # ----------------------------------------------------

        combined_embedding = torch.cat(
            [
                mean_embedding,
                max_embedding,
            ],
            dim=0,
        )

        # ----------------------------------------------------
        # Recording label
        # ----------------------------------------------------

        recording_label = int(
            cached_labels[
                indices[0]
            ].item()
        )

        # Verify every segment from a recording has
        # the same communication label.
        recording_segment_labels = (
            cached_labels[
                index_tensor
            ]
        )

        if not torch.all(
            recording_segment_labels
            == recording_label
        ):
            raise ValueError(
                f"Recording {recording_id} contains "
                "multiple communication labels."
            )

        recording_embeddings.append(
            combined_embedding
        )

        recording_labels.append(
            recording_label
        )

        recording_ids_final.append(
            recording_id
        )

        segment_counts.append(
            len(indices)
        )

    recording_embeddings = torch.stack(
        recording_embeddings
    )

    recording_labels = torch.tensor(
        recording_labels,
        dtype=torch.long,
    )

    print(
        f"[{split_name.upper()}] "
        f"Segments: {len(embeddings)}"
    )

    print(
        f"[{split_name.upper()}] "
        f"Recordings: {len(recording_embeddings)}"
    )

    print(
        f"[{split_name.upper()}] "
        f"Representation shape: "
        f"{tuple(recording_embeddings.shape)}"
    )

    print(
        f"[{split_name.upper()}] "
        f"Average segments/recording: "
        f"{np.mean(segment_counts):.2f}"
    )

    return (
        recording_embeddings,
        recording_labels,
        recording_ids_final,
        segment_counts,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    labels,
):

    labels_np = labels.numpy()

    all_classes = np.array(
        sorted(
            IDX_TO_COMMUNICATION.keys()
        )
    )

    present_classes = np.unique(
        labels_np
    )

    weights_present = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=labels_np,
    )

    weights = np.ones(
        len(all_classes),
        dtype=np.float32,
    )

    for class_id, weight in zip(
        present_classes,
        weights_present,
    ):

        weights[class_id] = weight

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for embeddings, labels in loader:

        embeddings = embeddings.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            embeddings
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0,
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.inference_mode():

        for embeddings, labels in loader:

            embeddings = embeddings.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                embeddings
            )

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    labels_all = list(
        sorted(
            IDX_TO_COMMUNICATION.keys()
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_all,
            average=None,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Macro metrics only across classes actually present
    # in this split.
    # --------------------------------------------------------

    present_classes = sorted(
        set(
            y_true.tolist()
        )
    )

    p_present, r_present, f_present, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=present_classes,
            average=None,
            zero_division=0,
        )
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            p_present.mean()
        ),
        "macro_recall": float(
            r_present.mean()
        ),
        "macro_f1": float(
            f_present.mean()
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECORDING-LEVEL COMMUNICATION CLASSIFIER")
    print("=" * 70)

    set_seed()

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    BEST_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD COMMUNICATION DATASETS
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING COMMUNICATION DATASETS")
    print("=" * 70)

    train_dataset = CommunicationDataset(
        TRAIN_MANIFEST
    )

    val_dataset = CommunicationDataset(
        VAL_MANIFEST
    )

    test_dataset = CommunicationDataset(
        TEST_MANIFEST
    )

    print()
    print(
        f"Train segments: {len(train_dataset)}"
    )

    print(
        f"Validation segments: {len(val_dataset)}"
    )

    print(
        f"Test segments: {len(test_dataset)}"
    )

    # ========================================================
    # LOAD EXISTING CACHED V2 EMBEDDINGS
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING CACHED V2 EMBEDDINGS")
    print("=" * 70)

    start_time = time.time()

    train_embeddings, train_labels, train_cache = (
        load_cached_embeddings("train")
    )

    val_embeddings, val_labels, val_cache = (
        load_cached_embeddings("val")
    )

    test_embeddings, test_labels, test_cache = (
        load_cached_embeddings("test")
    )

    print(
        f"Train cache : {train_cache}"
    )

    print(
        f"Val cache   : {val_cache}"
    )

    print(
        f"Test cache  : {test_cache}"
    )

    print(
        f"Cache loading completed in "
        f"{time.time() - start_time:.2f}s"
    )

    # ========================================================
    # RECORDING-LEVEL AGGREGATION
    # ========================================================

    print()
    print("=" * 70)
    print("CONVERTING SEGMENT EMBEDDINGS TO RECORDING EMBEDDINGS")
    print("=" * 70)

    (
        train_recording_embeddings,
        train_recording_labels,
        train_recording_ids,
        train_segment_counts,
    ) = build_recording_level_dataset(
        train_dataset,
        train_embeddings,
        train_labels,
        "train",
    )

    (
        val_recording_embeddings,
        val_recording_labels,
        val_recording_ids,
        val_segment_counts,
    ) = build_recording_level_dataset(
        val_dataset,
        val_embeddings,
        val_labels,
        "val",
    )

    (
        test_recording_embeddings,
        test_recording_labels,
        test_recording_ids,
        test_segment_counts,
    ) = build_recording_level_dataset(
        test_dataset,
        test_embeddings,
        test_labels,
        "test",
    )

    # ========================================================
    # CHECK RECORDING COUNTS
    # ========================================================

    assert len(
        train_recording_embeddings
    ) == 121

    assert len(
        val_recording_embeddings
    ) == 26

    assert len(
        test_recording_embeddings
    ) == 27

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    train_ids = set(
        train_recording_ids
    )

    val_ids = set(
        val_recording_ids
    )

    test_ids = set(
        test_recording_ids
    )

    print()
    print("=" * 70)
    print("RECORDING LEAKAGE CHECK")
    print("=" * 70)

    print(
        f"Train ∩ Val  : "
        f"{len(train_ids & val_ids)}"
    )

    print(
        f"Train ∩ Test : "
        f"{len(train_ids & test_ids)}"
    )

    print(
        f"Val ∩ Test   : "
        f"{len(val_ids & test_ids)}"
    )

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise RuntimeError(
            "Recording leakage detected."
        )

    print(
        "No recording-level leakage detected."
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("RECORDING-LEVEL CLASS DISTRIBUTION")
    print("=" * 70)

    distribution = pd.DataFrame(
        {
            "train": pd.Series(
                train_recording_labels.numpy()
            ).value_counts(),

            "val": pd.Series(
                val_recording_labels.numpy()
            ).value_counts(),

            "test": pd.Series(
                test_recording_labels.numpy()
            ).value_counts(),
        }
    ).fillna(0).astype(int)

    distribution.index = [
        IDX_TO_COMMUNICATION[
            int(index)
        ]
        for index in distribution.index
    ]

    print(
        distribution.to_string()
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_recording_labels
        )
    )

    print()
    print("=" * 70)
    print("RECORDING-LEVEL CLASS WEIGHTS")
    print("=" * 70)

    for class_id, weight in enumerate(
        class_weights.numpy()
    ):

        print(
            f"{class_id}: "
            f"{IDX_TO_COMMUNICATION[class_id]:20s} "
            f"{weight:.4f}"
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_tensor_dataset = TensorDataset(
        train_recording_embeddings,
        train_recording_labels,
    )

    val_tensor_dataset = TensorDataset(
        val_recording_embeddings,
        val_recording_labels,
    )

    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ========================================================
    # COMMUNICATION HEAD
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING RECORDING-LEVEL COMMUNICATION HEAD")
    print("=" * 70)

    # 128 mean + 128 max = 256 dimensions.
    model = CommunicationHead(
        embedding_dim=256,
        num_classes=4,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    class_weights = class_weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING RECORDING-LEVEL TRAINING")
    print("=" * 70)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        elapsed = (
            time.time() - epoch_start
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val Balanced Acc {val_metrics['balanced_accuracy']:.4f} | "
            f"Val Macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"LR {current_lr:.6f} | "
            f"{elapsed:.2f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "train_accuracy": float(
                    train_accuracy
                ),
                "val_loss": float(
                    val_metrics["loss"]
                ),
                "val_accuracy": float(
                    val_metrics["accuracy"]
                ),
                "val_balanced_accuracy": float(
                    val_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "val_macro_precision": float(
                    val_metrics[
                        "macro_precision"
                    ]
                ),
                "val_macro_recall": float(
                    val_metrics[
                        "macro_recall"
                    ]
                ),
                "val_macro_f1": float(
                    val_metrics[
                        "macro_f1"
                    ]
                ),
                "learning_rate": float(
                    current_lr
                ),
            }
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["macro_f1"]
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "embedding_dim": 256,
                    "num_classes": 4,
                    "best_val_macro_f1": (
                        best_val_f1
                    ),
                    "epoch": epoch,
                    "communication_to_idx": (
                        COMMUNICATION_TO_IDX
                    ),
                    "aggregation": (
                        "mean_plus_max"
                    ),
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  ✓ Best model saved "
                f"(Val Macro-F1: "
                f"{best_val_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_macro_f1": float(
                    best_val_f1
                ),
                "aggregation": "mean_plus_max",
                "history": history,
            },
            file,
            indent=2,
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST RECORDING-LEVEL MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RECORDING-LEVEL TEST")
    print("=" * 70)

    test_tensor_dataset = TensorDataset(
        test_recording_embeddings,
        test_recording_labels,
    )

    test_loader = DataLoader(
        test_tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1 "
        f"(classes present): "
        f"{test_metrics['macro_f1']:.4f}"
    )

    # ========================================================
    # TEST CLASSIFICATION REPORT
    # ========================================================

    labels_all = list(
        sorted(
            IDX_TO_COMMUNICATION.keys()
        )
    )

    target_names = [
        IDX_TO_COMMUNICATION[
            class_id
        ]
        for class_id in labels_all
    ]

    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=labels_all,
        target_names=target_names,
        zero_division=0,
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=labels_all,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names,
    )

    confusion_df.to_csv(
        CONFUSION_MATRIX_FILE
    )

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "RECORDING-LEVEL COMMUNICATION "
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Aggregation: mean + max "
            "segment embeddings\n\n"
        )

        file.write(
            f"Best epoch: {best_epoch}\n"
        )

        file.write(
            f"Best validation Macro-F1: "
            f"{best_val_f1:.4f}\n\n"
        )

        file.write(
            f"Test Accuracy: "
            f"{test_metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Test Balanced Accuracy: "
            f"{test_metrics['balanced_accuracy']:.4f}\n"
        )

        file.write(
            f"Test Macro-F1 "
            f"(classes present): "
            f"{test_metrics['macro_f1']:.4f}\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(report)

        file.write(
            "\n\n"
        )

        file.write(
            "DATASET LIMITATION\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            "The begging_distress class has no independent "
            "test recordings in the current dataset. "
            "Therefore, its held-out test performance "
            "cannot be considered reliable.\n"
        )

        file.write(
            "Alarm_warning also has very few independent "
            "test recordings, so its test metrics should "
            "be interpreted cautiously.\n"
        )

    # ========================================================
    # SAVE JSON RESULTS
    # ========================================================

    per_class_results = {}

    for class_id, class_name in (
        IDX_TO_COMMUNICATION.items()
    ):

        per_class_results[
            class_name
        ] = {
            "precision": float(
                test_metrics[
                    "precision"
                ][class_id]
            ),
            "recall": float(
                test_metrics[
                    "recall"
                ][class_id]
            ),
            "f1": float(
                test_metrics[
                    "f1"
                ][class_id]
            ),
            "support": int(
                test_metrics[
                    "support"
                ][class_id]
            ),
        }

    results = {
        "aggregation": "mean_plus_max",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(
            best_val_f1
        ),
        "test_accuracy": float(
            test_metrics["accuracy"]
        ),
        "test_balanced_accuracy": float(
            test_metrics[
                "balanced_accuracy"
            ]
        ),
        "test_macro_f1_present": float(
            test_metrics[
                "macro_f1"
            ]
        ),
        "recording_counts": {
            "train": len(
                train_recording_embeddings
            ),
            "val": len(
                val_recording_embeddings
            ),
            "test": len(
                test_recording_embeddings
            ),
        },
        "per_class": per_class_results,
    }

    with open(
        TEST_RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RECORDING-LEVEL TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation Macro-F1 : "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Test Accuracy            : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy   : "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1            : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()
    print("FILES SAVED")
    print("-" * 70)

    print(BEST_CHECKPOINT)
    print(HISTORY_FILE)
    print(REPORT_FILE)
    print(CONFUSION_MATRIX_FILE)
    print(TEST_RESULTS_FILE)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()