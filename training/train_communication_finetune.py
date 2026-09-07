from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from communication_dataset import CommunicationDataset

from communication_model import (
    CommunicationModel,
    COMMUNICATION_TO_IDX,
    IDX_TO_COMMUNICATION,
    load_pretrained_v2,
)


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

CHECKPOINT_DIR = (
    BASE_DIR
    / "models"
    / "communication_checkpoints"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "finetuned_best_communication_model.pth"
)

HISTORY_FILE = (
    CHECKPOINT_DIR
    / "finetuned_communication_history.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "communication"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "finetuned_classification_report.txt"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR
    / "finetuned_confusion_matrix.csv"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "finetuned_test_results.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 16

NUM_EPOCHS = 30

BACKBONE_LR = 1e-5

HEAD_LR = 2e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

NUM_WORKERS = 0

MAX_GRAD_NORM = 2.0


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
# CONFIGURE FINE-TUNING
# ============================================================

def configure_finetuning(model):
    """
    Fine-tune the later part of the pretrained V2 acoustic
    representation while preserving the early CNN features.

    Frozen:
        CNN Blocks 1-3

    Trainable:
        CNN Block 4
        Feature projection
        Transformer
        Transformer normalization
        Attention pooling
        Embedding layer
        Communication head
    """

    # --------------------------------------------------------
    # Freeze entire pretrained backbone
    # --------------------------------------------------------

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Actual V2 CNN Block 4
    #
    # From training/model.py:
    #
    # Block 1: indices 0-7
    # Block 2: indices 8-15
    # Block 3: indices 16-23
    # Block 4: indices 24-30
    # --------------------------------------------------------

    cnn_layers = model.backbone.cnn.cnn

    for index in range(24, len(cnn_layers)):

        for parameter in (
            cnn_layers[index].parameters()
        ):
            parameter.requires_grad = True

    # --------------------------------------------------------
    # Feature projection
    # --------------------------------------------------------

    for parameter in (
        model.backbone.feature_projection.parameters()
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    for parameter in (
        model.backbone.transformer.parameters()
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Transformer normalization
    # --------------------------------------------------------

    for parameter in (
        model.backbone.transformer_norm.parameters()
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Attention pooling
    # --------------------------------------------------------

    for parameter in (
        model.backbone.attention_pooling.parameters()
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Embedding layer
    # --------------------------------------------------------

    for parameter in (
        model.backbone.embedding_layer.parameters()
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Communication head
    # --------------------------------------------------------

    for parameter in (
        model.communication_head.parameters()
    ):
        parameter.requires_grad = True


# ============================================================
# SET TRAINING MODES
# ============================================================

def set_finetuning_modes(model):
    """
    Keep frozen CNN blocks in eval mode and train the
    communication-specific portion of the network.
    """

    model.train()

    cnn_layers = model.backbone.cnn.cnn

    # --------------------------------------------------------
    # Frozen CNN Blocks 1-3
    # --------------------------------------------------------

    for index in range(0, 24):

        cnn_layers[index].eval()

    # --------------------------------------------------------
    # Trainable CNN Block 4
    # --------------------------------------------------------

    for index in range(24, len(cnn_layers)):

        cnn_layers[index].train()


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    labels = np.array(
        [
            sample["label"]
            for sample in dataset.samples
        ],
        dtype=np.int64,
    )

    all_classes = np.array(
        sorted(
            IDX_TO_COMMUNICATION.keys()
        )
    )

    present_classes = np.unique(labels)

    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=labels,
    )

    weights = np.ones(
        len(all_classes),
        dtype=np.float32,
    )

    for class_id, weight in zip(
        present_classes,
        present_weights,
    ):

        weights[class_id] = weight

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

def evaluate_recordings(
    model,
    dataset,
    device,
    batch_size=BATCH_SIZE,
):
    """
    Predict every segment belonging to a recording,
    average the segment probabilities, and make one
    recording-level prediction.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model.eval()

    all_probabilities = []

    with torch.inference_mode():

        for batch_x, _ in loader:

            batch_x = batch_x.to(
                device
            )

            logits = model(
                batch_x
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            all_probabilities.append(
                probabilities.cpu()
            )

    if not all_probabilities:

        raise RuntimeError(
            "No predictions were produced."
        )

    all_probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )

    # --------------------------------------------------------
    # Group segment predictions by recording
    # --------------------------------------------------------

    recording_groups = {}

    for index, sample in enumerate(
        dataset.samples
    ):

        recording_id = str(
            sample["recording_id"]
        )

        label = int(
            sample["label"]
        )

        if recording_id not in recording_groups:

            recording_groups[
                recording_id
            ] = {
                "indices": [],
                "label": label,
            }

        recording_groups[
            recording_id
        ]["indices"].append(index)

    recording_ids = []

    y_true = []

    y_pred = []

    recording_probabilities = []

    for recording_id, info in (
        recording_groups.items()
    ):

        indices = info["indices"]

        segment_probabilities = (
            all_probabilities[
                indices
            ]
        )

        # Mean probability over all segments.
        mean_probability = (
            segment_probabilities
            .mean(dim=0)
            .numpy()
        )

        prediction = int(
            mean_probability.argmax()
        )

        recording_ids.append(
            recording_id
        )

        y_true.append(
            info["label"]
        )

        y_pred.append(
            prediction
        )

        recording_probabilities.append(
            mean_probability
        )

    return {
        "recording_ids": recording_ids,
        "y_true": np.array(
            y_true,
            dtype=np.int64,
        ),
        "y_pred": np.array(
            y_pred,
            dtype=np.int64,
        ),
        "probabilities": np.array(
            recording_probabilities
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate evaluation metrics at recording level.
    """

    all_labels = list(
        sorted(
            IDX_TO_COMMUNICATION.keys()
        )
    )

    present_labels = sorted(
        set(
            y_true.tolist()
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=all_labels,
            average=None,
            zero_division=0,
        )
    )

    p_present, r_present, f_present, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=present_labels,
            average=None,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    set_finetuning_modes(
        model
    )

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(
            device
        )

        batch_y = batch_y.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            batch_x
        )

        loss = criterion(
            logits,
            batch_y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
        )

        optimizer.step()

        batch_size = (
            batch_y.size(0)
        )

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
            predictions == batch_y
        ).sum().item()

        total_samples += (
            batch_size
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V2-FINETUNED COMMUNICATION CLASSIFIER")
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
        f"Backbone LR: {BACKBONE_LR}"
    )

    print(
        f"Communication head LR: {HEAD_LR}"
    )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATASETS
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING COMMUNICATION DATA")
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
        f"Train segments      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation segments : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test segments       : "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # DATA LOADER
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING PRETRAINED V2 MODEL")
    print("=" * 70)

    model = CommunicationModel(
        num_classes=4
    )

    load_pretrained_v2(
        model,
        device=device,
    )

    # --------------------------------------------------------
    # Configure partial fine-tuning
    # --------------------------------------------------------

    configure_finetuning(
        model
    )

    model = model.to(
        device
    )

    # ========================================================
    # PARAMETER REPORT
    # ========================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print()
    print(
        f"Total parameters    : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters   : "
        f"{frozen_parameters:,}"
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    train_counts = (
        pd.Series(
            [
                sample["label"]
                for sample in train_dataset.samples
            ]
        )
        .value_counts()
        .sort_index()
    )

    for class_id in sorted(
        IDX_TO_COMMUNICATION.keys()
    ):

        count = int(
            train_counts.get(
                class_id,
                0
            )
        )

        print(
            f"{class_id}: "
            f"{IDX_TO_COMMUNICATION[class_id]:20s} "
            f"{count}"
        )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_dataset
        ).to(device)
    )

    print()
    print("=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)

    for class_id, weight in enumerate(
        class_weights.cpu().numpy()
    ):

        print(
            f"{class_id}: "
            f"{IDX_TO_COMMUNICATION[class_id]:20s} "
            f"{weight:.4f}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # DIFFERENTIAL LEARNING RATES
    # ========================================================

    backbone_parameters = []

    head_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if name.startswith(
            "communication_head."
        ):

            head_parameters.append(
                parameter
            )

        else:

            backbone_parameters.append(
                parameter
            )

    print()
    print(
        f"Trainable backbone tensors: "
        f"{len(backbone_parameters)}"
    )

    print(
        f"Trainable head tensors: "
        f"{len(head_parameters)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR,
            },
            {
                "params": head_parameters,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING V2 COMMUNICATION FINE-TUNING")
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

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        # ----------------------------------------------------
        # Recording-level validation
        # ----------------------------------------------------

        val_result = (
            evaluate_recordings(
                model,
                val_dataset,
                device,
            )
        )

        val_metrics = calculate_metrics(
            val_result["y_true"],
            val_result["y_pred"],
        )

        current_backbone_lr = (
            optimizer.param_groups[0]["lr"]
        )

        current_head_lr = (
            optimizer.param_groups[1]["lr"]
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val Balanced Acc "
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"Val Macro-F1 "
            f"{val_metrics['macro_f1']:.4f} | "
            f"Backbone LR {current_backbone_lr:.1e} | "
            f"Head LR {current_head_lr:.1e} | "
            f"{elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "train_accuracy": float(
                    train_accuracy
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
                "backbone_lr": float(
                    current_backbone_lr
                ),
                "head_lr": float(
                    current_head_lr
                ),
                "epoch_time_seconds": float(
                    elapsed
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
        # Save best checkpoint
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
                    "epoch": epoch,
                    "best_val_macro_f1": (
                        best_val_f1
                    ),
                    "communication_to_idx": (
                        COMMUNICATION_TO_IDX
                    ),
                    "backbone_learning_rate": (
                        BACKBONE_LR
                    ),
                    "head_learning_rate": (
                        HEAD_LR
                    ),
                    "fine_tuning": True,
                    "trainable_parameters": (
                        trainable_parameters
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
                "fine_tuning": True,
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
    print("LOADING BEST FINE-TUNED MODEL")
    print("=" * 70)

    if not BEST_CHECKPOINT.exists():

        raise FileNotFoundError(
            "No best fine-tuned checkpoint was created."
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
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

    test_result = evaluate_recordings(
        model,
        test_dataset,
        device,
    )

    test_metrics = calculate_metrics(
        test_result["y_true"],
        test_result["y_pred"],
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
    # CLASSIFICATION REPORT
    # ========================================================

    labels = list(
        sorted(
            IDX_TO_COMMUNICATION.keys()
        )
    )

    target_names = [
        IDX_TO_COMMUNICATION[
            class_id
        ]
        for class_id in labels
    ]

    report = classification_report(
        test_result["y_true"],
        test_result["y_pred"],
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        report
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        test_result["y_true"],
        test_result["y_pred"],
        labels=labels,
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
            "V2-FINETUNED RECORDING-LEVEL "
            "COMMUNICATION CLASSIFICATION\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Architecture:\n"
        )

        file.write(
            "Pretrained V2 CNN + Transformer with "
            "partial fine-tuning\n"
        )

        file.write(
            "Evaluation: recording-level probability "
            "aggregation\n\n"
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

        file.write(
            report
        )

        file.write(
            "\n\n"
        )

        file.write(
            "DATASET LIMITATIONS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            "The begging_distress category has no independent "
            "test recording in the current dataset.\n"
        )

        file.write(
            "The alarm_warning category has only one "
            "independent test recording.\n"
        )

        file.write(
            "Consequently, minority-class test metrics "
            "should be interpreted cautiously.\n"
        )

    # ========================================================
    # SAVE JSON RESULTS
    # ========================================================

    per_class = {}

    for class_id, class_name in (
        IDX_TO_COMMUNICATION.items()
    ):

        per_class[
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
        "fine_tuned": True,
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
        "trainable_parameters": int(
            trainable_parameters
        ),
        "recording_counts": {
            "train": 121,
            "validation": 26,
            "test": 27,
        },
        "per_class": per_class,
    }

    with open(
        RESULTS_FILE,
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
    print("FINE-TUNED COMMUNICATION TRAINING COMPLETE")
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
    print(RESULTS_FILE)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()