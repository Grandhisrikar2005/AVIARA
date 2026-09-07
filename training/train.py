from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score

from dataset import create_datasets
from model import BirdsongCNNTransformer


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 30

LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-6

WEIGHT_DECAY = 2e-4

LABEL_SMOOTHING = 0.08

PATIENCE = 7

NUM_WORKERS = 0

EMBEDDING_DIM = 128

# V2 checkpoint files
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR /
    "v2_best_model.pth"
)

HISTORY_PATH = (
    CHECKPOINT_DIR /
    "v2_training_history.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

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
# WEIGHTED SAMPLER
# ============================================================

def create_weighted_sampler(dataset):
    """
    Creates a class-balanced sampling strategy.

    Rare species receive higher sampling probability.
    """

    labels = dataset.df["species"].map(
        dataset.label_to_idx
    ).values

    class_counts = np.bincount(labels)

    # Inverse-frequency class weights
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for spectrograms, labels in dataloader:

        spectrograms = spectrograms.to(device)

        labels = labels.to(device)

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits, embeddings = model(
            spectrograms
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0
        )

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item() *
            spectrograms.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for spectrograms, labels in dataloader:

            spectrograms = spectrograms.to(
                device
            )

            labels = labels.to(
                device
            )

            logits, embeddings = model(
                spectrograms
            )

            loss = criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item() *
                spectrograms.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )

    validation_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    validation_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    validation_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        validation_loss,
        validation_accuracy,
        validation_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("=" * 70)
    print("AI BIO-ACOUSTIC BIRDSONG MODEL V2 TRAINING")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cpu":

        print(
            "Running on CPU."
        )

        print(
            "Training may take some time."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        label_to_idx
    ) = create_datasets()

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples  : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples : "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Verify augmentation
    # --------------------------------------------------------

    print(
        "\nTraining augmentation: ENABLED"
    )

    print(
        "Validation augmentation: DISABLED"
    )

    print(
        "Test augmentation: DISABLED"
    )

    # --------------------------------------------------------
    # Weighted sampler
    # --------------------------------------------------------

    print(
        "\nCreating class-balanced sampler..."
    )

    train_sampler = create_weighted_sampler(
        train_dataset
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Create V2 model
    # --------------------------------------------------------

    print(
        "\nCreating V2 CNN + Transformer model..."
    )

    model = BirdsongCNNTransformer(
        num_classes=len(label_to_idx),
        cnn_dim=256,
        transformer_dim=256,
        num_heads=4,
        num_layers=3,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.30
    )

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    print(
        f"\nLabel smoothing: "
        f"{LABEL_SMOOTHING}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Cosine learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rate": []
    }

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_val_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting V2 training..."
    )

    print("-" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        (
            train_loss,
            train_accuracy,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ====================================================
        # VALIDATE
        # ====================================================

        (
            val_loss,
            val_accuracy,
            val_f1
        ) = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # ====================================================
        # Learning rate
        # ====================================================

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ====================================================
        # Save history
        # ====================================================

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["train_macro_f1"].append(
            train_f1
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_macro_f1"].append(
            val_f1
        )

        history["learning_rate"].append(
            current_lr
        )

        # ====================================================
        # Print epoch
        # ====================================================

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ====================================================
        # Save best checkpoint
        # ====================================================

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
                "model_version": "V2",
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_f1":
                    val_f1,

                "val_accuracy":
                    val_accuracy,

                "label_to_idx":
                    label_to_idx,

                "config": {
                    "cnn_dim": 256,
                    "transformer_dim": 256,
                    "num_heads": 4,
                    "num_layers": 3,
                    "embedding_dim": 128,
                    "dropout": 0.30,
                    "label_smoothing":
                        LABEL_SMOOTHING
                }
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print(
                f"  ✓ V2 best model saved "
                f"(Val Macro-F1: {val_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ====================================================
        # Update learning rate
        # ====================================================

        scheduler.step()

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered "
                f"after {PATIENCE} epochs "
                "without validation improvement."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "V2 TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        "\nV2 best model:"
    )

    print(
        f"  {BEST_MODEL_PATH}"
    )

    print(
        "\nV2 training history:"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print(
        "\nSpecies mapping:"
    )

    for species, index in sorted(
        label_to_idx.items(),
        key=lambda x: x[1]
    ):

        print(
            f"  {index}: {species}"
        )

    print(
        "\nV1 baseline:"
    )

    print(
        "  Test Accuracy : 67.05%"
    )

    print(
        "  Test Macro F1  : 56.84%"
    )

    print(
        "\nNext step:"
    )

    print(
        "Evaluate V2 on the held-out test set."
    )


if __name__ == "__main__":

    main()