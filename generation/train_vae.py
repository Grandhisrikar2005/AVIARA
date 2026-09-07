from pathlib import Path
import sys
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.append(
    str(PROJECT_ROOT)
)

from training.dataset import BirdsongDataset

from generation.vae import (
    BirdsongConditionalVAE,
    vae_loss
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LATENT_DIM = 64

SPECIES_EMBEDDING_DIM = 16

BETA_START = 0.00005

BETA_END = 0.0005

PATIENCE = 10

NUM_WORKERS = 0


# ============================================================
# PATHS
# ============================================================

MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "dataset_manifest.csv"
)

SPECTROGRAM_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "vae_spectrograms"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "vae_checkpoints"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "vae_normalized_best_model.pth"
)

HISTORY_FILE = (
    CHECKPOINT_DIR
    / "vae_normalized_training_history.json"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_TO_IDX = {

    "Eudynamys scolopaceus": 0,

    "Milvus migrans": 1,

    "Acridotherus tristis": 2,

    "Orthotomus sutorius": 3,

    "Centropus sinensis": 4,

    "Corvus splendens": 5,

    "Pavo cristatus": 6,

    "Pycnonotus cafer": 7,

    "Psittacula krameri": 8,

    "Halcyon smyrnensis": 9
}


# ============================================================
# TEMP MANIFEST
# ============================================================

def create_split_manifest(split):

    full_df = pd.read_csv(
        MANIFEST
    )

    split_df = full_df[
        full_df["split"] == split
    ].copy()

    split_df = split_df.reset_index(
        drop=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = (
        CHECKPOINT_DIR
        / f"_normalized_vae_{split}_manifest.csv"
    )

    split_df.to_csv(
        temp_file,
        index=False
    )

    return temp_file


# ============================================================
# DATASET
# ============================================================

def create_dataset(split):

    manifest_path = (
        create_split_manifest(
            split
        )
    )

    return BirdsongDataset(
        manifest_path,
        SPECTROGRAM_DIR,
        LABEL_TO_IDX
    )


# ============================================================
# DATALOADER
# ============================================================

def create_loader(
    dataset,
    shuffle
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )


# ============================================================
# BETA
# ============================================================

def get_beta(epoch):

    if EPOCHS <= 1:

        return BETA_END

    progress = (
        epoch - 1
    ) / (
        EPOCHS - 1
    )

    return (
        BETA_START
        +
        progress
        * (
            BETA_END
            - BETA_START
        )
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    beta
):

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    num_batches = 0

    for spectrograms, labels in loader:

        spectrograms = (
            spectrograms
            .float()
            .to(DEVICE)
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        reconstruction, mu, logvar, z = (
            model(
                spectrograms,
                labels
            )
        )

        loss, recon, kl = vae_loss(
            reconstruction,
            spectrograms,
            mu,
            logvar,
            beta=beta
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )

        optimizer.step()

        total_loss += loss.item()

        total_recon += recon.item()

        total_kl += kl.item()

        num_batches += 1

    return (
        total_loss / num_batches,
        total_recon / num_batches,
        total_kl / num_batches
    )


# ============================================================
# VALIDATE
# ============================================================

def validate(
    model,
    loader,
    beta
):

    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    num_batches = 0

    with torch.no_grad():

        for spectrograms, labels in loader:

            spectrograms = (
                spectrograms
                .float()
                .to(DEVICE)
            )

            labels = labels.to(
                DEVICE
            )

            reconstruction, mu, logvar, z = (
                model(
                    spectrograms,
                    labels
                )
            )

            loss, recon, kl = vae_loss(
                reconstruction,
                spectrograms,
                mu,
                logvar,
                beta=beta
            )

            total_loss += loss.item()

            total_recon += recon.item()

            total_kl += kl.item()

            num_batches += 1

    return (
        total_loss / num_batches,
        total_recon / num_batches,
        total_kl / num_batches
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    history
):

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_loss":
                val_loss,

            "latent_dim":
                LATENT_DIM,

            "species_embedding_dim":
                SPECIES_EMBEDDING_DIM,

            "label_to_idx":
                LABEL_TO_IDX,

            "spectrogram_normalization":
                {
                    "min_db": -80.0,
                    "max_db": 0.0
                },

            "history":
                history
        },
        BEST_CHECKPOINT
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NORMALIZED SPECIES-CONDITIONED BIRDSONG VAE"
    )
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Latent dimension: {LATENT_DIM}"
    )

    print(
        "\nSpectrogram representation:"
    )

    print(
        "Normalized 0...1"
    )

    print(
        "0 = -80 dB"
    )

    print(
        "1 = 0 dB"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "\nLoading training dataset..."
    )

    train_dataset = create_dataset(
        "train"
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        "\nLoading validation dataset..."
    )

    val_dataset = create_dataset(
        "val"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    train_loader = create_loader(
        train_dataset,
        shuffle=True
    )

    val_loader = create_loader(
        val_dataset,
        shuffle=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nCreating VAE..."
    )

    model = BirdsongConditionalVAE(
        num_classes=10,
        latent_dim=LATENT_DIM,
        species_embedding_dim=SPECIES_EMBEDDING_DIM
    )

    model.to(
        DEVICE
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    print(
        "\nStarting training..."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        beta = get_beta(
            epoch
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        train_loss, train_recon, train_kl = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                beta
            )
        )

        val_loss, val_recon, val_kl = (
            validate(
                model,
                val_loader,
                beta
            )
        )

        scheduler.step(
            val_loss
        )

        record = {

            "epoch": epoch,

            "learning_rate":
                current_lr,

            "beta":
                beta,

            "train_loss":
                train_loss,

            "train_reconstruction_loss":
                train_recon,

            "train_kl":
                train_kl,

            "val_loss":
                val_loss,

            "val_reconstruction_loss":
                val_recon,

            "val_kl":
                val_kl
        }

        history.append(
            record
        )

        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"LR: {current_lr:.7f}"
        )

        print(
            f"Beta: {beta:.6f}"
        )

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Train Recon: {train_recon:.6f}"
        )

        print(
            f"Train KL: {train_kl:.6f}"
        )

        print(
            f"Val Loss: {val_loss:.6f}"
        )

        print(
            f"Val Recon: {val_recon:.6f}"
        )

        print(
            f"Val KL: {val_kl:.6f}"
        )

        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                history
            )

            print(
                "✓ New best normalized VAE saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # CLEAN TEMP MANIFESTS
    # --------------------------------------------------------

    for split in [
        "train",
        "val"
    ]:

        temp_file = (
            CHECKPOINT_DIR
            / f"_normalized_vae_{split}_manifest.csv"
        )

        if temp_file.exists():

            temp_file.unlink()

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "NORMALIZED VAE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        "\nHistory:"
    )

    print(
        HISTORY_FILE
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()