from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from generation.vae import BirdsongConditionalVAE


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "vae_checkpoints"
    / "vae_best_model.pth"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "test_manifest.csv"
)

SPECTROGRAM_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spectrograms"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vae_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VAE CONFIGURATION
# EXACT VALUES USED DURING TRAINING
# ============================================================

NUM_CLASSES = 10
LATENT_DIM = 64
SPECIES_EMBEDDING_DIM = 16


# ============================================================
# EXACT V2 SPECIES MAPPING
# ============================================================

LABEL_TO_IDX = {
    "Eudynamys scolopaceus": 0,      # Asian Koel
    "Milvus migrans": 1,             # Black Kite
    "Acridotheres tristis": 2,       # Common Myna
    "Orthotomus sutorius": 3,        # Common Tailorbird
    "Centropus sinensis": 4,         # Greater Coucal
    "Corvus splendens": 5,           # House Crow
    "Pavo cristatus": 6,             # Indian Peafowl
    "Pycnonotus cafer": 7,           # Red-vented Bulbul
    "Psittacula krameri": 8,         # Rose-ringed Parakeet
    "Halcyon smyrnensis": 9,         # White-throated Kingfisher
}


# ============================================================
# FIND SPECTROGRAM
# ============================================================

def find_spectrogram(row):
    """
    Locate the .npy spectrogram belonging to a test segment.
    """

    segment_id = str(row["segment_id"])

    # Example:
    # segment_0000.wav -> segment_0000.npy
    filename = Path(segment_id).stem + ".npy"

    # First try direct location
    direct_path = SPECTROGRAM_DIR / filename

    if direct_path.exists():
        return direct_path

    # Then search recursively
    matches = list(
        SPECTROGRAM_DIR.rglob(filename)
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# LOAD VAE
# ============================================================

def load_model():

    print("=" * 70)
    print("VAE RECONSTRUCTION TEST")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "\nVAE checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Create EXACT trained architecture
    # --------------------------------------------------------

    model = BirdsongConditionalVAE(
        num_classes=NUM_CLASSES,
        latent_dim=LATENT_DIM,
        species_embedding_dim=SPECIES_EMBEDDING_DIM,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    print(
        "✓ VAE checkpoint loaded successfully"
    )

    return model


# ============================================================
# SAVE RECONSTRUCTION PLOT
# ============================================================

def save_reconstruction_plot(
    original,
    reconstructed,
    species,
    segment_id,
    output_path,
):

    difference = np.abs(
        original - reconstructed
    )

    fig = plt.figure(
        figsize=(15, 10)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        3,
        1,
        1
    )

    ax1.imshow(
        original,
        aspect="auto",
        origin="lower",
    )

    ax1.set_title(
        f"Original Mel Spectrogram — {species}"
    )

    ax1.set_ylabel(
        "Mel Frequency"
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        3,
        1,
        2
    )

    ax2.imshow(
        reconstructed,
        aspect="auto",
        origin="lower",
    )

    ax2.set_title(
        "VAE Reconstruction"
    )

    ax2.set_ylabel(
        "Mel Frequency"
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    ax3 = fig.add_subplot(
        3,
        1,
        3
    )

    ax3.imshow(
        difference,
        aspect="auto",
        origin="lower",
    )

    ax3.set_title(
        "Absolute Reconstruction Difference"
    )

    ax3.set_xlabel(
        "Time"
    )

    ax3.set_ylabel(
        "Mel Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            "\nTest manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"\nTest samples available: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # Select one test sample from every species
    # --------------------------------------------------------

    selected_rows = []

    for species in LABEL_TO_IDX.keys():

        species_rows = manifest[
            manifest["scientific_name"]
            == species
        ]

        if len(species_rows) > 0:

            selected_rows.append(
                species_rows.iloc[0]
            )

    print(
        f"Testing "
        f"{len(selected_rows)} species examples..."
    )

    # --------------------------------------------------------
    # Reconstruction results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    with torch.no_grad():

        for index, row in enumerate(
            selected_rows
        ):

            species = row[
                "scientific_name"
            ]

            segment_id = row[
                "segment_id"
            ]

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"Sample "
                f"{index + 1}/"
                f"{len(selected_rows)}"
            )

            print(
                f"Species: {species}"
            )

            print(
                f"Segment: {segment_id}"
            )

            # ------------------------------------------------
            # Find spectrogram
            # ------------------------------------------------

            spectrogram_path = (
                find_spectrogram(row)
            )

            if spectrogram_path is None:

                print(
                    "⚠ Spectrogram not found"
                )

                continue

            print(
                f"Spectrogram: "
                f"{spectrogram_path}"
            )

            # ------------------------------------------------
            # Load spectrogram
            # ------------------------------------------------

            spectrogram = np.load(
                spectrogram_path
            ).astype(
                np.float32
            )

            # Remove unnecessary dimensions
            if spectrogram.ndim == 3:

                spectrogram = (
                    spectrogram.squeeze()
                )

            print(
                f"Input shape: "
                f"{spectrogram.shape}"
            )

            # ------------------------------------------------
            # Validate shape
            # ------------------------------------------------

            if spectrogram.ndim != 2:

                print(
                    "⚠ Invalid spectrogram "
                    f"dimensions: "
                    f"{spectrogram.shape}"
                )

                continue

            # ------------------------------------------------
            # Convert to tensor
            # ------------------------------------------------

            input_tensor = (
                torch.from_numpy(
                    spectrogram
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Species index
            # ------------------------------------------------

            species_index = LABEL_TO_IDX[
                species
            ]

            species_tensor = torch.tensor(
                [species_index],
                dtype=torch.long,
                device=DEVICE,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                input_tensor,
                species_tensor,
            )

            # ------------------------------------------------
            # Extract reconstruction
            # ------------------------------------------------

            if isinstance(
                output,
                tuple
            ):

                reconstructed = output[0]

            else:

                reconstructed = output

            reconstructed = (
                reconstructed
                .squeeze()
                .cpu()
                .numpy()
            )

            print(
                f"Output shape: "
                f"{reconstructed.shape}"
            )

            # ------------------------------------------------
            # Validate reconstruction
            # ------------------------------------------------

            if reconstructed.shape != (
                128,
                501,
            ):

                print(
                    "⚠ Unexpected reconstruction "
                    f"shape: "
                    f"{reconstructed.shape}"
                )

                continue

            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            mse = np.mean(
                (
                    spectrogram
                    - reconstructed
                ) ** 2
            )

            mae = np.mean(
                np.abs(
                    spectrogram
                    - reconstructed
                )
            )

            # ------------------------------------------------
            # Safe species filename
            # ------------------------------------------------

            safe_species = (
                species
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "/",
                    "_"
                )
            )

            # ------------------------------------------------
            # Save reconstruction image
            # ------------------------------------------------

            plot_path = (
                OUTPUT_DIR
                / f"{safe_species}_reconstruction.png"
            )

            save_reconstruction_plot(
                original=spectrogram,
                reconstructed=reconstructed,
                species=species,
                segment_id=segment_id,
                output_path=plot_path,
            )

            # ------------------------------------------------
            # Save reconstructed spectrogram
            # ------------------------------------------------

            reconstructed_path = (
                OUTPUT_DIR
                / f"{safe_species}_reconstructed.npy"
            )

            np.save(
                reconstructed_path,
                reconstructed.astype(
                    np.float32
                ),
            )

            # ------------------------------------------------
            # Store metrics
            # ------------------------------------------------

            results.append(
                {
                    "species": species,
                    "segment_id": segment_id,
                    "mse": float(mse),
                    "mae": float(mae),
                    "original_shape": str(
                        spectrogram.shape
                    ),
                    "reconstructed_shape": str(
                        reconstructed.shape
                    ),
                }
            )

            print(
                f"MSE: {mse:.4f}"
            )

            print(
                f"MAE: {mae:.4f}"
            )

            print(
                "✓ Reconstruction saved"
            )

    # ========================================================
    # SAVE METRICS CSV
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        OUTPUT_DIR
        / "reconstruction_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VAE RECONSTRUCTION TEST COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessful reconstructions: "
        f"{len(results)}"
    )

    if len(results) > 0:

        print(
            f"Mean MSE: "
            f"{results_df['mse'].mean():.4f}"
        )

        print(
            f"Mean MAE: "
            f"{results_df['mae'].mean():.4f}"
        )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nMetrics:"
    )

    print(
        f"  {results_path}"
    )

    print(
        "\n✓ VAE reconstruction test "
        "completed successfully."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()