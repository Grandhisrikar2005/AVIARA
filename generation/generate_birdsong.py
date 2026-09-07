from pathlib import Path
import sys

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from generation.vae import BirdsongConditionalVAE


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SPECIES = "Eudynamys scolopaceus"   # Asian Koel

LATENT_DIM = 64
SPECIES_EMBEDDING_DIM = 16

SAMPLE_RATE = 32000
DURATION = 5

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320
FMIN = 50
FMAX = 14000

MIN_DB = -80.0
MAX_DB = 0.0

NUM_ITERATIONS = 256

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "vae_checkpoints"
    / "vae_normalized_best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "normalized_generation"
)


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


IDX_TO_SPECIES = {
    value: key
    for key, value in LABEL_TO_IDX.items()
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading normalized VAE...")

    model = BirdsongConditionalVAE(
        num_classes=10,
        latent_dim=LATENT_DIM,
        species_embedding_dim=SPECIES_EMBEDDING_DIM
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("VAE loaded successfully.")

    return model


# ============================================================
# GENERATE MEL SPECTROGRAM
# ============================================================

def generate_spectrogram(model, species_index):

    print("\nGenerating latent representation...")

    with torch.no_grad():

        z = torch.randn(
            1,
            LATENT_DIM,
            device=DEVICE
        )

        labels = torch.tensor(
            [species_index],
            dtype=torch.long,
            device=DEVICE
        )

        generated = model.decode(
            z,
            labels
        )

    generated = (
        generated
        .squeeze()
        .cpu()
        .numpy()
    )

    # Decoder output is normalized 0...1
    generated = np.clip(
        generated,
        0.0,
        1.0
    )

    # Convert normalized representation back to dB
    mel_db = (
        generated * (MAX_DB - MIN_DB)
        + MIN_DB
    )

    return mel_db


# ============================================================
# MEL → AUDIO
# ============================================================

def mel_to_audio(mel_db):

    print("Converting spectrogram to audio...")

    # dB → power
    mel_power = librosa.db_to_power(
        mel_db,
        ref=1.0
    )

    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        fmin=FMIN,
        fmax=FMAX,
        n_iter=NUM_ITERATIONS
    )

    # Normalize safely
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = audio / peak * 0.90

    return audio.astype(np.float32)


# ============================================================
# SAVE SPECTROGRAM
# ============================================================

def save_spectrogram(mel_db, path):

    plt.figure(
        figsize=(12, 5)
    )

    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        f"Generated Asian Koel Spectrogram"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NORMALIZED VAE BIRDSONG GENERATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Species: {SPECIES}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not CHECKPOINT.exists():

        print(
            "\nERROR: Normalized VAE checkpoint not found."
        )

        print(CHECKPOINT)

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()

    species_index = LABEL_TO_IDX[
        SPECIES
    ]

    mel_db = generate_spectrogram(
        model,
        species_index
    )

    print(
        f"Generated Mel shape: {mel_db.shape}"
    )

    print(
        f"Mel dB range: "
        f"{mel_db.min():.2f} "
        f"to "
        f"{mel_db.max():.2f}"
    )

    # Save spectrogram
    spectrogram_path = (
        OUTPUT_DIR
        / "normalized_vae_asian_koel.png"
    )

    save_spectrogram(
        mel_db,
        spectrogram_path
    )

    # Generate audio
    audio = mel_to_audio(
        mel_db
    )

    audio_path = (
        OUTPUT_DIR
        / "normalized_vae_asian_koel.wav"
    )

    sf.write(
        audio_path,
        audio,
        SAMPLE_RATE
    )

    duration = len(audio) / SAMPLE_RATE
    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    print("\n" + "=" * 70)
    print("GENERATION SUCCESSFUL")
    print("=" * 70)

    print(
        f"\nSpecies : Asian Koel"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"RMS     : {rms:.4f}"
    )

    print(
        f"\nAudio:"
    )

    print(audio_path)

    print(
        f"\nSpectrogram:"
    )

    print(spectrogram_path)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()