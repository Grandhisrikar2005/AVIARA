from pathlib import Path
import sys

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from generation.vae import BirdsongConditionalVAE


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "vae_checkpoints"
    / "vae_normalized_best_model.pth"
)

SOURCE_AUDIO = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "Eudynamys_scolopaceus"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hybrid_generation"
)

SPECIES = "Eudynamys scolopaceus"

LATENT_DIM = 64
SPECIES_EMBEDDING_DIM = 16

SAMPLE_RATE = 32000

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320
FMIN = 50
FMAX = 14000

MIN_DB = -80.0
MAX_DB = 0.0


# ============================================================
# ARTIFACT CONTROL
# ============================================================

# Overall AI influence.
AI_STRENGTH = 0.07

# Small latent movement.
LATENT_NOISE = 0.07

# Only stronger vocal regions are modified.
VOCAL_THRESHOLD = 0.35

# Maximum allowed spectral modification.
# Prevents extreme AI-generated frequency spikes.
MAX_SPECTRAL_CHANGE = 0.12

# Smooth AI spectrum in:
# frequency direction and time direction.
FREQUENCY_SMOOTHING = 1.2
TIME_SMOOTHING = 1.5


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
# MODEL
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
# SOURCE
# ============================================================

def find_source_audio():

    files = list(
        SOURCE_AUDIO.rglob("*.wav")
    )

    if not files:
        raise FileNotFoundError(
            f"No WAV files found in {SOURCE_AUDIO}"
        )

    return files[0]


# ============================================================
# MEL
# ============================================================

def audio_to_mel(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    return librosa.power_to_db(
        mel,
        ref=1.0,
        top_db=80.0
    )


def fix_mel_length(mel):

    target_frames = 501

    if mel.shape[1] > target_frames:

        return mel[:, :target_frames]

    if mel.shape[1] < target_frames:

        padding = (
            target_frames - mel.shape[1]
        )

        return np.pad(
            mel,
            ((0, 0), (0, padding)),
            mode="constant",
            constant_values=MIN_DB
        )

    return mel


def normalize_mel(mel_db):

    normalized = (
        mel_db - MIN_DB
    ) / (
        MAX_DB - MIN_DB
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# AI MEL
# ============================================================

def generate_ai_mel(
    model,
    source_mel,
    species_index
):

    normalized = normalize_mel(
        source_mel
    )

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    labels = torch.tensor(
        [species_index],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        _, mu, _, _ = model(
            tensor,
            labels
        )

        z = (
            mu
            + LATENT_NOISE
            * torch.randn_like(mu)
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

    generated = np.clip(
        generated,
        0.0,
        1.0
    )

    return (
        generated
        * (MAX_DB - MIN_DB)
        + MIN_DB
    )


# ============================================================
# VOCAL MASK
# ============================================================

def create_vocal_mask(
    original_mel
):

    energy = np.mean(
        original_mel,
        axis=0
    )

    minimum = energy.min()
    maximum = energy.max()

    if maximum - minimum < 1e-8:

        return np.ones_like(
            energy
        )

    energy = (
        energy - minimum
    ) / (
        maximum - minimum
    )

    mask = (
        energy - VOCAL_THRESHOLD
    ) / (
        1.0 - VOCAL_THRESHOLD
    )

    mask = np.clip(
        mask,
        0.0,
        1.0
    )

    # Smooth transitions so AI does not
    # suddenly switch on/off.
    mask = np.convolve(
        mask,
        np.ones(15) / 15,
        mode="same"
    )

    return np.clip(
        mask,
        0.0,
        1.0
    )


# ============================================================
# CLEAN AI SPECTRUM
# ============================================================

def clean_ai_spectrum(
    original_mel,
    ai_mel
):

    frames = min(
        original_mel.shape[1],
        ai_mel.shape[1]
    )

    original_mel = (
        original_mel[:, :frames]
    )

    ai_mel = (
        ai_mel[:, :frames]
    )

    # Difference between AI and original.
    difference = (
        ai_mel - original_mel
    )

    # Remove isolated spectral spikes.
    difference = gaussian_filter(
        difference,
        sigma=(
            FREQUENCY_SMOOTHING,
            TIME_SMOOTHING
        )
    )

    # Hard-limit the maximum AI change.
    difference = np.clip(
        difference,
        -MAX_SPECTRAL_CHANGE * 80.0,
        MAX_SPECTRAL_CHANGE * 80.0
    )

    cleaned = (
        original_mel
        + difference
    )

    return cleaned


# ============================================================
# CREATE HYBRID MEL
# ============================================================

def create_hybrid_mel(
    original_mel,
    ai_mel
):

    frames = min(
        original_mel.shape[1],
        ai_mel.shape[1]
    )

    original_mel = (
        original_mel[:, :frames]
    )

    ai_mel = (
        ai_mel[:, :frames]
    )

    # Clean AI representation first.
    cleaned_ai = clean_ai_spectrum(
        original_mel,
        ai_mel
    )

    vocal_mask = create_vocal_mask(
        original_mel
    )

    vocal_mask = vocal_mask[
        np.newaxis,
        :
    ]

    # AI only operates on vocal regions.
    effective_strength = (
        AI_STRENGTH
        * vocal_mask
    )

    hybrid = (
        original_mel
        +
        effective_strength
        * (
            cleaned_ai
            - original_mel
        )
    )

    return hybrid


# ============================================================
# SPECTRAL RECONSTRUCTION
# ============================================================

def mel_to_stft_magnitude(
    mel_db
):

    mel_power = librosa.db_to_power(
        mel_db,
        ref=1.0
    )

    stft_power = (
        librosa.feature.inverse.mel_to_stft(
            mel_power,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            power=2.0,
            fmin=FMIN,
            fmax=FMAX
        )
    )

    return np.sqrt(
        np.maximum(
            stft_power,
            1e-10
        )
    )


def reconstruct_audio(
    original_audio,
    hybrid_mel
):

    original_stft = librosa.stft(
        original_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT
    )

    original_magnitude = np.abs(
        original_stft
    )

    original_phase = np.angle(
        original_stft
    )

    hybrid_magnitude = (
        mel_to_stft_magnitude(
            hybrid_mel
        )
    )

    frames = min(
        original_magnitude.shape[1],
        hybrid_magnitude.shape[1]
    )

    original_magnitude = (
        original_magnitude[:, :frames]
    )

    original_phase = (
        original_phase[:, :frames]
    )

    hybrid_magnitude = (
        hybrid_magnitude[:, :frames]
    )

    # Relative spectral change.
    ratio = (
        hybrid_magnitude
        /
        np.maximum(
            original_magnitude,
            1e-6
        )
    )

    # Strong protection against artifacts.
    ratio = np.clip(
        ratio,
        0.90,
        1.10
    )

    # Apply only a tiny controlled change.
    final_magnitude = (
        original_magnitude
        * (
            1.0
            + AI_STRENGTH
            * (ratio - 1.0)
        )
    )

    final_stft = (
        final_magnitude
        * np.exp(
            1j * original_phase
        )
    )

    audio = librosa.istft(
        final_stft,
        hop_length=HOP_LENGTH,
        win_length=N_FFT
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_audio(audio):

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = (
            audio
            / peak
            * 0.90
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ARTIFACT-SUPPRESSED AI BIRDSONG SYNTHESIS")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Species: {SPECIES}"
    )

    print(
        f"AI strength: {AI_STRENGTH}"
    )

    print(
        f"Latent noise: {LATENT_NOISE}"
    )

    print(
        f"Vocal threshold: {VOCAL_THRESHOLD}"
    )

    print(
        f"Spectral smoothing: "
        f"{FREQUENCY_SMOOTHING}, "
        f"{TIME_SMOOTHING}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source_path = find_source_audio()

    print(
        "\nSource recording:"
    )

    print(source_path)

    y, _ = librosa.load(
        source_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=5.0
    )

    # --------------------------------------------------------
    # ORIGINAL MEL
    # --------------------------------------------------------

    print(
        "\nExtracting original acoustic structure..."
    )

    original_mel = audio_to_mel(
        y
    )

    original_mel = fix_mel_length(
        original_mel
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    species_index = LABEL_TO_IDX[
        SPECIES
    ]

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    print(
        "\nGenerating controlled AI variation..."
    )

    ai_mel = generate_ai_mel(
        model,
        original_mel,
        species_index
    )

    # --------------------------------------------------------
    # CLEAN + HYBRID
    # --------------------------------------------------------

    print(
        "Removing isolated spectral artifacts..."
    )

    hybrid_mel = create_hybrid_mel(
        original_mel,
        ai_mel
    )

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    print(
        "Reconstructing waveform..."
    )

    audio = reconstruct_audio(
        y,
        hybrid_mel
    )

    audio = normalize_audio(
        audio
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "ai_clean_hybrid_asian_koel.wav"
    )

    sf.write(
        output_path,
        audio,
        SAMPLE_RATE
    )

    print("\n" + "=" * 70)
    print("CLEAN HYBRID SYNTHESIS COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput:"
    )

    print(output_path)

    print(
        f"\nDuration: "
        f"{len(audio) / SAMPLE_RATE:.2f} seconds"
    )

    print(
        "\nAI spectral modifications were:"
    )

    print("  • restricted to vocal regions")
    print("  • smoothed across frequency/time")
    print("  • limited to small changes")
    print("  • reconstructed with original phase")

    print("=" * 70)


if __name__ == "__main__":
    main()