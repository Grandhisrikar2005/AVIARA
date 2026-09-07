from pathlib import Path
import random

import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

from generation.vae import BirdsongConditionalVAE


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    ROOT
    / "models"
    / "vae_checkpoints"
    / "vae_best_model.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "natural_generation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO CONFIG
# ============================================================

SR = 32000
DURATION = 5.0

N_FFT = 1024
HOP_LENGTH = 320

N_MELS = 128
FMIN = 50
FMAX = 14000

LATENT_DIM = 64
SPECIES_EMBEDDING_DIM = 16

# How strongly the generated VAE spectrum influences
# the original natural recording.
#
# Lower = more natural
# Higher = more AI variation
AI_STRENGTH = 0.35


# ============================================================
# EXACT V2 LABEL MAPPING
# ============================================================

SPECIES = [
    ("Eudynamys scolopaceus", "Asian Koel"),
    ("Milvus migrans", "Black Kite"),
    ("Acridotheres tristis", "Common Myna"),
    ("Orthotomus sutorius", "Common Tailorbird"),
    ("Centropus sinensis", "Greater Coucal"),
    ("Corvus splendens", "House Crow"),
    ("Pavo cristatus", "Indian Peafowl"),
    ("Pycnonotus cafer", "Red-vented Bulbul"),
    ("Psittacula krameri", "Rose-ringed Parakeet"),
    ("Halcyon smyrnensis", "White-throated Kingfisher"),
]

LABEL_TO_IDX = {
    scientific: i
    for i, (scientific, common) in enumerate(SPECIES)
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading Conditional VAE...")

    model = BirdsongConditionalVAE(
        num_classes=10,
        latent_dim=LATENT_DIM,
        species_embedding_dim=SPECIES_EMBEDDING_DIM,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("VAE loaded successfully.")

    return model


# ============================================================
# FIND REAL RECORDING
# ============================================================

def find_recording():

    processed_dir = ROOT / "dataset" / "processed"

    files = list(
        processed_dir.rglob("*.wav")
    )

    if not files:
        raise FileNotFoundError(
            "No WAV files found inside dataset/processed/"
        )

    # Prefer Asian Koel for the demo.
    preferred = [
        f
        for f in files
        if "Eudynamys_scolopaceus" in str(f)
    ]

    if preferred:
        files = preferred

    random.seed(42)

    return random.choice(files)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True,
    )

    target = int(SR * DURATION)

    if len(y) < target:

        y = np.pad(
            y,
            (0, target - len(y)),
        )

    else:

        y = y[:target]

    y = y.astype(np.float32)

    return y


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def audio_to_mel_db(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80,
    )

    return mel_db.astype(np.float32)


# ============================================================
# NORMALIZE TO MODEL REPRESENTATION
# ============================================================

def prepare_model_input(mel_db):

    # Your training pipeline uses dB values.
    x = torch.tensor(
        mel_db,
        dtype=torch.float32,
    )

    x = x.unsqueeze(0).unsqueeze(0)

    return x.to(DEVICE)


# ============================================================
# VAE LATENT VARIATION
# ============================================================

@torch.no_grad()
def generate_variation(
    model,
    mel_input,
    species_idx,
):

    species_tensor = torch.tensor(
        [species_idx],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Encode real birdsong
    # --------------------------------------------------------

    encoded = model.encode(
        mel_input,
        species_tensor,
    )

    # Support common return formats.
    if isinstance(encoded, tuple):

        mu = encoded[0]

    else:

        mu = encoded

    # --------------------------------------------------------
    # Controlled variation
    # --------------------------------------------------------

    noise = torch.randn_like(mu)

    # Small movement around the real recording.
    latent = mu + 0.20 * noise

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated = model.decode(
        latent,
        species_tensor,
    )

    generated = generated.squeeze()
    generated = generated.cpu().numpy()

    generated = np.nan_to_num(
        generated,
        nan=-80.0,
        posinf=0.0,
        neginf=-80.0,
    )

    generated = np.clip(
        generated,
        -80.0,
        0.0,
    )

    return generated


# ============================================================
# NATURAL-PHASE SPECTRAL GUIDANCE
# ============================================================

def create_natural_variation(
    original_audio,
    generated_mel_db,
):

    # --------------------------------------------------------
    # Original STFT
    # --------------------------------------------------------

    original_stft = librosa.stft(
        original_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )

    original_mag = np.abs(
        original_stft
    )

    original_phase = (
        original_stft
        / (original_mag + 1e-8)
    )

    # --------------------------------------------------------
    # Generated Mel → linear-frequency magnitude
    # --------------------------------------------------------

    generated_mel_power = librosa.db_to_power(
        generated_mel_db,
        ref=1.0,
    )

    generated_mel_power = np.maximum(
        generated_mel_power,
        1e-10,
    )

    generated_stft_mag = (
        librosa.feature.inverse.mel_to_stft(
            generated_mel_power,
            sr=SR,
            n_fft=N_FFT,
            power=2.0,
            fmin=FMIN,
            fmax=FMAX,
        )
    )

    # --------------------------------------------------------
    # Match dimensions
    # --------------------------------------------------------

    min_frames = min(
        original_mag.shape[1],
        generated_stft_mag.shape[1],
    )

    original_mag = (
        original_mag[:, :min_frames]
    )

    original_phase = (
        original_phase[:, :min_frames]
    )

    generated_stft_mag = (
        generated_stft_mag[:, :min_frames]
    )

    # --------------------------------------------------------
    # Log-magnitude blending
    #
    # This is much more stable than directly adding
    # magnitudes.
    # --------------------------------------------------------

    original_log = np.log1p(
        original_mag
    )

    generated_log = np.log1p(
        generated_stft_mag
    )

    blended_log = (
        (1.0 - AI_STRENGTH)
        * original_log
        +
        AI_STRENGTH
        * generated_log
    )

    blended_mag = np.expm1(
        blended_log
    )

    # --------------------------------------------------------
    # Preserve NATURAL phase
    # --------------------------------------------------------

    final_stft = (
        blended_mag
        * original_phase
    )

    # --------------------------------------------------------
    # Inverse STFT
    # --------------------------------------------------------

    output = librosa.istft(
        final_stft,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        length=len(original_audio),
    )

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Remove DC.
    output = output - np.mean(output)

    # Normalize.
    peak = np.max(
        np.abs(output)
    )

    if peak > 1e-8:
        output = output / peak

    # Leave headroom.
    output *= 0.90

    return output.astype(
        np.float32
    )


# ============================================================
# SAVE SPECTROGRAM
# ============================================================

def save_plot(
    original_mel,
    generated_mel,
    output_audio,
):

    plt.figure(
        figsize=(13, 5)
    )

    librosa.display.specshow(
        generated_mel,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        "VAE-Generated Acoustic Structure"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "generated_spectrogram.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NATURAL-PHASE AI BIRDSONG SYNTHESIS")
    print("=" * 70)

    model = load_model()

    # --------------------------------------------------------
    # Select real recording
    # --------------------------------------------------------

    source_path = find_recording()

    print()
    print("Source recording:")
    print(source_path)

    # Try to determine species from path.
    species_idx = 0
    species_name = "Asian Koel"

    for idx, (scientific, common) in enumerate(SPECIES):

        if scientific.replace(
            " ",
            "_",
        ) in str(source_path):

            species_idx = idx
            species_name = common
            break

    print(
        f"Detected source species: {species_name}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    audio = load_audio(
        source_path
    )

    # --------------------------------------------------------
    # Mel
    # --------------------------------------------------------

    mel_db = audio_to_mel_db(
        audio
    )

    model_input = prepare_model_input(
        mel_db
    )

    # --------------------------------------------------------
    # VAE variation
    # --------------------------------------------------------

    generated_mel = generate_variation(
        model,
        model_input,
        species_idx,
    )

    # --------------------------------------------------------
    # Natural-phase reconstruction
    # --------------------------------------------------------

    output_audio = create_natural_variation(
        audio,
        generated_mel,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "ai_natural_birdsong.wav"
    )

    sf.write(
        output_path,
        output_audio,
        SR,
        subtype="PCM_16",
    )

    # Save source too.
    sf.write(
        OUTPUT_DIR
        / "source_birdsong.wav",
        audio,
        SR,
        subtype="PCM_16",
    )

    # Save generated Mel.
    np.save(
        OUTPUT_DIR
        / "generated_mel.npy",
        generated_mel,
    )

    save_plot(
        mel_db,
        generated_mel,
        output_audio,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=output_audio
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=output_audio,
        sr=SR,
    )[0]

    print()
    print("=" * 70)
    print("GENERATION SUCCESSFUL")
    print("=" * 70)

    print(
        f"Species       : {species_name}"
    )

    print(
        f"Duration      : "
        f"{len(output_audio) / SR:.2f} sec"
    )

    print(
        f"RMS energy    : "
        f"{np.mean(rms):.4f}"
    )

    print(
        f"Centroid      : "
        f"{np.mean(centroid):.2f} Hz"
    )

    print()
    print(
        "AI birdsong:"
    )

    print(
        output_path
    )

    print()
    print(
        "AI strength:",
        AI_STRENGTH
    )

    print("=" * 70)


if __name__ == "__main__":
    main()