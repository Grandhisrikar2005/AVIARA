from pathlib import Path
import sys
import inspect
import io
import tempfile
import re

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.communication_interpreter import (
    interpret_birdsong,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aviara | AI Bioacoustic Lab",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "v2_best_model.pth"
)

VAE_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "vae_checkpoints"
    / "vae_normalized_best_model.pth"
)

TEST_EMBEDDINGS = (
    PROJECT_ROOT
    / "outputs"
    / "embeddings"
    / "test_embeddings.csv"
)

TEST_ACOUSTIC = (
    PROJECT_ROOT
    / "outputs"
    / "acoustic_analysis"
    / "test_acoustic_features.csv"
)

GENERATED_AUDIO_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_birdsong"
    / "audio"
)


# ============================================================
# AUDIO PARAMETERS
# ============================================================

SAMPLE_RATE = 32000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320
FMIN = 50
FMAX = 14000
AUDIO_DURATION = 5
LATENT_DIM = 64


# ============================================================
# SPECIES
# ============================================================

SPECIES = {
    0: {
        "scientific": "Eudynamys scolopaceus",
        "common": "Asian Koel",
        "symbol": "🌿",
    },
    1: {
        "scientific": "Milvus migrans",
        "common": "Black Kite",
        "symbol": "🪶",
    },
    2: {
        "scientific": "Acridotheres tristis",
        "common": "Common Myna",
        "symbol": "🎵",
    },
    3: {
        "scientific": "Orthotomus sutorius",
        "common": "Common Tailorbird",
        "symbol": "🌱",
    },
    4: {
        "scientific": "Centropus sinensis",
        "common": "Greater Coucal",
        "symbol": "🌳",
    },
    5: {
        "scientific": "Corvus splendens",
        "common": "House Crow",
        "symbol": "🌑",
    },
    6: {
        "scientific": "Pavo cristatus",
        "common": "Indian Peafowl",
        "symbol": "🦚",
    },
    7: {
        "scientific": "Pycnonotus cafer",
        "common": "Red-vented Bulbul",
        "symbol": "🌺",
    },
    8: {
        "scientific": "Psittacula krameri",
        "common": "Rose-ringed Parakeet",
        "symbol": "🍃",
    },
    9: {
        "scientific": "Halcyon smyrnensis",
        "common": "White-throated Kingfisher",
        "symbol": "💧",
    },
}

SCIENTIFIC_TO_IDX = {
    value["scientific"]: key
    for key, value in SPECIES.items()
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(53, 220, 164, 0.11),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(92, 116, 255, 0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(166, 83, 255, 0.08),
            transparent 35%
        ),
        #070A12;
    color: #F4F7FB;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
}

.hero {
    padding: 2.8rem 3rem;
    border-radius: 30px;
    background:
        linear-gradient(
            135deg,
            rgba(20, 28, 43, 0.95),
            rgba(12, 19, 32, 0.82)
        );
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 25px 80px rgba(0,0,0,0.40),
        inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}

.hero:before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    right: -100px;
    top: -170px;
    background: rgba(61, 224, 166, 0.16);
    filter: blur(80px);
    border-radius: 50%;
}

.hero-eyebrow {
    color: #6EF5C0;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5vw, 5.2rem);
    line-height: 0.98;
    font-weight: 700;
    letter-spacing: -0.055em;
    margin: 0;
    background: linear-gradient(
        100deg,
        #FFFFFF 10%,
        #8AF5D0 48%,
        #9FAEFF 90%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 780px;
    margin-top: 1.2rem;
    color: #AEB9CA;
    font-size: 1.08rem;
    line-height: 1.7;
}

.pill {
    display: inline-block;
    padding: 0.4rem 0.75rem;
    margin-right: 0.45rem;
    margin-top: 1.1rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    color: #CBD4E2;
    font-size: 0.78rem;
}

.card {
    background:
        linear-gradient(
            145deg,
            rgba(22, 30, 45, 0.88),
            rgba(12, 18, 29, 0.88)
        );
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 22px;
    padding: 1.35rem;
    box-shadow:
        0 15px 45px rgba(0,0,0,0.20),
        inset 0 1px 0 rgba(255,255,255,0.035);
}

.metric-card {
    background: rgba(20, 28, 42, 0.80);
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 18px;
    padding: 1.1rem;
    min-height: 115px;
}

.metric-label {
    color: #8996AA;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}

.metric-value {
    color: #F4F7FB;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 0.35rem;
}

.section-description {
    color: #8F9BAD;
    margin-bottom: 1.2rem;
}

.species-result {
    padding: 1.8rem;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(57, 218, 163, 0.13),
            rgba(79, 94, 221, 0.10)
        );
    border: 1px solid rgba(110,245,192,0.18);
}

.species-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
}

.species-scientific {
    color: #8FA0B8;
    font-style: italic;
    margin-top: 0.2rem;
}

.confidence {
    color: #6EF5C0;
    font-size: 2.7rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}

.status-good {
    color: #6EF5C0;
}

.status-muted {
    color: #8996AA;
}

div[data-testid="stFileUploader"] {
    background: rgba(18, 25, 38, 0.65);
    border: 1px dashed rgba(110,245,192,0.35);
    border-radius: 20px;
    padding: 0.8rem;
}

.stButton > button {
    border-radius: 13px;
    border: 1px solid rgba(110,245,192,0.25);
    background:
        linear-gradient(
            135deg,
            rgba(69, 226, 170, 0.18),
            rgba(79, 101, 239, 0.18)
        );
    color: #F4F7FB;
    font-weight: 700;
    min-height: 2.8rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(110,245,192,0.55);
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(69,226,170,0.10);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #090D16,
            #0B101A
        );
    border-right: 1px solid rgba(255,255,255,0.06);
}

.sidebar-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #FFFFFF;
}

.sidebar-caption {
    color: #78869A;
    font-size: 0.82rem;
    line-height: 1.5;
}

.small-note {
    color: #6F7D91;
    font-size: 0.78rem;
    line-height: 1.5;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: #657286;
    font-size: 0.78rem;
    text-align: center;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_audio(audio):
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def prepare_audio(audio_bytes):
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True,
    )

    target_length = (
        SAMPLE_RATE * AUDIO_DURATION
    )

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (
                0,
                target_length - len(audio),
            ),
        )
    else:
        audio = audio[:target_length]

    return normalize_audio(audio)


def create_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
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
    )

    return mel_db.astype(
        np.float32
    )


def extract_acoustic_features(audio):
    rms = librosa.feature.rms(
        y=audio
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=audio
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE,
        roll_percent=0.85,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    magnitude = np.abs(stft)

    frequencies = librosa.fft_frequencies(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
    )

    mask = (
        (frequencies >= FMIN)
        & (frequencies <= FMAX)
    )

    valid_freq = frequencies[mask]
    valid_mag = magnitude[mask]

    dominant = valid_freq[
        np.argmax(
            valid_mag,
            axis=0,
        )
    ]

    return {
        "RMS Energy": float(np.mean(rms)),
        "Energy Variation": float(np.std(rms)),
        "Zero Crossing Rate": float(np.mean(zcr)),
        "Spectral Centroid": float(np.mean(centroid)),
        "Spectral Bandwidth": float(np.mean(bandwidth)),
        "Spectral Rolloff": float(np.mean(rolloff)),
        "Spectral Flatness": float(np.mean(flatness)),
        "Dominant Frequency": float(np.mean(dominant)),
    }


def spectrogram_figure(
    mel_db,
    title,
):
    fig, ax = plt.subplots(
        figsize=(12, 4.7)
    )

    ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[
            0,
            AUDIO_DURATION,
            0,
            FMAX / 1000,
        ],
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Frequency (kHz)"
    )

    ax.grid(
        alpha=0.08
    )

    fig.tight_layout()

    return fig


# ============================================================
# MODEL DISCOVERY
# ============================================================

@st.cache_resource
def load_classifier():

    try:
        import training.model as model_module

        candidates = []

        for name, obj in vars(
            model_module
        ).items():

            if (
                inspect.isclass(obj)
                and issubclass(
                    obj,
                    nn.Module,
                )
                and obj is not nn.Module
                and obj.__module__
                == model_module.__name__
            ):
                candidates.append(obj)

        if not candidates:
            raise RuntimeError(
                "No neural-network model class "
                "was found in training/model.py"
            )

        # Prefer a class containing classifier/model keywords.
        preferred = sorted(
            candidates,
            key=lambda cls: (
                not any(
                    word in cls.__name__.lower()
                    for word in [
                        "bird",
                        "transformer",
                        "classifier",
                        "cnn",
                    ]
                ),
                cls.__name__,
            ),
        )

        model_class = preferred[0]

        # Try common constructor patterns.
        attempts = [
            {},
            {"num_classes": 10},
            {"num_species": 10},
            {"n_classes": 10},
            {"num_classes": 10, "embedding_dim": 128},
            {"num_species": 10, "embedding_dim": 128},
        ]

        model = None

        last_error = None

        for kwargs in attempts:

            try:
                model = model_class(
                    **kwargs
                )
                break

            except Exception as error:
                last_error = error

        if model is None:
            raise RuntimeError(
                f"Could not instantiate "
                f"{model_class.__name__}: "
                f"{last_error}"
            )

        checkpoint = torch.load(
            CLASSIFIER_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        if isinstance(
            checkpoint,
            dict,
        ):

            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get(
                    "state_dict",
                    checkpoint,
                ),
            )

        else:
            state_dict = checkpoint

        model.load_state_dict(
            state_dict
        )

        model.to(DEVICE)
        model.eval()

        return model, model_class.__name__

    except Exception as error:

        return None, str(error)


@st.cache_resource
def load_vae():

    try:

        from generation.vae import (
            BirdsongConditionalVAE
        )

        model = BirdsongConditionalVAE(
            num_classes=10,
            latent_dim=64,
            species_embedding_dim=16,
        )

        checkpoint = torch.load(
            VAE_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        if isinstance(
            checkpoint,
            dict,
        ):

            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get(
                    "state_dict",
                    checkpoint,
                ),
            )

        else:
            state_dict = checkpoint

        model.load_state_dict(
            state_dict
        )

        model.to(DEVICE)
        model.eval()

        return model

    except Exception as error:

        return None, str(error)


# ============================================================
# CLASSIFIER INFERENCE
# ============================================================

def classify_audio(
    model,
    mel_db,
):

    tensor = torch.from_numpy(
        mel_db
    ).unsqueeze(
        0
    ).unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        output = model(
            tensor
        )

        if isinstance(
            output,
            tuple,
        ):

            logits = output[0]

            embedding = (
                output[1]
                if len(output) > 1
                else None
            )

        else:

            logits = output
            embedding = None

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    return (
        probabilities.cpu().numpy(),
        None
        if embedding is None
        else embedding.squeeze().cpu().numpy(),
    )


# ============================================================
# HYBRID AI-ASSISTED SYNTHESIS
# ============================================================

def normalize_vae_mel(mel_db):
    """Map -80..0 dB into the 0..1 range used by the normalized VAE."""
    normalized = (mel_db + 80.0) / 80.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def denormalize_vae_mel(normalized):
    """Map normalized VAE output back to -80..0 dB."""
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 80.0 - 80.0).astype(np.float32)


def smooth_mask(mask, sigma=8):
    """Smooth a frame mask so edits fade in/out naturally."""
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(mask.astype(np.float32), sigma=sigma)
    maximum = float(np.max(smoothed))

    if maximum > 1e-8:
        smoothed /= maximum

    return np.clip(smoothed, 0.0, 1.0)


def hybrid_ai_synthesis(
    vae,
    audio,
    species_index,
    seed,
    ai_strength=0.045,
    latent_noise=0.055,
    vocal_threshold=0.30,
):
    """
    Generate a controlled AI-assisted acoustic variation.

    The original waveform remains the acoustic anchor:
      original waveform
            +
      subtle VAE-derived spectral variation
            +
      original STFT phase

    This avoids directly converting a fully synthetic Mel spectrogram
    into audio, which can introduce strong reconstruction artifacts.
    """

    from scipy.ndimage import gaussian_filter

    # The function receives an already prepared waveform.
    original_audio = np.asarray(audio, dtype=np.float32)

    # --------------------------------------------------------
    # Original Mel spectrogram
    # --------------------------------------------------------
    original_mel_db = create_mel(original_audio)
    original_vae_mel = normalize_vae_mel(original_mel_db)

    input_tensor = torch.from_numpy(
        original_vae_mel
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    species_tensor = torch.tensor(
        [species_index],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Encode the real recording, then make a very small
    # latent perturbation around its learned representation.
    # --------------------------------------------------------
    with torch.no_grad():
        _, mu, _, _ = vae(
            input_tensor,
            species_tensor,
        )

        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(int(seed))

        noise = torch.randn(
            mu.shape,
            generator=generator,
            device=DEVICE,
        ) * latent_noise

        latent = mu + noise

        decoded = vae.decode(
            latent,
            species_tensor,
        )

    generated_norm = (
        decoded
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    generated_mel_db = denormalize_vae_mel(
        np.nan_to_num(
            generated_norm,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    )

    # Match dimensions exactly.
    if generated_mel_db.shape != original_mel_db.shape:
        generated_mel_db = np.asarray(
            librosa.util.fix_length(
                generated_mel_db,
                size=original_mel_db.shape[1],
                axis=1,
            ),
            dtype=np.float32,
        )

        if generated_mel_db.shape[0] != original_mel_db.shape[0]:
            generated_mel_db = np.resize(
                generated_mel_db,
                original_mel_db.shape,
            ).astype(np.float32)

    # --------------------------------------------------------
    # Detect stronger vocal regions.
    # AI edits happen mostly where the recording contains
    # meaningful acoustic activity.
    # --------------------------------------------------------
    frame_energy = librosa.feature.rms(
        y=original_audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    energy_floor = float(np.percentile(frame_energy, 35))
    energy_span = float(
        np.percentile(frame_energy, 90) - energy_floor
    )

    if energy_span <= 1e-8:
        vocal_mask = np.ones_like(frame_energy)
    else:
        vocal_mask = np.clip(
            (frame_energy - energy_floor)
            / energy_span,
            0.0,
            1.0,
        )

    # Match Mel frame count.
    target_frames = original_mel_db.shape[1]
    if len(vocal_mask) != target_frames:
        x_old = np.linspace(
            0.0,
            1.0,
            len(vocal_mask),
        )
        x_new = np.linspace(
            0.0,
            1.0,
            target_frames,
        )
        vocal_mask = np.interp(
            x_new,
            x_old,
            vocal_mask,
        )

    vocal_mask = np.clip(
        (vocal_mask - vocal_threshold)
        / max(1e-6, 1.0 - vocal_threshold),
        0.0,
        1.0,
    )

    vocal_mask = smooth_mask(
        vocal_mask,
        sigma=7,
    )

    vocal_mask_2d = vocal_mask[None, :]

    # --------------------------------------------------------
    # Compute only a tiny spectral change.
    # --------------------------------------------------------
    mel_delta = (
        generated_mel_db
        - original_mel_db
    )

    # Smooth the model's spectral proposal to suppress
    # isolated bins and frame-to-frame ringing.
    mel_delta = gaussian_filter(
        mel_delta.astype(np.float32),
        sigma=(1.2, 1.4),
    )

    # Very tight cap keeps the variation subtle.
    mel_delta = np.clip(
        mel_delta,
        -4.0,
        4.0,
    )

    hybrid_mel_db = (
        original_mel_db
        + mel_delta
        * vocal_mask_2d
        * ai_strength
        * 8.0
    )

    # Keep the Mel representation close to the source.
    hybrid_mel_db = np.clip(
        hybrid_mel_db,
        -80.0,
        0.0,
    )

    # --------------------------------------------------------
    # Reconstruct only magnitude.
    # Preserve the original phase for natural temporal
    # structure and transient alignment.
    # --------------------------------------------------------
    original_stft = librosa.stft(
        original_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )

    original_magnitude = np.abs(original_stft)
    original_phase = np.angle(original_stft)

    hybrid_magnitude = librosa.feature.inverse.mel_to_stft(
        librosa.db_to_power(hybrid_mel_db),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        power=2.0,
        fmin=FMIN,
        fmax=FMAX,
    )

    # Match STFT frame count.
    target_stft_frames = original_magnitude.shape[1]

    if hybrid_magnitude.shape[1] != target_stft_frames:
        hybrid_magnitude = librosa.util.fix_length(
            hybrid_magnitude,
            size=target_stft_frames,
            axis=1,
        )

    if hybrid_magnitude.shape[0] != original_magnitude.shape[0]:
        hybrid_magnitude = np.resize(
            hybrid_magnitude,
            original_magnitude.shape,
        )

    # Convert the magnitude estimate into a small, bounded ratio.
    ratio = (
        hybrid_magnitude
        / np.maximum(
            original_magnitude,
            1e-6,
        )
    )

    ratio = np.clip(
        ratio,
        0.96,
        1.04,
    )

    # Apply the modification gently and mainly inside active
    # vocal regions.
    stft_vocal_mask = np.interp(
        np.linspace(
            0.0,
            1.0,
            target_stft_frames,
        ),
        np.linspace(
            0.0,
            1.0,
            len(vocal_mask),
        ),
        vocal_mask,
    )

    ratio = (
        1.0
        + (ratio - 1.0)
        * stft_vocal_mask[None, :]
    )

    final_magnitude = (
        original_magnitude
        * ratio
    )

    final_magnitude = np.maximum(
        final_magnitude,
        0.0,
    )

    reconstructed = librosa.istft(
        final_magnitude
        * np.exp(1j * original_phase),
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        length=len(original_audio),
    )

    reconstructed = normalize_audio(
        reconstructed.astype(np.float32)
    )

    return hybrid_mel_db.astype(np.float32), reconstructed


# ============================================================
# PLOT CONFIDENCE
# ============================================================

def confidence_chart(
    probabilities
):

    names = [
        SPECIES[i]["common"]
        for i in range(10)
    ]

    values = probabilities * 100

    order = np.argsort(values)

    fig = go.Figure(
        go.Bar(
            x=values[order],
            y=np.array(names)[order],
            orientation="h",
            text=[
                f"{value:.1f}%"
                for value in values[order]
            ],
            textposition="outside",
            hovertemplate=(
                "%{y}: %{x:.2f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(
            l=10,
            r=60,
            t=20,
            b=20,
        ),
        xaxis=dict(
            title="Confidence (%)",
            range=[
                0,
                max(
                    100,
                    values.max() + 10,
                ),
            ],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title="",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#D8E0EC"
        ),
        showlegend=False,
    )

    return fig


# ============================================================
# EMBEDDING RADAR
# ============================================================

def embedding_fingerprint(
    embedding
):

    if embedding is None:
        return None

    # 128 dimensions grouped into 16 regions.
    chunks = np.array_split(
        embedding,
        16,
    )

    values = [
        float(
            np.mean(
                np.abs(chunk)
            )
        )
        for chunk in chunks
    ]

    labels = [
        f"Z{i + 1}"
        for i in range(16)
    ]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            hovertemplate=(
                "%{theta}: %{r:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
            ),
        ),
        height=400,
        margin=dict(
            l=30,
            r=30,
            t=30,
            b=30,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#D8E0EC"
        ),
        showlegend=False,
    )

    return fig




# ============================================================
# COMMUNICATION METADATA LOOKUP
# ============================================================

COMMUNICATION_MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "communication_manifest.csv"
)

COMMUNICATION_TYPE_MAP = {
    "song": "song",
    "song, displaying": "song",
    "displaying": "song",
    "call": "contact_social",
    "contact": "contact_social",
    "contact call": "contact_social",
    "contact/social": "contact_social",
    "contact social": "contact_social",
    "contact_social": "contact_social",
    "alarm": "alarm_warning",
    "alarm call": "alarm_warning",
    "warning": "alarm_warning",
    "alarm/warning": "alarm_warning",
    "alarm_warning": "alarm_warning",
    "flight call": "contact_social",
    "flight_call": "contact_social",
    "call, flight": "contact_social",
    "begging": "begging_distress",
    "begging call": "begging_distress",
    "begging/distress": "begging_distress",
    "distress": "begging_distress",
    "begging_distress": "begging_distress",
}

COMMUNICATION_DISPLAY = {
    "song": "Song",
    "contact_social": "Contact / Social Call",
    "alarm_warning": "Alarm / Warning",
    "begging_distress": "Begging / Distress Call",
    "uncertain": "Uncertain",
}


def _clean_text(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _recording_id_from_name(name):
    match = re.search(r"XC(\d+)", _clean_text(name), re.IGNORECASE)
    return match.group(1) if match else None


def _normalize_communication_type(value):
    raw = _clean_text(value).lower()
    if not raw:
        return None

    # Exact normalized labels first.
    if raw in COMMUNICATION_TYPE_MAP:
        return COMMUNICATION_TYPE_MAP[raw]

    # Xeno-canto may provide multiple labels in one field.
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    for part in parts:
        if part in COMMUNICATION_TYPE_MAP:
            return COMMUNICATION_TYPE_MAP[part]

    # Conservative keyword fallback for unusual formatting.
    if "alarm" in raw or "warning" in raw:
        return "alarm_warning"

    if "begging" in raw or "distress" in raw:
        return "begging_distress"

    if "contact" in raw:
        return "contact_social"

    if "flight" in raw:
        return "contact_social"

    if "song" in raw or "display" in raw:
        return "song"

    if raw == "call":
        return "contact_social"

    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_communication_metadata():
    """Load communication annotations with robust ID/filename matching.

    The cleaned project manifest is preferred, with raw Xeno-canto metadata
    used as a fallback. The returned dataframe also exposes annotation text
    so the behavioral interpreter can use the original vocalization label.
    """
    frames = []

    if COMMUNICATION_MANIFEST.exists():
        try:
            df = pd.read_csv(COMMUNICATION_MANIFEST)
            if not df.empty:
                df["_metadata_source"] = "communication_manifest.csv"
                frames.append(df)
        except Exception:
            pass

    raw_root = PROJECT_ROOT / "dataset" / "raw" / "pilot"
    if raw_root.exists():
        for metadata_file in raw_root.rglob("metadata.csv"):
            try:
                df = pd.read_csv(metadata_file)
                if not df.empty:
                    df["_metadata_source"] = str(metadata_file.relative_to(PROJECT_ROOT))
                    frames.append(df)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def _candidate_columns(df, names):
    lower_map = {str(c).lower(): c for c in df.columns}
    return [lower_map[name] for name in names if name in lower_map]


def _annotation_text_from_row(row):
    """Collect original annotation/description text without inventing meaning."""
    preferred = [
        "annotation",
        "remarks",
        "remark",
        "comments",
        "comment",
        "description",
        "notes",
        "note",
        "meaning",
        "english",
        "type",
    ]

    parts = []
    existing = {str(col).lower(): col for col in row.index}

    for name in preferred:
        col = existing.get(name)
        if col is None:
            continue
        value = row.get(col, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())

    return " | ".join(dict.fromkeys(parts))


def _row_to_annotation(row, xc_id=None):
    type_columns = _candidate_columns(
        row.to_frame().T,
        ["communication_type", "vocalization_type", "type", "sound_type", "call_type"],
    )

    raw_type = ""
    for column in type_columns:
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            raw_type = str(value).strip()
            break

    normalized = _normalize_communication_type(raw_type)
    if not normalized:
        return None

    return {
        "xc_id": xc_id or _recording_id_from_name(row.get("id", "")),
        "raw_type": raw_type,
        "normalized_type": normalized,
        "type": normalized,
        "annotation_text": _annotation_text_from_row(row),
        "source": _clean_text(row.get("_metadata_source", "metadata")),
    }


def find_communication_annotation(filename):
    """Match an uploaded recording to its strongest available metadata annotation."""
    df = load_communication_metadata()
    if df.empty:
        return None

    filename = _clean_text(filename)
    filename_lower = filename.lower()
    basename_lower = Path(filename_lower).name
    xc_id = _recording_id_from_name(filename)

    id_columns = _candidate_columns(
        df,
        ["id", "recording_id", "xc_id", "recording_number", "nr", "recording"],
    )
    filename_columns = _candidate_columns(
        df,
        [
            "source_file",
            "filename",
            "file_name",
            "audio_filename",
            "original_filename",
            "recording_filename",
            "file",
            "name",
            "segment_id",
        ],
    )

    # 1) Exact Xeno-canto recording ID.
    if xc_id:
        for column in id_columns:
            values = df[column].astype(str).str.extract(r"(\d+)", expand=False)
            matches = df[values == xc_id]
            if not matches.empty:
                result = _row_to_annotation(matches.iloc[0], xc_id=xc_id)
                if result:
                    result["match"] = f"Xeno-canto ID {xc_id}"
                    return result

    # 2) Exact filename/basename.
    for column in filename_columns:
        values = df[column].astype(str).str.lower()
        exact = df[values == filename_lower]
        if exact.empty:
            exact = df[values.str.replace(r".*[\\/]", "", regex=True) == basename_lower]
        if not exact.empty:
            result = _row_to_annotation(exact.iloc[0], xc_id=xc_id)
            if result:
                result["match"] = f"Filename match via {column}"
                return result

    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🐦 AVIARA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-caption">'
        'AI Bioacoustic Intelligence Lab'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Species Identification",
            "Acoustic Intelligence",
            "Communication Interpreter",
            "Birdsong Synthesizer",
            "About the Model",
        ],
    )

    st.divider()

    st.markdown(
        "**SYSTEM STATUS**"
    )

    st.caption(
        f"Compute: {DEVICE.type.upper()}"
    )

    st.caption(
        "CNN + Transformer"
    )

    st.caption(
        "Conditional VAE"
    )

    st.caption(
        "10 species"
    )

    st.divider()

    st.markdown(
        '<div class="small-note">'
        'Scientific note: the system performs '
        'acoustic classification and structural '
        'analysis. Generated sounds represent '
        'learned acoustic patterns and are not '
        'claimed to be literal translations of '
        'bird communication.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-eyebrow">
AI BIOACOUSTIC INTELLIGENCE · DEEP LEARNING
</div>

<div class="hero-title">
Listen beyond<br>
the waveform.
</div>

<div class="hero-subtitle">
A multimodal deep-learning laboratory for
birdsong identification, acoustic structure
analysis, evidence-aware communication
interpretation, latent representation learning,
and species-conditioned sound synthesis.
</div>

<span class="pill">CNN</span>
<span class="pill">TRANSFORMER</span>
<span class="pill">128-D EMBEDDINGS</span>
<span class="pill">COMMUNICATION INTERPRETER</span>
<span class="pill">CONDITIONAL VAE</span>
<span class="pill">SOUND SYNTHESIS</span>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        'The Bioacoustic Observatory'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'One interface connecting perception, '
        'representation, analysis and generation.'
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    overview_metrics = [
        ("10", "Species"),
        ("77.01%", "Test Accuracy"),
        ("64", "VAE Latent Dimensions"),
        ("128", "Birdsong Embedding"),
        ("30", "Offline Generated Samples"),
    ]

    for col, (
        value,
        label,
    ) in zip(
        cols,
        overview_metrics,
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        [1.25, 0.75]
    )

    with col1:

        st.markdown(
            """
            <div class="card">

            ### 🧬 From sound to representation

            Every recording passes through a carefully
            designed acoustic pipeline:

            **Waveform → Mel Spectrogram → CNN → Transformer
            → 128-D Birdsong Representation**

            The learned representation captures acoustic
            structure useful for species identification and
            downstream analysis.

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            ### ✨ From representation to synthesis

            The conditional VAE learns a structured
            latent space of birdsong.

            **Species + latent vector → new acoustic structure**

            The synthesis pipeline preserves the source waveform's
            phase while applying a controlled VAE-derived spectral variation.

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">'
        'Architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    st.code(
        """
Xeno-canto Recordings
        │
        ▼
Audio Preprocessing
        │
        ▼
128-bin Mel Spectrogram
        │
        ▼
CNN Feature Extractor
        │
        ▼
Transformer Encoder
        │
        ├───────────────► Species Classification
        │
        ▼
128-D Birdsong Embedding
        │
        ├───────────────► Acoustic Intelligence
        │                         │
        │                         ▼
        │                 Evidence-Aware
        │                 Communication Interpreter
        │
        ▼
Species-Conditioned VAE
        │
        ▼
64-D Latent Space
        │
        ▼
Controlled Spectral Variation
        │
        ▼
Phase-Preserving ISTFT
        │
        ▼
AI-Assisted Birdsong
        """,
        language="text",
    )


# ============================================================
# SPECIES IDENTIFICATION
# ============================================================

elif page == "Species Identification":

    st.markdown(
        '<div class="section-title">'
        '🔍 Species Identification'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Upload a bird recording and let the CNN + '
        'Transformer model infer its most likely species.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop a bird recording here",
        type=[
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
        ],
    )

    if uploaded is None:

        st.markdown(
            """
            <div class="card">

            ### 🎙️ Ready when you are

            Upload a bird recording to unlock:

            **Species prediction · Confidence distribution
            · Mel spectrogram · Acoustic fingerprint
            · 128-D representation**

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        audio_bytes = uploaded.read()

        st.session_state["latest_audio_bytes"] = audio_bytes
        st.session_state["latest_audio_name"] = uploaded.name

        audio = prepare_audio(
            audio_bytes
        )

        mel_db = create_mel(
            audio
        )

        model, model_status = (
            load_classifier()
        )

        if model is None:

            st.error(
                "Classifier could not be loaded."
            )

            st.code(
                model_status
            )

            st.stop()

        with st.spinner(
            "Analyzing acoustic structure..."
        ):

            probabilities, embedding = (
                classify_audio(
                    model,
                    mel_db,
                )
            )

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        predicted = SPECIES[
            predicted_index
        ]

        confidence = (
            probabilities[
                predicted_index
            ]
            * 100
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        result_col, audio_col = st.columns(
            [1.2, 0.8]
        )

        with result_col:

            st.markdown(
                f"""
                <div class="species-result">

                <div class="metric-label">
                    MODEL PREDICTION
                </div>

                <div class="species-name">
                    {predicted["symbol"]}
                    {predicted["common"]}
                </div>

                <div class="species-scientific">
                    {predicted["scientific"]}
                </div>

                <br>

                <div class="confidence">
                    {confidence:.1f}%
                </div>

                <div class="metric-label">
                    PREDICTION CONFIDENCE
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with audio_col:

            st.markdown(
                """
                <div class="card">
                <div class="metric-label">
                    INPUT RECORDING
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.audio(
                audio,
                sample_rate=SAMPLE_RATE,
            )

            st.caption(
                "Normalized to a 5-second analysis window."
            )

        st.markdown(
            '<div class="section-title">'
            'Confidence Landscape'
            '</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            confidence_chart(
                probabilities
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "🌈 Spectrogram",
                "🧠 Embedding",
                "📋 Raw Scores",
            ]
        )

        with tab1:

            fig = spectrogram_figure(
                mel_db,
                "Input Mel Spectrogram",
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )

        with tab2:

            if embedding is not None:

                st.plotly_chart(
                    embedding_fingerprint(
                        embedding
                    ),
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

                st.caption(
                    "128-dimensional learned birdsong "
                    "representation visualized as a "
                    "16-region acoustic fingerprint."
                )

                embedding_df = pd.DataFrame(
                    {
                        "dimension": [
                            f"z_{i+1}"
                            for i in range(
                                len(embedding)
                            )
                        ],
                        "value": embedding,
                    }
                )

                st.download_button(
                    "Download 128-D embedding",
                    embedding_df.to_csv(
                        index=False
                    ),
                    file_name=(
                        "birdsong_embedding.csv"
                    ),
                    mime="text/csv",
                )

            else:

                st.info(
                    "The current classifier did not "
                    "return an embedding."
                )

        with tab3:

            score_df = pd.DataFrame(
                {
                    "Species": [
                        SPECIES[i]["common"]
                        for i in range(10)
                    ],
                    "Scientific Name": [
                        SPECIES[i]["scientific"]
                        for i in range(10)
                    ],
                    "Confidence": (
                        probabilities * 100
                    ),
                }
            ).sort_values(
                "Confidence",
                ascending=False,
            )

            score_df[
                "Confidence"
            ] = score_df[
                "Confidence"
            ].round(3)

            st.dataframe(
                score_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# ACOUSTIC INTELLIGENCE
# ============================================================

elif page == "Acoustic Intelligence":

    st.markdown(
        '<div class="section-title">'
        '📊 Acoustic Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Measure the acoustic structure of a recording '
        'through interpretable signal-processing features.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a recording for acoustic analysis",
        type=[
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
        ],
        key="acoustic_upload",
    )

    if uploaded:

        audio_bytes = uploaded.read()

        audio = prepare_audio(
            audio_bytes
        )

        features = extract_acoustic_features(
            audio
        )

        feature_items = list(
            features.items()
        )

        for start in range(
            0,
            len(feature_items),
            4,
        ):

            row = feature_items[
                start:start + 4
            ]

            cols = st.columns(
                len(row)
            )

            for col, (
                name,
                value,
            ) in zip(
                cols,
                row,
            ):

                with col:

                    if "Hz" in name or (
                        "Centroid" in name
                        or "Bandwidth" in name
                        or "Rolloff" in name
                        or "Frequency" in name
                    ):

                        display = (
                            f"{value:,.0f} Hz"
                        )

                    else:

                        display = (
                            f"{value:.5f}"
                        )

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                {name}
                            </div>
                            <div class="metric-value">
                                {display}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<div class="section-title">'
            'Acoustic Structure'
            '</div>',
            unsafe_allow_html=True,
        )

        fig = spectrogram_figure(
            create_mel(audio),
            "Acoustic Time–Frequency Structure",
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="card">

            ### What these features mean

            **Spectral centroid** approximates where the
            spectral energy is concentrated.

            **Spectral bandwidth** measures how widely
            energy is distributed around the centroid.

            **Dominant frequency** identifies the strongest
            frequency region.

            **Zero-crossing rate** captures rapid waveform
            sign changes and can help distinguish acoustic
            textures.

            **RMS energy** describes signal intensity.

            </div>
            """,
            unsafe_allow_html=True,
        )

        feature_df = pd.DataFrame(
            [
                {
                    "Feature": key,
                    "Value": value,
                }
                for key, value in features.items()
            ]
        )

        st.download_button(
            "Download acoustic feature report",
            feature_df.to_csv(
                index=False
            ),
            file_name=(
                "acoustic_analysis.csv"
            ),
            mime="text/csv",
        )

    else:

        st.markdown(
            """
            <div class="card">

            ### 🎚️ Signal laboratory

            Upload a recording to calculate an interpretable
            acoustic fingerprint from the waveform.

            </div>
            """,
            unsafe_allow_html=True,
        )



# ============================================================
# COMMUNICATION INTERPRETER
# ============================================================

elif page == "Communication Interpreter":

    st.markdown(
        '<div class="section-title">🗣️ Communication Interpreter</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Use the bird recording, the CNN + Transformer species model, '
        'and trusted Xeno-canto vocalization annotations to produce '
        'a human-readable behavioral interpretation with a clearly labeled probable intent.'
        '</div>',
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([1.15, 0.85])

    with upload_col:
        uploaded_comm = st.file_uploader(
            "Upload a bird recording",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="communication_upload",
        )

    with info_col:
        st.markdown(
            """
            <div class="card">
                <div class="metric-label">HOW THIS WORKS</div>
                <br>
                <strong>Species model + Xeno-canto annotation + acoustic evidence</strong>
                <br><br>
                When the uploaded file belongs to the project's Xeno-canto dataset,
                AVIARA identifies its recording ID and uses the original vocalization
                annotation instead of forcing a weak acoustic heuristic to guess.
            </div>
            """,
            unsafe_allow_html=True,
        )

    source_bytes = None
    source_name = None

    if uploaded_comm is not None:
        source_bytes = uploaded_comm.read()
        source_name = uploaded_comm.name
        st.session_state["latest_audio_bytes"] = source_bytes
        st.session_state["latest_audio_name"] = source_name
    elif "latest_audio_bytes" in st.session_state:
        source_bytes = st.session_state["latest_audio_bytes"]
        source_name = st.session_state.get(
            "latest_audio_name",
            "previously analyzed recording",
        )
        st.info(f"Using the latest analyzed recording: {source_name}")

    if source_bytes is None:
        st.markdown(
            """
            <div class="card">
                <div class="metric-label">READY FOR INTERPRETATION</div>
                <br>
                Upload a bird recording to obtain:
                <br><br>
                <strong>
                    Species identification · Vocalization context ·
                    Human-readable behavioral message · Acoustic evidence
                </strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        audio = prepare_audio(source_bytes)
        mel_db = create_mel(audio)

        classifier, classifier_status = load_classifier()

        if classifier is None:
            st.error("Species classifier could not be loaded.")
            st.code(classifier_status)
            st.stop()

        with st.spinner("Analyzing species and communication context..."):
            probabilities, embedding = classify_audio(
                classifier,
                mel_db,
            )
            acoustic_features = extract_acoustic_features(audio)

        predicted_index = int(np.argmax(probabilities))
        predicted_species = SPECIES[predicted_index]
        species_confidence = float(probabilities[predicted_index])

        acoustic_payload = {
            "duration_seconds": AUDIO_DURATION,
            "rms_energy_mean": acoustic_features["RMS Energy"],
            "zero_crossing_rate": acoustic_features["Zero Crossing Rate"],
            "spectral_centroid_hz": acoustic_features["Spectral Centroid"],
            "spectral_bandwidth_hz": acoustic_features["Spectral Bandwidth"],
            "spectral_rolloff_hz": acoustic_features["Spectral Rolloff"],
            "spectral_flatness": acoustic_features["Spectral Flatness"],
            "dominant_frequency_hz": acoustic_features["Dominant Frequency"],
        }

        annotation = find_communication_annotation(source_name)

        if annotation is not None:
            result = interpret_birdsong(
                species=predicted_species["scientific"],
                communication_type=annotation["normalized_type"],
                confidence=1.0,
                acoustic_features=acoustic_payload,
                source="Xeno-canto vocalization annotation",
                annotation_text=(
                    annotation["annotation_text"]
                    or annotation["raw_type"]
                    or source_name
                ),
            )
            evidence_source = "Trusted Xeno-canto vocalization annotation"
            evidence_detail = (
                f"Matched {annotation.get('match', 'project metadata')}; "
                f"original label: {annotation['raw_type'] or annotation['normalized_type']}"
            )
        else:
            result = interpret_birdsong(
                species=predicted_species["scientific"],
                communication_type=None,
                confidence=0.0,
                acoustic_features=acoustic_payload,
                source="Acoustic fallback",
                annotation_text=source_name,
            )
            evidence_source = "Acoustic fallback"
            evidence_detail = (
                "No matching Xeno-canto vocalization annotation was found "
                "for this uploaded file, so AVIARA does not force a specific "
                "communication category."
            )

        # ----------------------------------------------------
        # HUMAN-READABLE MESSAGE FIRST
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">💬 What Is the Bird Trying to Say?</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="species-result" style="padding:30px; margin-bottom:20px;">
                <div class="metric-label">PROBABLE BEHAVIORAL MESSAGE</div>
                <div style="font-size:1.55rem; line-height:1.75; font-weight:600; color:#F4F7FB; margin-top:14px;">
                    🐦 “{result.message}”
                </div>
                <div style="font-size:0.98rem; line-height:1.7; color:#B7C2D3; margin-top:16px;">
                    {result.interpretation}
                </div>
                <div style="margin-top:20px;">
                    <span class="pill">{result.communication_display}</span>
                    <span class="pill">{result.intent_label}</span>
                    <span class="pill">{evidence_source}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        result_col, metrics_col = st.columns([1.2, 0.8])

        with result_col:
            st.markdown(
                f"""
                <div class="species-result">
                    <div class="metric-label">IDENTIFIED SPECIES</div>
                    <div class="species-name">
                        {predicted_species['symbol']} {predicted_species['common']}
                    </div>
                    <div class="species-scientific">
                        {predicted_species['scientific']}
                    </div>
                    <br>
                    <div class="confidence">
                        {species_confidence * 100:.1f}%
                    </div>
                    <div class="metric-label">SPECIES CONFIDENCE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metrics_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(
                "Communication Context",
                result.communication_display,
            )
            st.metric(
                "Evidence Level",
                result.evidence_level,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------
        # EVIDENCE SOURCE
        # ----------------------------------------------------

        if annotation is not None:
            st.success(
                f"✅ {evidence_source}: {evidence_detail}"
            )
        else:
            st.warning(
                f"⚠ {evidence_source}. {evidence_detail}"
            )

        intent_col, evidence_col = st.columns(2)

        with intent_col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">PROBABLE BEHAVIORAL INTENT</div>
                    <div class="metric-value">{result.intent_label}</div>
                    <div class="small-note" style="margin-top:10px;">
                        Interpretation confidence: {result.intent_confidence * 100:.0f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with evidence_col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">EVIDENCE SOURCE</div>
                    <div style="font-size:1.05rem; font-weight:600; color:#F4F7FB; margin-top:10px;">
                        {evidence_source}
                    </div>
                    <div class="small-note" style="margin-top:8px; line-height:1.6;">
                        {evidence_detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title">🔬 Why AVIARA thinks this</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:1rem; line-height:1.8; color:#D8E0EC;">
                    {result.acoustic_summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # ACOUSTIC FEATURES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Acoustic Evidence</div>',
            unsafe_allow_html=True,
        )

        evidence_items = [
            ("RMS Energy", f"{acoustic_features['RMS Energy']:.5f}"),
            ("Zero Crossing Rate", f"{acoustic_features['Zero Crossing Rate']:.5f}"),
            ("Spectral Centroid", f"{acoustic_features['Spectral Centroid']:,.0f} Hz"),
            ("Spectral Bandwidth", f"{acoustic_features['Spectral Bandwidth']:,.0f} Hz"),
            ("Spectral Rolloff", f"{acoustic_features['Spectral Rolloff']:,.0f} Hz"),
            ("Spectral Flatness", f"{acoustic_features['Spectral Flatness']:.5f}"),
            ("Dominant Frequency", f"{acoustic_features['Dominant Frequency']:,.0f} Hz"),
        ]

        evidence_cols = st.columns(4)

        for index, (name, value) in enumerate(evidence_items):
            with evidence_cols[index % 4]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # SPECTROGRAM + AUDIO
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🌈 Vocalization Spectrogram</div>',
            unsafe_allow_html=True,
        )

        st.pyplot(
            spectrogram_figure(
                mel_db,
                "Communication Analysis — Mel Spectrogram",
            ),
            use_container_width=True,
        )

        st.markdown(
            '<div class="section-title">🎧 Recording</div>',
            unsafe_allow_html=True,
        )

        st.audio(
            audio,
            sample_rate=SAMPLE_RATE,
        )

        # ----------------------------------------------------
        # SCIENTIFIC NOTE
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">SCIENTIFIC INTERPRETATION NOTE</div>
                <br>
                <div style="color:#C8D2E0;line-height:1.7;">
                    {result.caution}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View species confidence distribution"):
            st.plotly_chart(
                confidence_chart(probabilities),
                use_container_width=True,
                config={"displayModeBar": False},
            )


# ============================================================
# BIRDSONG SYNTHESIZER
# ============================================================

elif page == "Birdsong Synthesizer":

    st.markdown(
        '<div class="section-title">'
        '🎵 Birdsong Synthesizer'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Create a controlled AI-assisted acoustic variation '
        'while preserving the character of the uploaded recording.'
        '</div>',
        unsafe_allow_html=True,
    )

    vae_result = load_vae()

    if isinstance(vae_result, tuple):
        vae = None
        vae_error = vae_result[1]
    else:
        vae = vae_result
        vae_error = None

    if vae is None:
        st.error("VAE could not be loaded.")
        st.code(str(vae_error))
        st.stop()

    source_col, control_col = st.columns(
        [1.15, 0.85]
    )

    with source_col:

        st.markdown(
            """
            <div class="card">
            <div class="metric-label">
                SOURCE RECORDING
            </div>
            <br>
            Upload a real bird recording. The system will use
            that recording as the acoustic anchor instead of
            generating a completely synthetic waveform.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_synth = st.file_uploader(
            "Choose a source recording",
            type=[
                "wav",
                "mp3",
                "flac",
                "ogg",
                "m4a",
            ],
            key="synth_source_upload",
        )

        source_bytes = None
        source_name = None

        if uploaded_synth is not None:
            source_bytes = uploaded_synth.read()
            source_name = uploaded_synth.name

        elif "latest_audio_bytes" in st.session_state:
            source_bytes = st.session_state[
                "latest_audio_bytes"
            ]
            source_name = st.session_state.get(
                "latest_audio_name",
                "previously analyzed recording",
            )
            st.info(
                f"Using the latest analyzed recording: "
                f"{source_name}"
            )

    with control_col:

        st.markdown(
            """
            <div class="card">
            <div class="metric-label">
                SYNTHESIS CONDITION
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_common = st.selectbox(
            "Choose a species",
            [
                SPECIES[i]["common"]
                for i in range(10)
            ],
        )

        selected_index = next(
            i
            for i in range(10)
            if SPECIES[i]["common"]
            == selected_common
        )

        selected_species = SPECIES[
            selected_index
        ]

        seed = st.number_input(
            "Generation seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

        st.markdown(
            f"""
            <div class="species-result">
                <div class="species-name">
                    {selected_species["symbol"]}
                    {selected_species["common"]}
                </div>
                <div class="species-scientific">
                    {selected_species["scientific"]}
                </div>
                <br>
                <div class="metric-label">
                    GENERATIVE LATENT SPACE
                </div>
                <div class="metric-value">
                    64 dimensions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        generate = st.button(
            "✨ Generate AI-Assisted Variation",
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="card" style="margin-top:1rem;">
            <div class="metric-label">
                WHY THIS VERSION SOUNDS MORE NATURAL
            </div>
            <br>
            <strong>Real recording → VAE encoding → small latent variation
            → subtle spectral edit → original phase → WAV</strong>
            <br><br>
            The system does not replace the complete waveform with a
            Griffin-Lim reconstruction. Instead, it preserves the
            source recording and applies a small learned acoustic
            variation in stronger vocal regions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if generate:

        if source_bytes is None:
            st.warning(
                "Upload a source bird recording first, "
                "or analyze one in Species Identification."
            )
        else:

            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

            source_audio = prepare_audio(
                source_bytes
            )

            with st.spinner(
                "Encoding the recording and generating a controlled acoustic variation..."
            ):

                generated_mel, generated_audio = (
                    hybrid_ai_synthesis(
                        vae=vae,
                        audio=source_audio,
                        species_index=selected_index,
                        seed=int(seed),
                        ai_strength=0.045,
                        latent_noise=0.055,
                        vocal_threshold=0.30,
                    )
                )

            st.success(
                f"Created a 5-second AI-assisted "
                f"{selected_common} acoustic variation."
            )

            audio_col, info_col = st.columns(
                [1.25, 0.75]
            )

            with audio_col:

                st.markdown(
                    '<div class="section-title">'
                    'Generated Sound'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.audio(
                    generated_audio,
                    sample_rate=SAMPLE_RATE,
                )

                wav_buffer = io.BytesIO()

                sf.write(
                    wav_buffer,
                    generated_audio,
                    SAMPLE_RATE,
                    format="WAV",
                )

                st.download_button(
                    "⬇ Download AI-assisted WAV",
                    wav_buffer.getvalue(),
                    file_name=(
                        selected_common
                        .lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                        + "_ai_assisted.wav"
                    ),
                    mime="audio/wav",
                    use_container_width=True,
                )

            with info_col:

                generated_features = (
                    extract_acoustic_features(
                        generated_audio
                    )
                )

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "### Generation fingerprint"
                )

                st.metric(
                    "Duration",
                    "5.00 s",
                )

                st.metric(
                    "RMS energy",
                    f"{generated_features['RMS Energy']:.5f}",
                )

                st.metric(
                    "Spectral centroid",
                    f"{generated_features['Spectral Centroid']:,.0f} Hz",
                )

                st.metric(
                    "Dominant frequency",
                    f"{generated_features['Dominant Frequency']:,.0f} Hz",
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-title">'
                'Source vs AI-Assisted Structure'
                '</div>',
                unsafe_allow_html=True,
            )

            comparison_left, comparison_right = st.columns(2)

            with comparison_left:

                st.markdown(
                    '<div class="metric-label">'
                    'SOURCE RECORDING'
                    '</div>',
                    unsafe_allow_html=True,
                )

                source_mel = create_mel(
                    source_audio
                )

                source_fig = spectrogram_figure(
                    source_mel,
                    "Original Mel Spectrogram",
                )

                st.pyplot(
                    source_fig,
                    use_container_width=True,
                )

            with comparison_right:

                st.markdown(
                    '<div class="metric-label">'
                    'AI-ASSISTED OUTPUT'
                    '</div>',
                    unsafe_allow_html=True,
                )

                generated_fig = spectrogram_figure(
                    generated_mel,
                    f"AI-Assisted {selected_common}",
                )

                st.pyplot(
                    generated_fig,
                    use_container_width=True,
                )

            st.markdown(
                """
                <div class="card">
                    <strong>Scientific framing:</strong>
                    this output is an <em>AI-assisted acoustic variation</em>.
                    The learned VAE proposes a small structured change while
                    the original recording remains the acoustic reference.
                    It is not presented as a literal translation of bird
                    communication into human language.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ABOUT MODEL
# ============================================================

elif page == "About the Model":

    st.markdown(
        '<div class="section-title">'
        '🧠 About the Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'The architecture behind the bioacoustic laboratory.'
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)

    model_metrics = [
        ("3.69M", "Classifier Parameters"),
        ("15.09M", "VAE Parameters"),
        ("128-D", "Acoustic Embedding"),
        ("64-D", "Generative Latent Space"),
    ]

    for col, (
        value,
        label,
    ) in zip(
        cols,
        model_metrics,
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">'
        'Architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    architecture = pd.DataFrame(
        {
            "Stage": [
                "Input",
                "CNN",
                "Transformer",
                "Representation",
                "Classifier",
                "Generator",
                "Audio Reconstruction",
            ],
            "Technology": [
                "128 × 501 Mel Spectrogram",
                "Convolutional Feature Extraction",
                "3-layer Transformer Encoder",
                "128-D Birdsong Embedding",
                "10-class Softmax",
                "Species-conditioned CNN VAE",
                "Phase-preserving ISTFT",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="card">

        ### 🔬 Scientific framing

        This system learns statistical acoustic structure from
        bird recordings.

        The identification model learns species-discriminative
        acoustic representations, while the conditional VAE
        learns a generative latent space capable of producing
        new Mel-spectrogram structures conditioned on species.

        The term **translation** in this project is therefore
        best understood as **acoustic/structural translation**:
        mapping complex bird audio into machine-interpretable
        representations and generated acoustic structures.

        It does not claim to decode bird communication into
        human language.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        'Evaluation Snapshot'
        '</div>',
        unsafe_allow_html=True,
    )

    evaluation = pd.DataFrame(
        {
            "Metric": [
                "Test Accuracy",
                "Balanced Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted F1",
                "VAE Best Validation Loss",
                "Generated Samples",
            ],
            "Result": [
                "77.01%",
                "67.39%",
                "66.28%",
                "67.39%",
                "64.43%",
                "77.37%",
                "0.025722",
                "30",
            ],
        }
    )

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    AVIARA · AI Bioacoustic Birdsong Translator & Synthesizer
    <br>
    CNN + Transformer · Conditional VAE · Acoustic Intelligence
</div>
""",
    unsafe_allow_html=True,
)