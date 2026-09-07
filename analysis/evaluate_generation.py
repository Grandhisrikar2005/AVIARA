from pathlib import Path
import sys

import numpy as np
import pandas as pd
import librosa


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DIRECTORIES
# ============================================================

GENERATED_AUDIO_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_birdsong"
    / "audio"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generation_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 32000

N_FFT = 1024

HOP_LENGTH = 320

FMIN = 50

FMAX = 14000


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    duration = (
        len(audio) / sr
    )

    rms = librosa.feature.rms(
        y=audio
    )[0]

    zero_crossing = (
        librosa.feature.zero_crossing_rate(
            y=audio
        )[0]
    )

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr,
            roll_percent=0.85,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    spectral_flatness = (
        librosa.feature.spectral_flatness(
            y=audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    magnitude = np.abs(
        stft
    )

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT,
    )

    valid_frequency_mask = (
        (frequencies >= FMIN)
        & (frequencies <= FMAX)
    )

    valid_frequencies = (
        frequencies[
            valid_frequency_mask
        ]
    )

    valid_magnitude = (
        magnitude[
            valid_frequency_mask,
            :
        ]
    )

    dominant_indices = (
        np.argmax(
            valid_magnitude,
            axis=0,
        )
    )

    dominant_frequencies = (
        valid_frequencies[
            dominant_indices
        ]
    )

    return {
        "duration_seconds": float(
            duration
        ),

        "rms_energy_mean": float(
            np.mean(rms)
        ),

        "rms_energy_std": float(
            np.std(rms)
        ),

        "zero_crossing_rate": float(
            np.mean(zero_crossing)
        ),

        "spectral_centroid_hz": float(
            np.mean(
                spectral_centroid
            )
        ),

        "spectral_bandwidth_hz": float(
            np.mean(
                spectral_bandwidth
            )
        ),

        "spectral_rolloff_hz": float(
            np.mean(
                spectral_rolloff
            )
        ),

        "spectral_flatness": float(
            np.mean(
                spectral_flatness
            )
        ),

        "dominant_frequency_hz": float(
            np.mean(
                dominant_frequencies
            )
        ),
    }


# ============================================================
# EXTRACT SPECIES FROM FILENAME
# ============================================================

def get_species_from_filename(filename):

    name = filename.lower()

    species_map = {
        "asian_koel": "Asian Koel",
        "black_kite": "Black Kite",
        "common_myna": "Common Myna",
        "common_tailorbird": "Common Tailorbird",
        "greater_coucal": "Greater Coucal",
        "house_crow": "House Crow",
        "indian_peafowl": "Indian Peafowl",
        "red_vented_bulbul": "Red-vented Bulbul",
        "rose_ringed_parakeet": "Rose-ringed Parakeet",
        "white_throated_kingfisher": (
            "White-throated Kingfisher"
        ),
    }

    for key, species in species_map.items():

        if name.startswith(key):

            return species

    return "Unknown"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATED BIRDSONG EVALUATION")
    print("=" * 70)

    if not GENERATED_AUDIO_DIR.exists():

        raise FileNotFoundError(
            f"\nGenerated audio directory "
            f"not found:\n"
            f"{GENERATED_AUDIO_DIR}"
        )

    audio_files = sorted(
        GENERATED_AUDIO_DIR.glob(
            "*.wav"
        )
    )

    print(
        f"\nGenerated audio files: "
        f"{len(audio_files)}"
    )

    if len(audio_files) == 0:

        raise RuntimeError(
            "No generated WAV files found."
        )

    results = []

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for index, audio_path in enumerate(
        audio_files,
        start=1,
    ):

        print(
            f"[{index}/{len(audio_files)}] "
            f"{audio_path.name}"
        )

        try:

            features = extract_features(
                audio_path
            )

            species = (
                get_species_from_filename(
                    audio_path.name
                )
            )

            row = {
                "filename": audio_path.name,
                "species": species,
            }

            row.update(
                features
            )

            results.append(
                row
            )

            print(
                f"  Species: {species}"
            )

            print(
                f"  Duration: "
                f"{features['duration_seconds']:.2f}s"
            )

            print(
                f"  Centroid: "
                f"{features['spectral_centroid_hz']:.2f} Hz"
            )

            print(
                f"  Dominant: "
                f"{features['dominant_frequency_hz']:.2f} Hz"
            )

            print(
                "  ✓ Processed"
            )

        except Exception as error:

            print(
                f"  ✗ Failed: {error}"
            )

    # --------------------------------------------------------
    # Save sample-level results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        OUTPUT_DIR
        / "generated_audio_features.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Species summary
    # --------------------------------------------------------

    if len(results_df) > 0:

        numeric_columns = [
            "duration_seconds",
            "rms_energy_mean",
            "rms_energy_std",
            "zero_crossing_rate",
            "spectral_centroid_hz",
            "spectral_bandwidth_hz",
            "spectral_rolloff_hz",
            "spectral_flatness",
            "dominant_frequency_hz",
        ]

        species_summary = (
            results_df
            .groupby("species")[
                numeric_columns
            ]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
            .reset_index()
        )

        summary_path = (
            OUTPUT_DIR
            / "species_generation_summary.csv"
        )

        species_summary.to_csv(
            summary_path,
            index=False,
        )

    else:

        summary_path = (
            OUTPUT_DIR
            / "species_generation_summary.csv"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GENERATION EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessfully processed: "
        f"{len(results)}/"
        f"{len(audio_files)}"
    )

    if len(results_df) > 0:

        print(
            "\nOverall generated-audio statistics:"
        )

        print(
            f"Mean duration: "
            f"{results_df['duration_seconds'].mean():.3f}s"
        )

        print(
            f"Mean RMS energy: "
            f"{results_df['rms_energy_mean'].mean():.6f}"
        )

        print(
            f"Mean spectral centroid: "
            f"{results_df['spectral_centroid_hz'].mean():.2f} Hz"
        )

        print(
            f"Mean dominant frequency: "
            f"{results_df['dominant_frequency_hz'].mean():.2f} Hz"
        )

    print(
        "\nSample-level results:"
    )

    print(
        f"  {results_path}"
    )

    print(
        "\nSpecies-level summary:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "\n✓ Generation evaluation completed."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()