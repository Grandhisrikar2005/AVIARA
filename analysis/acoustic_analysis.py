"""
BIRDSONG ACOUSTIC STRUCTURAL ANALYSIS

Analyzes the actual 5-second test segments.

Features:
    - Duration
    - RMS energy
    - Zero-crossing rate
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff
    - Spectral flatness
    - Dominant frequency
"""

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "dataset_manifest.csv"
)

SEGMENTS_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "segments"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "acoustic_analysis"
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return audio, sr


# ============================================================
# EXTRACT ACOUSTIC FEATURES
# ============================================================

def extract_features(audio, sr):

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = len(audio) / sr

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    rms_mean = float(
        np.mean(rms)
    )

    rms_std = float(
        np.std(rms)
    )

    # --------------------------------------------------------
    # ZERO-CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    zcr_mean = float(
        np.mean(zcr)
    )

    # --------------------------------------------------------
    # SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    centroid_mean = float(
        np.mean(centroid)
    )

    # --------------------------------------------------------
    # SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    bandwidth_mean = float(
        np.mean(bandwidth)
    )

    # --------------------------------------------------------
    # SPECTRAL ROLLOFF
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85
    )

    rolloff_mean = float(
        np.mean(rolloff)
    )

    # --------------------------------------------------------
    # SPECTRAL FLATNESS
    # --------------------------------------------------------

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    flatness_mean = float(
        np.mean(flatness)
    )

    # --------------------------------------------------------
    # DOMINANT FREQUENCY
    # --------------------------------------------------------

    spectrum = np.abs(
        librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT
    )

    dominant_frequency = []

    for frame in spectrum.T:

        if np.max(frame) > 0:

            dominant_frequency.append(
                frequencies[
                    np.argmax(frame)
                ]
            )

    if dominant_frequency:

        dominant_frequency_mean = float(
            np.mean(dominant_frequency)
        )

    else:

        dominant_frequency_mean = 0.0

    return {
        "duration_seconds": duration,
        "rms_energy_mean": rms_mean,
        "rms_energy_std": rms_std,
        "zero_crossing_rate": zcr_mean,
        "spectral_centroid_hz": centroid_mean,
        "spectral_bandwidth_hz": bandwidth_mean,
        "spectral_rolloff_hz": rolloff_mean,
        "spectral_flatness": flatness_mean,
        "dominant_frequency_hz": dominant_frequency_mean
    }


# ============================================================
# FIND ACTUAL SEGMENT
# ============================================================

def find_segment_file(row):

    segment_id = str(
        row["segment_id"]
    )

    species = str(
        row["scientific_name"]
    )

    # --------------------------------------------------------
    # First: expected species directory
    # --------------------------------------------------------

    direct_path = (
        SEGMENTS_DIR
        / species
        / segment_id
    )

    if direct_path.exists():

        return direct_path

    # --------------------------------------------------------
    # Fallback: search all segment directories
    # --------------------------------------------------------

    matches = list(
        SEGMENTS_DIR.rglob(segment_id)
    )

    if matches:

        return matches[0]

    return None


# ============================================================
# ANALYZE TEST SEGMENTS
# ============================================================

def analyze_dataset():

    print("\nLoading dataset manifest...")

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    test_df = df[
        df["split"] == "test"
    ].copy()

    test_df = test_df.reset_index(
        drop=True
    )

    total = len(test_df)

    print(
        f"Test segments: {total}"
    )

    results = []

    failed = 0

    for position, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):

        segment_id = str(
            row["segment_id"]
        )

        segment_path = find_segment_file(
            row
        )

        if segment_path is None:

            failed += 1

            print(
                f"\nSegment not found: "
                f"{segment_id}"
            )

            continue

        try:

            audio, sr = load_audio(
                segment_path
            )

            features = extract_features(
                audio,
                sr
            )

            result = {
                "segment_id": segment_id,
                "scientific_name": row[
                    "scientific_name"
                ],
                "audio_file": str(
                    segment_path
                )
            }

            result.update(
                features
            )

            results.append(
                result
            )

        except Exception as error:

            failed += 1

            print(
                f"\nFailed: {segment_id}"
            )

            print(
                f"Reason: {error}"
            )

        if (
            position % 25 == 0
            or position == total
        ):

            print(
                f"Processed: "
                f"{position}/{total}"
            )

    results_df = pd.DataFrame(
        results
    )

    return results_df, failed


# ============================================================
# SPECIES-LEVEL SUMMARY
# ============================================================

def create_species_summary(results_df):

    feature_columns = [
        "duration_seconds",
        "rms_energy_mean",
        "rms_energy_std",
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_hz",
        "spectral_flatness",
        "dominant_frequency_hz"
    ]

    summary = (
        results_df
        .groupby("scientific_name")[
            feature_columns
        ]
        .mean()
        .reset_index()
    )

    return summary


# ============================================================
# CREATE FEATURE PLOT
# ============================================================

def create_feature_plot(
    summary,
    feature,
    title,
    ylabel,
    filename
):

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        summary["scientific_name"],
        summary[feature]
    )

    plt.title(
        title,
        fontsize=14
    )

    plt.xlabel(
        "Species"
    )

    plt.ylabel(
        ylabel
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved plot: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BIRDSONG ACOUSTIC STRUCTURAL ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # ANALYZE ACTUAL TEST SEGMENTS
    # --------------------------------------------------------

    results_df, failed = analyze_dataset()

    if len(results_df) == 0:

        raise RuntimeError(
            "No acoustic features were extracted."
        )

    # --------------------------------------------------------
    # SAVE SEGMENT-LEVEL FEATURES
    # --------------------------------------------------------

    feature_file = (
        OUTPUT_DIR
        / "test_acoustic_features.csv"
    )

    results_df.to_csv(
        feature_file,
        index=False
    )

    print(
        f"\nSaved acoustic features:"
    )

    print(
        feature_file
    )

    # --------------------------------------------------------
    # SPECIES SUMMARY
    # --------------------------------------------------------

    summary = create_species_summary(
        results_df
    )

    summary_file = (
        OUTPUT_DIR
        / "species_acoustic_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"\nSaved species summary:"
    )

    print(
        summary_file
    )

    # --------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------

    create_feature_plot(
        summary,
        "spectral_centroid_hz",
        "Average Spectral Centroid by Bird Species",
        "Spectral Centroid (Hz)",
        "spectral_centroid_by_species.png"
    )

    create_feature_plot(
        summary,
        "dominant_frequency_hz",
        "Average Dominant Frequency by Bird Species",
        "Dominant Frequency (Hz)",
        "dominant_frequency_by_species.png"
    )

    create_feature_plot(
        summary,
        "rms_energy_mean",
        "Average Acoustic Energy by Bird Species",
        "Mean RMS Energy",
        "rms_energy_by_species.png"
    )

    create_feature_plot(
        summary,
        "zero_crossing_rate",
        "Average Zero-Crossing Rate by Bird Species",
        "Zero-Crossing Rate",
        "zero_crossing_rate_by_species.png"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACOUSTIC ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessful test segments: "
        f"{len(results_df)}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Species analyzed: "
        f"{results_df['scientific_name'].nunique()}"
    )

    print(
        "\nAll features are calculated from "
        "the actual 5-second test segments."
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print("\nGenerated:")

    print(
        "  test_acoustic_features.csv"
    )

    print(
        "  species_acoustic_summary.csv"
    )

    print(
        "  spectral_centroid_by_species.png"
    )

    print(
        "  dominant_frequency_by_species.png"
    )

    print(
        "  rms_energy_by_species.png"
    )

    print(
        "  zero_crossing_rate_by_species.png"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()