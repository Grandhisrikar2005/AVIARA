from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "dataset_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spectrograms"
)


# ============================================================
# AUDIO / SPECTROGRAM SETTINGS
# ============================================================

SAMPLE_RATE = 32000

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320

FMIN = 50
FMAX = 14000

# Fixed dynamic range for generative modeling.
TOP_DB = 80.0


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_FILE}"
        )

    return pd.read_csv(MANIFEST_FILE)


# ============================================================
# GENERATE ONE NORMALIZED MEL SPECTROGRAM
# ============================================================

def create_spectrogram(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # Fixed reference.
    log_mel = librosa.power_to_db(
        mel,
        ref=1.0,
        top_db=TOP_DB
    )

    # Convert:
    #
    # -80 dB → 0.0
    #   0 dB → 1.0
    #
    normalized = (
        log_mel + TOP_DB
    ) / TOP_DB

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AI BIO-ACOUSTIC NORMALIZED MEL SPECTROGRAM GENERATION"
    )
    print("=" * 70)

    print("\nManifest:")
    print(MANIFEST_FILE)

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nConfiguration:")
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Mel bins    : {N_MELS}")
    print(f"FFT size    : {N_FFT}")
    print(f"Hop length  : {HOP_LENGTH}")
    print(f"Frequency   : {FMIN} - {FMAX} Hz")
    print(f"Dynamic range: {TOP_DB} dB")

    print("\nRepresentation:")
    print("0.0 = -80 dB")
    print("1.0 =   0 dB")

    print("\n" + "=" * 70)

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = load_manifest()

    print(
        f"\nTotal segments: {len(df)}"
    )

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating normalized spectrograms"
    ):

        split = row["split"]

        segment_id = row["segment_id"]

        audio_path = (
            PROJECT_ROOT
            / row["file_path"]
        )

        split_dir = (
            OUTPUT_DIR
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            split_dir
            / f"{Path(segment_id).stem}.npy"
        )

        # IMPORTANT:
        # We deliberately overwrite old spectrograms.
        # The old files use a different representation.
        if output_file.exists():
            output_file.unlink()

        if not audio_path.exists():

            print(
                f"\nWARNING: Audio not found:"
                f"\n{audio_path}"
            )

            failed += 1
            continue

        try:

            spectrogram = create_spectrogram(
                audio_path
            )

            if spectrogram.shape != (
                128,
                501
            ):

                print(
                    f"\nWARNING: Unexpected shape "
                    f"for {segment_id}: "
                    f"{spectrogram.shape}"
                )

                failed += 1
                continue

            np.save(
                output_file,
                spectrogram
            )

            successful += 1

        except Exception as e:

            print(
                f"\nERROR: {segment_id}"
            )

            print(
                f"Reason: {e}"
            )

            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "SPECTROGRAM GENERATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Total segments : {len(df)}"
    )

    print(
        f"Successful     : {successful}"
    )

    print(
        f"Failed         : {failed}"
    )

    print(
        "\nSpectrograms saved to:"
    )

    print(OUTPUT_DIR)

    print("=" * 70)


if __name__ == "__main__":
    main()