from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "dataset_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "vae_spectrograms"
)

SAMPLE_RATE = 32000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320
FMIN = 50
FMAX = 14000

MIN_DB = -80.0
MAX_DB = 0.0


def audio_to_normalized_mel(audio_path):
    """Convert audio into normalized 0-1 log-mel spectrogram."""

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # Fixed reference makes the representation consistent
    log_mel = librosa.power_to_db(
        mel,
        ref=1.0,
        top_db=80.0
    )

    # Convert -80...0 dB → 0...1
    normalized = (
        log_mel - MIN_DB
    ) / (
        MAX_DB - MIN_DB
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized.astype(np.float32)


def main():

    print("=" * 70)
    print("VAE NORMALIZED MEL-SPECTROGRAM GENERATION")
    print("=" * 70)

    if not MANIFEST.exists():
        print(f"\nERROR: Manifest not found:")
        print(MANIFEST)
        sys.exit(1)

    df = pd.read_csv(MANIFEST)

    print(f"\nTotal segments: {len(df)}")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    successful = 0
    failed = 0

    for split in ["train", "val", "test"]:

        split_df = df[
            df["split"] == split
        ].copy()

        split_output = (
            OUTPUT_DIR / split
        )

        split_output.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\nProcessing {split}: "
            f"{len(split_df)} segments"
        )

        for _, row in tqdm(
            split_df.iterrows(),
            total=len(split_df),
            desc=split
        ):

            segment_id = str(
                row["segment_id"]
            )

            # Segment IDs contain .wav
            output_name = (
                Path(segment_id).stem
                + ".npy"
            )

            output_path = (
                split_output
                / output_name
            )

            if output_path.exists():
                successful += 1
                continue

            # Actual segment location
            source_path = (
                PROJECT_ROOT
                / "dataset"
                / "segments"
                / str(row["scientific_name"])
                / segment_id
            )

            if not source_path.exists():

                # Fallback search
                matches = list(
                    (
                        PROJECT_ROOT
                        / "dataset"
                        / "segments"
                    ).rglob(segment_id)
                )

                if not matches:
                    failed += 1
                    continue

                source_path = matches[0]

            try:

                spectrogram = (
                    audio_to_normalized_mel(
                        source_path
                    )
                )

                np.save(
                    output_path,
                    spectrogram
                )

                successful += 1

            except Exception as e:

                failed += 1

                print(
                    f"\nFailed: {segment_id}"
                )
                print(e)

    print("\n" + "=" * 70)
    print("VAE SPECTROGRAM GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nSuccessful: {successful}")
    print(f"Failed    : {failed}")

    print(
        f"\nSaved to:\n{OUTPUT_DIR}"
    )

    # Verify one file
    sample_files = list(
        OUTPUT_DIR.rglob("*.npy")
    )

    if sample_files:

        sample = np.load(
            sample_files[0]
        )

        print(
            f"\nSample shape: {sample.shape}"
        )

        print(
            f"Sample min : {sample.min():.4f}"
        )

        print(
            f"Sample max : {sample.max():.4f}"
        )

        print(
            "\nExpected:"
        )
        print("Shape: approximately (128, 501)")
        print("Range: 0.0 - 1.0")


if __name__ == "__main__":
    main()