from pathlib import Path
import json
import math
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SEGMENTS_DIR = PROJECT_ROOT / "dataset" / "segments"
METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata"

CONFIG_FILE = METADATA_DIR / "species_config.json"

MANIFEST_FILE = METADATA_DIR / "segments_manifest.csv"


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 32000

SEGMENT_DURATION = 5          # seconds
OVERLAP = 0.5                 # 50% overlap

SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
HOP_SAMPLES = int(SEGMENT_SAMPLES * (1 - OVERLAP))


# ============================================================
# LOAD SPECIES CONFIGURATION
# ============================================================

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

VALID_SPECIES = {
    item["scientific_name"]: item["common_name"]
    for item in config["species"]
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

SEGMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS ONE RECORDING
# ============================================================

def segment_recording(audio_file, species_scientific, species_common):

    try:

        audio, sr = librosa.load(
            audio_file,
            sr=SAMPLE_RATE,
            mono=True
        )

        total_samples = len(audio)

        recording_duration = total_samples / SAMPLE_RATE

        # Skip recordings shorter than the required segment length
        if total_samples < SEGMENT_SAMPLES:
            return []

        species_folder = (
            species_scientific.replace(" ", "_")
        )

        output_dir = SEGMENTS_DIR / species_folder

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        recording_id = audio_file.stem

        records = []

        segment_number = 0

        start = 0

        while start + SEGMENT_SAMPLES <= total_samples:

            end = start + SEGMENT_SAMPLES

            segment = audio[start:end]

            start_time = start / SAMPLE_RATE
            end_time = end / SAMPLE_RATE

            output_name = (
                f"{recording_id}"
                f"_segment_{segment_number:04d}.wav"
            )

            output_file = output_dir / output_name

            sf.write(
                output_file,
                segment,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            records.append({

                "segment_id": output_name,

                "recording_id": recording_id,

                "species": species_common,

                "scientific_name": species_scientific,

                "source_file": audio_file.name,

                "segment_number": segment_number,

                "start_time": round(start_time, 3),

                "end_time": round(end_time, 3),

                "segment_duration": SEGMENT_DURATION,

                "recording_duration": round(
                    recording_duration,
                    3
                ),

                "sample_rate": SAMPLE_RATE,

                "file_path": str(
                    output_file.relative_to(PROJECT_ROOT)
                )

            })

            segment_number += 1

            start += HOP_SAMPLES

        return records

    except Exception as e:

        print(
            f"\nERROR processing "
            f"{audio_file.name}: {e}"
        )

        return []


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI BIO-ACOUSTIC BIRDSONG SEGMENTATION")
    print("=" * 70)

    print(f"\nInput:")
    print(PROCESSED_DIR)

    print(f"\nOutput:")
    print(SEGMENTS_DIR)

    print(f"\nSegment duration : {SEGMENT_DURATION} seconds")
    print(f"Overlap          : {OVERLAP * 100:.0f}%")
    print(f"Sample rate      : {SAMPLE_RATE} Hz")

    print("\n" + "=" * 70)


    all_records = []

    total_recordings = 0

    successful_recordings = 0


    # ========================================================
    # PROCESS EACH SPECIES
    # ========================================================

    for scientific_name, common_name in VALID_SPECIES.items():

        species_folder = (
            scientific_name.replace(" ", "_")
        )

        input_dir = (
            PROCESSED_DIR /
            species_folder
        )

        if not input_dir.exists():

            print(
                f"\nWARNING: Missing folder:"
                f"\n{input_dir}"
            )

            continue


        audio_files = list(
            input_dir.rglob("*.wav")
        )


        print(
            f"\n[{common_name}]"
        )

        print(
            f"Found {len(audio_files)} recordings"
        )


        for audio_file in tqdm(
            audio_files,
            desc="Segmenting"
        ):

            total_recordings += 1


            records = segment_recording(
                audio_file,
                scientific_name,
                common_name
            )


            if records:

                all_records.extend(records)

                successful_recordings += 1


    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    if all_records:

        manifest = pd.DataFrame(
            all_records
        )

        manifest.to_csv(
            MANIFEST_FILE,
            index=False
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("SEGMENTATION COMPLETE")
    print("=" * 70)

    print(
        f"Original recordings : "
        f"{total_recordings}"
    )

    print(
        f"Recordings processed : "
        f"{successful_recordings}"
    )

    print(
        f"Total segments      : "
        f"{len(all_records)}"
    )

    print(
        f"\nManifest saved to:"
        f"\n{MANIFEST_FILE}"
    )

    print(
        f"\nSegments saved to:"
        f"\n{SEGMENTS_DIR}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()