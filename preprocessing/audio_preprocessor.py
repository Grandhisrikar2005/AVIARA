from pathlib import Path
import json
import librosa
import soundfile as sf
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "pilot"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
CONFIG_FILE = PROJECT_ROOT / "dataset" / "metadata" / "species_config.json"


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SAMPLE_RATE = 32000


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
# CREATE OUTPUT DIRECTORY
# ============================================================

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS ONE AUDIO FILE
# ============================================================

def process_audio(input_file: Path, output_file: Path):

    try:

        audio, _ = librosa.load(
            input_file,
            sr=TARGET_SAMPLE_RATE,
            mono=True
        )

        sf.write(
            output_file,
            audio,
            TARGET_SAMPLE_RATE,
            subtype="PCM_16"
        )

        return True

    except Exception as e:

        print(f"\nERROR processing: {input_file}")
        print(f"Reason: {e}")

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    total_files = 0
    successful = 0
    failed = 0

    print("=" * 65)
    print("AI BIO-ACOUSTIC BIRDSONG PREPROCESSING")
    print("=" * 65)

    print("\nInput directory:")
    print(RAW_DIR)

    print("\nOutput directory:")
    print(PROCESSED_DIR)

    print("\nSpecies being processed:")

    for scientific_name, common_name in VALID_SPECIES.items():

        print(
            f"  - {common_name} "
            f"({scientific_name})"
        )

    print("\n" + "=" * 65)


    # ========================================================
    # PROCESS EACH SPECIES
    # ========================================================

    for scientific_name, common_name in VALID_SPECIES.items():

        species_folder = scientific_name.replace(" ", "_")

        input_dir = RAW_DIR / species_folder

        output_dir = PROCESSED_DIR / species_folder

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        if not input_dir.exists():

            print("\nWARNING: Folder not found:")
            print(input_dir)

            continue


        # Search recursively through nested folders
        audio_files = [

            file

            for file in input_dir.rglob("*")

            if file.is_file()

            and file.suffix.lower()
            in {".wav", ".mp3", ".flac"}

        ]


        print(f"\n[{common_name}]")
        print(f"Found {len(audio_files)} audio files")


        for audio_file in tqdm(
            audio_files,
            desc="Processing"
        ):

            total_files += 1


            # Always save standardized WAV
            output_file = (
                output_dir /
                f"{audio_file.stem}.wav"
            )


            # Skip if already processed
            if output_file.exists():

                successful += 1

                continue


            if process_audio(
                audio_file,
                output_file
            ):

                successful += 1

            else:

                failed += 1


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE")
    print("=" * 65)

    print(f"Total files found : {total_files}")
    print(f"Successful        : {successful}")
    print(f"Failed            : {failed}")

    print("\nProcessed files saved to:")
    print(PROCESSED_DIR)

    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()