from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_DIR = BASE_DIR / "dataset" / "metadata" / "pilot"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
SEGMENTS_DIR = BASE_DIR / "dataset" / "segments"

OUTPUT_FILE = (
    BASE_DIR
    / "dataset"
    / "metadata"
    / "communication_manifest.csv"
)


# ============================================================
# FINAL PROJECT SPECIES
# ============================================================

SPECIES_MAP = {
    "Asian Koel": "Eudynamys scolopaceus",
    "Common Myna": "Acridotheres tristis",
    "Red-vented Bulbul": "Pycnonotus cafer",
    "Common Tailorbird": "Orthotomus sutorius",
    "Rose-ringed Parakeet": "Psittacula krameri",
    "Indian Peafowl": "Pavo cristatus",
    "Greater Coucal": "Centropus sinensis",
    "White-throated Kingfisher": "Halcyon smyrnensis",
    "Black Kite": "Milvus migrans",
    "House Crow": "Corvus splendens",
}


# ============================================================
# COMMUNICATION LABEL MAPPING
# ============================================================

def map_communication_type(raw_type):
    """
    Map Xeno-canto vocalization annotations to
    four usable communication categories.

    Returns None for uncertain / missing annotations,
    because those should NOT become training labels.
    """

    if pd.isna(raw_type):
        return None

    value = str(raw_type).strip().lower()

    # --------------------------------------------------------
    # UNCERTAIN -> EXCLUDE
    # --------------------------------------------------------
    if "uncertain" in value:
        return None

    # --------------------------------------------------------
    # SONG
    # --------------------------------------------------------
    if (
        "song" in value
        or "displaying" in value
        or "duet" in value
        or "subsong" in value
    ):
        return "song"

    # --------------------------------------------------------
    # ALARM / WARNING
    # --------------------------------------------------------
    if "alarm" in value:
        return "alarm_warning"

    # --------------------------------------------------------
    # BEGGING / DISTRESS
    # --------------------------------------------------------
    if (
        "begging" in value
        or "distress" in value
    ):
        return "begging_distress"

    # --------------------------------------------------------
    # CONTACT / SOCIAL
    # --------------------------------------------------------
    if (
        "call" in value
        or "flight" in value
        or "aberrant" in value
    ):
        return "contact_social"

    # --------------------------------------------------------
    # UNKNOWN -> EXCLUDE
    # --------------------------------------------------------
    return None


# ============================================================
# RECORDING ID EXTRACTION
# ============================================================

def extract_recording_id(filename):
    if not filename:
        return None

    match = re.search(
        r"XC(\d+)",
        str(filename)
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# LOCAL RECORDINGS
# ============================================================

def collect_local_recordings():

    recordings = {}

    if not PROCESSED_DIR.exists():
        return recordings

    audio_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }

    for file_path in PROCESSED_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in audio_extensions:
            continue

        recording_id = extract_recording_id(
            file_path.name
        )

        if recording_id:
            recordings[recording_id] = file_path

    return recordings


# ============================================================
# SEGMENTS
# ============================================================

def collect_segments():

    segments_by_recording = {}

    if not SEGMENTS_DIR.exists():
        return segments_by_recording

    for file_path in SEGMENTS_DIR.rglob("*.wav"):

        recording_id = extract_recording_id(
            file_path.name
        )

        if not recording_id:
            continue

        segments_by_recording.setdefault(
            recording_id,
            []
        ).append(file_path)

    return segments_by_recording


# ============================================================
# METADATA
# ============================================================

def load_metadata():

    csv_files = list(
        METADATA_DIR.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No metadata CSV files found in {METADATA_DIR}"
        )

    frames = []

    for csv_file in csv_files:

        try:
            df = pd.read_csv(csv_file)

            if "id" not in df.columns:
                continue

            frames.append(df)

        except Exception as exc:
            print(
                f"Could not read {csv_file}: {exc}"
            )

    if not frames:
        raise RuntimeError(
            "No valid metadata CSV files were loaded."
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df = df.drop_duplicates(
        subset=["id"],
        keep="first"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING CLEAN COMMUNICATION MANIFEST")
    print("=" * 70)

    metadata = load_metadata()

    print(
        f"Metadata records found : {len(metadata)}"
    )

    local_recordings = collect_local_recordings()

    print(
        f"Local recordings found  : {len(local_recordings)}"
    )

    segments_by_recording = collect_segments()

    print(
        f"Recordings with segments: "
        f"{len(segments_by_recording)}"
    )

    rows = []

    matched_recordings = 0
    skipped_uncertain = 0
    skipped_species = 0
    skipped_unmatched = 0

    # ========================================================
    # MATCH METADATA TO LOCAL AUDIO
    # ========================================================

    for _, row in metadata.iterrows():

        recording_id = str(
            row["id"]
        ).strip()

        # Must exist locally
        if recording_id not in local_recordings:
            skipped_unmatched += 1
            continue

        # Must have generated segments
        if recording_id not in segments_by_recording:
            skipped_unmatched += 1
            continue

        species = str(
            row.get("en", "")
        ).strip()

        # Only final 10 species
        if species not in SPECIES_MAP:
            skipped_species += 1
            continue

        raw_type = row.get(
            "type",
            None
        )

        communication_type = map_communication_type(
            raw_type
        )

        # Exclude uncertain/missing/unknown
        if communication_type is None:
            skipped_uncertain += 1
            continue

        scientific_name = SPECIES_MAP[
            species
        ]

        local_file = local_recordings[
            recording_id
        ]

        segment_files = sorted(
            segments_by_recording[
                recording_id
            ]
        )

        # ====================================================
        # ONE ROW PER SEGMENT
        # ====================================================

        for segment_file in segment_files:

            rows.append(
                {
                    "recording_id": recording_id,
                    "species": species,
                    "scientific_name": scientific_name,
                    "raw_type": raw_type,
                    "communication_type": communication_type,

                    "recording_file": str(
                        local_file.relative_to(BASE_DIR)
                    ),

                    "segment_file": str(
                        segment_file.relative_to(BASE_DIR)
                    ),

                    "segment_filename": segment_file.name,

                    "recordist": row.get(
                        "rec",
                        ""
                    ),

                    "country": row.get(
                        "cnt",
                        ""
                    ),

                    "location": row.get(
                        "loc",
                        ""
                    ),

                    "date": row.get(
                        "date",
                        ""
                    ),

                    "quality": row.get(
                        "q",
                        ""
                    ),

                    "license": row.get(
                        "lic",
                        ""
                    ),

                    "source_url": row.get(
                        "url",
                        ""
                    ),
                }
            )

        matched_recordings += 1

    # ========================================================
    # DATAFRAME
    # ========================================================

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError(
            "No communication samples were created."
        )

    manifest = manifest.sort_values(
        [
            "scientific_name",
            "recording_id",
            "segment_filename",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLEAN COMMUNICATION MANIFEST CREATED")
    print("=" * 70)

    print(
        f"Matched recordings        : {matched_recordings}"
    )

    print(
        f"Skipped uncertain/missing : {skipped_uncertain}"
    )

    print(
        f"Skipped other species     : {skipped_species}"
    )

    print(
        f"Unmatched recordings      : {skipped_unmatched}"
    )

    print(
        f"Total usable segments     : {len(manifest)}"
    )

    print()
    print("COMMUNICATION DISTRIBUTION")
    print("-" * 70)

    print(
        manifest[
            "communication_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("COMMUNICATION BY SPECIES")
    print("-" * 70)

    print(
        pd.crosstab(
            manifest["species"],
            manifest["communication_type"]
        ).to_string()
    )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()