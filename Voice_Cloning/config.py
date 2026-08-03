from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOICE_CLONING_DIRECTORY = PROJECT_ROOT / "Voice_Cloning"

VOICE_CLONING_DATA_DIRECTORY = (
    PROJECT_ROOT / "data" / "voice_cloning"
)

REFERENCE_AUDIO_DIRECTORY = (
    VOICE_CLONING_DATA_DIRECTORY / "references"
)

GENERATED_AUDIO_DIRECTORY = (
    VOICE_CLONING_DATA_DIRECTORY / "generated_audio"
)

DATASET_DIRECTORY = (
    VOICE_CLONING_DATA_DIRECTORY / "datasets"
)

TRAINED_MODELS_DIRECTORY = (
    VOICE_CLONING_DATA_DIRECTORY / "trained_models"
)

METADATA_DIRECTORY = (
    VOICE_CLONING_DATA_DIRECTORY / "metadata"
)


# ============================================================
# XTTS MODEL SETTINGS
# ============================================================

XTTS_MODEL_NAME = (
    "tts_models/multilingual/multi-dataset/xtts_v2"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DEFAULT_LANGUAGE = "en"


# ============================================================
# REFERENCE AUDIO SETTINGS
# ============================================================

REFERENCE_AUDIO_FILENAME = "reference.wav"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
}


# ============================================================
# GENERATED AUDIO SETTINGS
# ============================================================

GENERATED_AUDIO_EXTENSION = ".wav"

DEFAULT_OUTPUT_FILENAME = "generated_voice.wav"


# ============================================================
# XTTS GENERATION SETTINGS
# ============================================================

SPLIT_SENTENCES = True


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [
    VOICE_CLONING_DATA_DIRECTORY,
    REFERENCE_AUDIO_DIRECTORY,
    GENERATED_AUDIO_DIRECTORY,
    DATASET_DIRECTORY,
    TRAINED_MODELS_DIRECTORY,
    METADATA_DIRECTORY
]


def create_required_directories():
    """
    Create all required Voice Cloning directories
    if they do not already exist.
    """

    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )