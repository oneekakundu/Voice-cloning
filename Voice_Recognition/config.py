from pathlib import Path
import torch


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory where speaker profiles will eventually be stored
VOICE_PROFILES_DIR = PROJECT_ROOT / "data" / "voice_profiles"

# ECAPA-TDNN model
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# Device selection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# =============================================================
# SYSTEM CONFIGURATION
# =============================================================


# Paths

DATA_DIR = PROJECT_ROOT / "data"

VOICE_PROFILES_DIR = DATA_DIR / "voice_profiles"

RECORDINGS_DIR = DATA_DIR / "recordings"


# Audio Settings

RECORDING_DURATION = 3.0

SAMPLE_RATE = 16000


# Voice Enrollment Configuration

ENROLLMENT_THRESHOLD = 0.70

REQUIRED_MATCHES = 3  # References needed for enrollment completion


# =============================================================
# SPEAKER ENCODER CONFIGURATION (ECAPA-TDNN)
# =============================================================

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"