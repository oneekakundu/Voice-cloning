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