"""
Test Suite: Voice Cloning Pipeline Test

Tests the standalone Voice_Cloning pipeline end-to-end using the top-1 reference audio
selected from the existing Top-5 active reference paths.
"""

import sys
import wave
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Cloning.pipeline import VoiceCloningPipeline
from Voice_Recognition.voice_profile_manager import VoiceProfileManager


def get_audio_duration(file_path: Path) -> float:
    with wave.open(str(file_path), "rb") as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        return nframes / float(framerate)


def test_voice_cloning_pipeline():
    profile_mgr = VoiceProfileManager()
    profiles = profile_mgr.list_profiles()
    target_user_id = None
    active_paths = None

    for prof in profiles:
        uid = prof.get("user_id")
        if not uid:
            continue
        try:
            paths = profile_mgr.get_active_reference_audio_paths(uid)
            if paths:
                target_user_id = uid
                active_paths = paths
                break
        except Exception:
            continue

    if not target_user_id or not active_paths:
        raise RuntimeError("No enrolled user with resolvable active reference audio files was found.")

    output_text = (
        "Hello. This is a test of the CARE Doll personalized voice cloning system."
    )

    print("============================================================")
    print("STARTING VOICE CLONING PIPELINE TEST")
    print("============================================================")
    print()
    print("User ID:")
    print(f"{target_user_id}")
    print()
    print("Output text:")
    print(f"{output_text}")
    print()
    print("Retrieving the existing active Top-5 reference paths...")
    print()
    print("Number of active paths returned:")
    print(f"{len(active_paths)}")
    print()
    print("Selecting only the first path according to the existing ranked order...")
    print()
    selected_reference = active_paths[0]
    print("Selected XTTS reference:")
    print(f"{selected_reference}")
    print()
    print("Starting XTTS-v2 generation...")

    pipeline = VoiceCloningPipeline()
    generated_file = pipeline.generate_for_identified_user(
        user_id=target_user_id,
        text=output_text,
        language="en"
    )

    # Verifications
    assert generated_file.exists(), f"Generated audio file does not exist: {generated_file}"
    assert generated_file.is_file(), f"Generated output path is not a file: {generated_file}"

    file_size = generated_file.stat().st_size
    assert file_size > 0, "Generated audio file must be non-empty (> 0 bytes)."

    duration = get_audio_duration(generated_file)
    assert duration > 0.0, "Generated audio duration must be greater than zero."

    print()
    print("Generated audio:")
    print(f"{generated_file}")
    print()
    print("Generated file size:")
    print(f"{file_size} bytes")
    print()
    print("Generated duration:")
    print(f"{duration:.2f} seconds")
    print()
    print("============================================================")
    print("VOICE CLONING PIPELINE TEST PASSED")
    print("============================================================")
    print()
    print("Reference policy:")
    print("Existing Top-5 ranked paths reused; only the first path is used for XTTS.")
    print()
    print("Manual Listening Checklist:")
    print("1. Is the generated sentence understandable?")
    print("2. Is the generated voice recognizable as the selected user?")
    print("3. Is there excessive leading silence?")
    print("4. Is there excessive trailing silence?")
    print("5. Are words or phrases repeated?")
    print("6. Is the speech unnaturally slow?")


if __name__ == "__main__":
    test_voice_cloning_pipeline()
