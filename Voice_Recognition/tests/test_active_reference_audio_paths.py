"""
Test Suite: Active Reference Audio Paths Retrieval

Tests that VoiceProfileManager correctly retrieves the active Top-5 reference audio paths
for an existing enrolled user using get_active_reference_audio_paths(user_id).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Recognition.voice_profile_manager import VoiceProfileManager


def run_active_reference_audio_retrieval_test():
    manager = VoiceProfileManager()

    # Find an existing enrolled user with valid profiles
    profiles = manager.list_profiles()
    target_user_id = None
    paths = None

    for prof in profiles:
        user_id = prof.get("user_id")
        if not user_id:
            continue
        try:
            res_paths = manager.get_active_reference_audio_paths(user_id)
            if res_paths:
                target_user_id = user_id
                paths = res_paths
                break
        except Exception:
            continue

    if not target_user_id or not paths:
        raise RuntimeError(
            "No existing enrolled user with resolvable active reference audio paths was found."
        )

    # Print formatted test headers and output as required
    print("============================================================")
    print("STARTING ACTIVE REFERENCE AUDIO RETRIEVAL TEST")
    print("============================================================")
    print()
    print(f"User ID: {target_user_id}")
    print()
    print(f"Number of active reference audio files: {len(paths)}")
    print()

    for idx, path in enumerate(paths, start=1):
        print(f"Reference {idx}:")
        print(f"{path}")
        print()

    # Verifications
    assert isinstance(paths, list), "Returned object must be a list"
    assert len(paths) > 0, "The returned list must not be empty"
    assert len(paths) <= manager.MAX_REFERENCE_EMBEDDINGS, (
        f"List contains more than MAX_REFERENCE_EMBEDDINGS ({manager.MAX_REFERENCE_EMBEDDINGS}) paths"
    )

    seen = set()
    for path in paths:
        assert isinstance(path, Path), f"Every item must be a Path object, got {type(path)}"
        assert path.exists(), f"Path does not exist: {path}"
        assert path.is_file(), f"Path is not a file: {path}"
        assert path.suffix.lower() in [".wav", ".flac", ".mp3", ".ogg"], (
            f"Path is not an audio file: {path}"
        )
        resolved = path.resolve()
        assert resolved not in seen, f"Duplicate path detected: {path}"
        seen.add(resolved)

    # Verify returned paths correspond only to existing active records
    metadata = manager.load_metadata(target_user_id)
    active_records = manager.get_active_reference_records(metadata)
    assert len(paths) <= len(active_records), (
        "Returned paths count exceeds active records count"
    )

    print("============================================================")
    print("ACTIVE REFERENCE AUDIO RETRIEVAL TEST PASSED")
    print("============================================================")


if __name__ == "__main__":
    run_active_reference_audio_retrieval_test()
