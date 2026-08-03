"""
Test Suite: Duration-Aware Voice Profile Reference Selection Policy

Tests all 8 required scenarios for the new reference selection policy:
1. Fewer than 5 valid recordings
2. Exactly 5 valid recordings
3. More than 5 valid recordings
4. New longer recording (replaces shortest active reference)
5. New shorter recording (retained in history, active set unchanged)
6. Equal-duration recordings (deterministic tie-breaking by earlier enrollment)
7. Existing identification behavior (cosine similarity, 0.70 threshold, majority vote)
8. Legacy profile loading (handles missing duration metadata, safe migration)
"""

import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from Voice_Recognition.voice_profile_manager import VoiceProfileManager
from Voice_Recognition.speaker_identifier import SpeakerIdentifier

TEST_PROFILES_DIR = "data/test_duration_policy_profiles"
TEST_USER_ID = "duration_user_001"


def clean_test_dir():
    p = Path(TEST_PROFILES_DIR)
    if p.exists():
        shutil.rmtree(p)


def create_mock_embedding(seed: int = 0) -> torch.Tensor:
    """Create deterministic 192-dim embedding."""
    torch.manual_seed(seed)
    embedding = torch.randn(192)
    return torch.nn.functional.normalize(embedding, p=2, dim=0)


def test_1_fewer_than_five_recordings():
    print("\n========================================")
    print("TEST 1: Fewer than five valid recordings")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    durations = [5.0, 8.5, 12.0]
    for idx, dur in enumerate(durations, start=1):
        emb = create_mock_embedding(idx)
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb,
            usable_speech_duration=dur
        )

    assert manager.get_total_enrollment_count(TEST_USER_ID) == 3
    assert manager.get_reference_count(TEST_USER_ID) == 3

    metadata = manager.load_metadata(TEST_USER_ID)
    active_records = manager.get_active_reference_records(metadata)
    assert len(active_records) == 3
    assert [r["usable_speech_duration"] for r in active_records] == [12.0, 8.5, 5.0]

    print("[OK] All 3 recordings retained and active.")


def test_2_exactly_five_recordings():
    print("\n========================================")
    print("TEST 2: Exactly five valid recordings")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    durations = [4.0, 7.0, 10.0, 15.0, 6.0]
    for idx, dur in enumerate(durations, start=1):
        emb = create_mock_embedding(idx)
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb,
            usable_speech_duration=dur
        )

    assert manager.get_total_enrollment_count(TEST_USER_ID) == 5
    assert manager.get_reference_count(TEST_USER_ID) == 5

    metadata = manager.load_metadata(TEST_USER_ID)
    active_records = manager.get_active_reference_records(metadata)
    assert len(active_records) == 5
    assert [r["usable_speech_duration"] for r in active_records] == [15.0, 10.0, 7.0, 6.0, 4.0]

    print("[OK] All 5 recordings retained and active.")


def test_3_more_than_five_recordings():
    print("\n========================================")
    print("TEST 3: More than five valid recordings")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    durations = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    for idx, dur in enumerate(durations, start=1):
        emb = create_mock_embedding(idx)
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb,
            usable_speech_duration=dur
        )

    assert manager.get_total_enrollment_count(TEST_USER_ID) == 8
    assert manager.get_reference_count(TEST_USER_ID) == 5

    metadata = manager.load_metadata(TEST_USER_ID)
    active_records = manager.get_active_reference_records(metadata)
    active_durations = [r["usable_speech_duration"] for r in active_records]
    assert active_durations == [18.0, 16.0, 14.0, 12.0, 10.0]

    print(f"[OK] All 8 stored. Active top 5 durations: {active_durations}")


def test_4_new_longer_recording():
    print("\n========================================")
    print("TEST 4: New longer recording enters active set")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    initial_durations = [10.0, 12.0, 14.0, 16.0, 18.0]
    for idx, dur in enumerate(initial_durations, start=1):
        emb = create_mock_embedding(idx)
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb,
            usable_speech_duration=dur
        )

    # Add 6th recording longer than shortest active reference (10.0s)
    new_emb = create_mock_embedding(6)
    manager.add_reference_embedding(
        user_id=TEST_USER_ID,
        embedding=new_emb,
        usable_speech_duration=22.0
    )

    assert manager.get_total_enrollment_count(TEST_USER_ID) == 6
    assert manager.get_reference_count(TEST_USER_ID) == 5

    metadata = manager.load_metadata(TEST_USER_ID)
    active_records = manager.get_active_reference_records(metadata)
    active_durations = [r["usable_speech_duration"] for r in active_records]

    assert 22.0 in active_durations
    assert 10.0 not in active_durations
    assert active_durations == [22.0, 18.0, 16.0, 14.0, 12.0]

    # Verify 10.0s record is still in total enrollment history
    all_durations = [r["usable_speech_duration"] for r in metadata["enrollment_records"]]
    assert 10.0 in all_durations

    print(f"[OK] 22.0s recording entered active set. Active durations: {active_durations}")


def test_5_new_shorter_recording():
    print("\n========================================")
    print("TEST 5: New shorter recording stays in history without affecting active set")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    initial_durations = [10.0, 12.0, 14.0, 16.0, 18.0]
    for idx, dur in enumerate(initial_durations, start=1):
        emb = create_mock_embedding(idx)
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb,
            usable_speech_duration=dur
        )

    # Add 6th recording shorter than all active references
    new_emb = create_mock_embedding(6)
    manager.add_reference_embedding(
        user_id=TEST_USER_ID,
        embedding=new_emb,
        usable_speech_duration=5.0
    )

    assert manager.get_total_enrollment_count(TEST_USER_ID) == 6
    assert manager.get_reference_count(TEST_USER_ID) == 5

    metadata = manager.load_metadata(TEST_USER_ID)
    active_records = manager.get_active_reference_records(metadata)
    active_durations = [r["usable_speech_duration"] for r in active_records]

    assert active_durations == [18.0, 16.0, 14.0, 12.0, 10.0]
    assert 5.0 not in active_durations

    print(f"[OK] 5.0s recording retained in history; active set unchanged: {active_durations}")


def test_6_equal_duration_recordings():
    print("\n========================================")
    print("TEST 6: Equal-duration recordings tie-breaker")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    # Enroll 6 recordings: 5 with duration 10.0s, then 6th with duration 10.0s
    for idx in range(1, 7):
        emb = create_mock_embedding(idx)
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb,
            usable_speech_duration=10.0,
            recording_id=f"rec_{idx}"
        )

    metadata = manager.load_metadata(TEST_USER_ID)
    active_records = manager.get_active_reference_records(metadata)
    active_ids = [r["recording_id"] for r in active_records]

    # Earlier enrollment orders (rec_1..rec_5) must be selected deterministically
    assert active_ids == ["rec_1", "rec_2", "rec_3", "rec_4", "rec_5"]
    assert "rec_6" not in active_ids

    print(f"[OK] Deterministic tie-breaking selected earlier enrollments: {active_ids}")


def test_7_existing_identification_behavior():
    print("\n========================================")
    print("TEST 7: Identification uses only active top five references")
    print("========================================")
    clean_test_dir()

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)
    manager.create_profile(TEST_USER_ID, "Test User")

    emb_target = create_mock_embedding(100)
    emb_other = create_mock_embedding(200)

    # Add 5 target embeddings (duration 15s) and 3 other embeddings (duration 5s)
    for i in range(5):
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb_target,
            usable_speech_duration=15.0
        )
    for i in range(3):
        manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=emb_other,
            usable_speech_duration=5.0
        )

    identifier = SpeakerIdentifier(profile_manager=manager, similarity_threshold=0.70)
    result = identifier.compare_with_profile(new_embedding=emb_target, user_id=TEST_USER_ID)

    # Should compare against top 5 (which are all emb_target with 15s duration)
    assert result["reference_count"] == 5
    assert result["match_count"] == 5
    assert result["is_match"] is True

    print("[OK] Identification correctly uses only the active top 5 reference embeddings.")


def test_8_legacy_profile_loading():
    print("\n========================================")
    print("TEST 8: Legacy profile loading and migration")
    print("========================================")
    clean_test_dir()

    profile_dir = Path(TEST_PROFILES_DIR) / TEST_USER_ID
    embeddings_dir = profile_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Create legacy profile.json without enrollment_records
    legacy_metadata = {
        "user_id": TEST_USER_ID,
        "profile_name": "Legacy User",
        "embedding_dimension": 192,
        "reference_embedding_count": 3,
        "maximum_reference_embeddings": 5,
        "enrollment_complete": False,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "status": "active"
    }
    with open(profile_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(legacy_metadata, f)

    # Create 3 legacy reference files reference_1.pt, reference_2.pt, reference_3.pt
    for i in range(1, 4):
        emb = create_mock_embedding(i)
        torch.save(emb, embeddings_dir / f"reference_{i}.pt")

    manager = VoiceProfileManager(profiles_directory=TEST_PROFILES_DIR)

    # Load legacy profile
    metadata = manager.load_metadata(TEST_USER_ID)
    assert "enrollment_records" in metadata
    assert len(metadata["enrollment_records"]) == 3
    assert metadata["enrollment_records"][0]["usable_speech_duration"] is None

    # Load embeddings
    embeddings = manager.load_all_embeddings(TEST_USER_ID)
    assert len(embeddings) == 3

    # Add a new duration-aware enrollment
    new_emb = create_mock_embedding(99)
    manager.add_reference_embedding(
        user_id=TEST_USER_ID,
        embedding=new_emb,
        usable_speech_duration=10.0
    )

    assert manager.get_total_enrollment_count(TEST_USER_ID) == 4
    updated_metadata = manager.load_metadata(TEST_USER_ID)
    assert len(updated_metadata["enrollment_records"]) == 4

    clean_test_dir()
    print("[OK] Legacy profile loaded without crash, migrated safely, and received new enrollment.")


def main():
    print("\n========================================")
    print("RUNNING DURATION-AWARE POLICY TESTS 1-8")
    print("========================================")

    test_1_fewer_than_five_recordings()
    test_2_exactly_five_recordings()
    test_3_more_than_five_recordings()
    test_4_new_longer_recording()
    test_5_new_shorter_recording()
    test_6_equal_duration_recordings()
    test_7_existing_identification_behavior()
    test_8_legacy_profile_loading()

    print("\n========================================")
    print("ALL DURATION-AWARE POLICY TESTS PASSED!")
    print("========================================\n")


if __name__ == "__main__":
    main()
