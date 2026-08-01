"""
Tests for Unknown Speaker Auto-Enrollment & Incremental Embedding Updates

Verifies:
1. Test 1: Unknown speaker creates a new user (user_002)
2. Test 2: Another unknown speaker creates the next user (user_003)
3. Test 3: Known speaker does not create a new user
4. Test 4: Incremental embedding (3 references -> 4 references)
5. Test 5: Maximum embedding limit (5 embeddings max, 6th rejected)
6. Test 6: Persistence across pipeline restarts
"""

import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from Voice_Recognition.voice_enrollment import VoiceRecognitionPipeline
from Voice_Recognition.voice_profile_manager import VoiceProfileManager



TEST_PROFILES_DIR = "data/test_unknown_speaker_profiles"


def clean_test_dir():
    p = Path(TEST_PROFILES_DIR)
    if p.exists():
        shutil.rmtree(p)


def create_vector(index: int) -> torch.Tensor:
    """Create orthogonal 192-dim vector for deterministic testing."""
    vec = torch.zeros(192)
    vec[index] = 1.0
    return vec


def test_1_unknown_speaker_creates_user_002():
    print("\n========================================")
    print("TEST 1: Unknown speaker creates a new user (user_002)")
    print("========================================")
    clean_test_dir()

    pipeline = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)

    # Initial profile: user_001
    v1 = create_vector(0)
    pipeline.profile_manager.create_profile("user_001", "User One")
    pipeline.profile_manager.add_reference_embedding("user_001", v1)

    # Input: speaker 2 (v2) does not match user_001
    v2 = create_vector(1)
    result = pipeline.identify(v2)

    assert result["newly_enrolled"] is True
    assert result["user_id"] == "user_002"
    assert result["reference_count"] == 1
    assert pipeline.profile_manager.profile_exists("user_002")

    stored_embeddings = pipeline.profile_manager.load_all_embeddings("user_002")
    assert len(stored_embeddings) == 1
    assert torch.allclose(stored_embeddings[0], v2)

    print("[OK] Test 1 Passed: user_002 created and 1st embedding persisted.")


def test_2_another_unknown_speaker_creates_user_003():
    print("\n========================================")
    print("TEST 2: Another unknown speaker creates next user (user_003)")
    print("========================================")

    pipeline = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)
    assert pipeline.profile_manager.profile_exists("user_001")
    assert pipeline.profile_manager.profile_exists("user_002")

    # Input: speaker 3 (v3) matches neither user_001 nor user_002
    v3 = create_vector(2)
    result = pipeline.identify(v3)

    assert result["newly_enrolled"] is True
    assert result["user_id"] == "user_003"
    assert pipeline.profile_manager.profile_exists("user_003")

    print("[OK] Test 2 Passed: user_003 created successfully.")


def test_3_known_speaker_does_not_create_new_user():
    print("\n========================================")
    print("TEST 3: Known speaker does not create a new user")
    print("========================================")

    pipeline = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)

    # Input: recording/embedding matching user_001 (v1)
    v1 = create_vector(0)
    result = pipeline.identify(v1)

    assert result["identified"] is True
    assert result["user_id"] == "user_001"
    assert result["newly_enrolled"] is False
    assert not pipeline.profile_manager.profile_exists("user_004")

    print("[OK] Test 3 Passed: Known speaker identified as user_001, no new user created.")


def test_4_incremental_embedding():
    print("\n========================================")
    print("TEST 4: Incremental embedding")
    print("========================================")

    clean_test_dir()
    pipeline = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)

    # Initial state: user_001 has 3 embeddings
    v1 = create_vector(0)
    pipeline.profile_manager.create_profile("user_001", "User One")
    for _ in range(3):
        pipeline.profile_manager.add_reference_embedding("user_001", v1)

    assert pipeline.profile_manager.get_reference_count("user_001") == 3

    # Input: recording matching user_001
    result = pipeline.identify(v1)

    assert result["identified"] is True
    assert result["user_id"] == "user_001"
    assert result["embedding_added"] is True
    assert result["reference_count"] == 4
    assert pipeline.profile_manager.get_reference_count("user_001") == 4

    print("[OK] Test 4 Passed: user_001 reference count updated from 3 to 4.")


def test_5_maximum_embedding_limit():
    print("\n========================================")
    print("TEST 5: Maximum embedding limit (5 embeddings)")
    print("========================================")

    clean_test_dir()
    pipeline = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)

    # Initial state: user_001 has 5 embeddings
    v1 = create_vector(0)
    pipeline.profile_manager.create_profile("user_001", "User One")
    for _ in range(5):
        pipeline.profile_manager.add_reference_embedding("user_001", v1)

    assert pipeline.profile_manager.get_reference_count("user_001") == 5

    # Input: recording matching user_001
    result = pipeline.identify(v1)

    assert result["identified"] is True
    assert result["user_id"] == "user_001"
    assert result["embedding_added"] is False
    assert result["reference_count"] == 5
    assert pipeline.profile_manager.get_reference_count("user_001") == 5
    assert not pipeline.profile_manager.profile_exists("user_002")

    print("[OK] Test 5 Passed: user_001 kept at 5 embeddings, 6th rejected.")


def test_6_persistence():
    print("\n========================================")
    print("TEST 6: Persistence across pipeline restart")
    print("========================================")

    clean_test_dir()

    # Session 1: Create user_001 and user_002
    pipe1 = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)
    v1 = create_vector(0)
    pipe1.profile_manager.create_profile("user_001", "User One")
    pipe1.profile_manager.add_reference_embedding("user_001", v1)

    v2 = create_vector(1)
    res1 = pipe1.identify(v2)
    assert res1["user_id"] == "user_002"

    del pipe1

    # Session 2: Restart pipeline and test identification
    pipe2 = VoiceRecognitionPipeline(profiles_directory=TEST_PROFILES_DIR)
    assert pipe2.profile_manager.profile_exists("user_001")
    assert pipe2.profile_manager.profile_exists("user_002")

    # Present v2 again - should match user_002!
    res2 = pipe2.identify(v2)
    assert res2["identified"] is True
    assert res2["user_id"] == "user_002"
    # Incremental update: reference count for user_002 should become 2!
    assert res2["reference_count"] == 2

    clean_test_dir()
    print("[OK] Test 6 Passed: user_002 persisted and matched after restart.")



def main():
    print("\n========================================")
    print("RUNNING SPEAKER PROFILE TESTS 1-6")
    print("========================================")
    test_1_unknown_speaker_creates_user_002()
    test_2_another_unknown_speaker_creates_user_003()
    test_3_known_speaker_does_not_create_new_user()
    test_4_incremental_embedding()
    test_5_maximum_embedding_limit()
    test_6_persistence()
    print("\n========================================")
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("========================================\n")


if __name__ == "__main__":
    main()
