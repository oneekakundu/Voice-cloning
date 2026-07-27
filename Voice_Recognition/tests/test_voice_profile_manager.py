import shutil
from pathlib import Path
import torch
from Voice_Recognition.voice_profile_manager import VoiceProfileManager

# =========================================================
# TEST CONFIGURATION
# =========================================================

TEST_DIRECTORY = "data/test_voice_profiles"
TEST_USER_ID = "test_user_001"
TEST_PROFILE_NAME = "Test User"
EMBEDDING_DIMENSION = 192
MAX_REFERENCES = 5

# =========================================================
# CREATE DETERMINISTIC TEST EMBEDDINGS
# =========================================================

def create_test_embeddings():
    embeddings = []
    for index in range(MAX_REFERENCES):
        torch.manual_seed(index)
        embedding = torch.randn(EMBEDDING_DIMENSION)
        embeddings.append(embedding)
    return embeddings

# =========================================================
# TEST PROFILE CREATION
# =========================================================

def test_create_profile():
    print("\n========================================")
    print("TEST: CREATE EMPTY VOICE PROFILE")
    print("========================================")
    
    test_directory = Path(TEST_DIRECTORY)
    if test_directory.exists():
        shutil.rmtree(test_directory)
        
    manager = VoiceProfileManager(profiles_directory=TEST_DIRECTORY)
    
    result = manager.create_profile(
        user_id=TEST_USER_ID,
        profile_name=TEST_PROFILE_NAME
    )
    
    assert result is True
    assert manager.profile_exists(TEST_USER_ID)
    print("OK: Empty profile created successfully")
    
    reference_count = manager.get_reference_count(TEST_USER_ID)
    assert reference_count == 0
    print("OK: Initial reference count is 0")
    
    return manager

def test_add_embeddings(manager, original_embeddings):
    print("\n========================================")
    print("TEST: ADD REFERENCE EMBEDDINGS")
    print("========================================")
    
    for index, embedding in enumerate(original_embeddings, start=1):
        print(f"\nAdding reference embedding {index}...")
        reference_number = manager.add_reference_embedding(
            user_id=TEST_USER_ID,
            embedding=embedding
        )
        assert reference_number == index
        print(f"OK: reference_{index}.pt saved successfully")
        
        current_count = manager.get_reference_count(TEST_USER_ID)
        assert current_count == index
        print(f"OK: Current reference count: {current_count}")
        
    final_count = manager.get_reference_count(TEST_USER_ID)
    assert final_count == MAX_REFERENCES
    print(f"OK: Final reference count is {final_count}")

def test_load_all_embeddings(manager, original_embeddings):
    print("\n========================================")
    print("TEST: LOAD ALL EMBEDDINGS")
    print("========================================")
    
    loaded_embeddings = manager.load_all_embeddings(TEST_USER_ID)
    assert len(loaded_embeddings) == MAX_REFERENCES
    print(f"OK: Loaded {len(loaded_embeddings)} embeddings")
    
    for index in range(MAX_REFERENCES):
        assert torch.allclose(loaded_embeddings[index], original_embeddings[index])
    print("OK: All embeddings match original")

def test_metadata(manager):
    print("\n========================================")
    print("TEST: METADATA")
    print("========================================")
    
    metadata = manager.load_metadata(TEST_USER_ID)
    assert metadata["user_id"] == TEST_USER_ID
    assert metadata["profile_name"] == TEST_PROFILE_NAME
    assert metadata["embedding_dimension"] == EMBEDDING_DIMENSION
    assert metadata["maximum_reference_embeddings"] == MAX_REFERENCES
    assert metadata["status"] == "active"
    assert metadata["enrollment_complete"] is True
    
    print("OK: Metadata verified successfully")

def test_single_embedding_loading(manager):
    print("\n========================================")
    print("TEST: SINGLE EMBEDDING LOADING")
    print("========================================")
    
    embedding = manager.load_embedding(TEST_USER_ID, reference_number=1)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (EMBEDDING_DIMENSION,)
    print("OK: Loaded single embedding successfully")

def test_list_profiles(manager):
    print("\n========================================")
    print("TEST: LIST PROFILES")
    print("========================================")
    
    profiles = manager.list_profiles()
    assert len(profiles) == 1
    assert profiles[0]["user_id"] == TEST_USER_ID
    print(f"OK: Profile list contains {len(profiles)} profile(s)")

def test_sixth_embedding_rejected(manager):
    print("\n========================================")
    print("TEST: SIXTH EMBEDDING REJECTED")
    print("========================================")
    
    extra_embedding = torch.randn(EMBEDDING_DIMENSION)
    try:
        manager.add_reference_embedding(user_id=TEST_USER_ID, embedding=extra_embedding)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("OK: Successfully rejected 6th embedding")

def test_delete_profile(manager):
    print("\n========================================")
    print("TEST: DELETE PROFILE")
    print("========================================")
    
    result = manager.delete_profile(TEST_USER_ID)
    assert result is True
    assert not manager.profile_exists(TEST_USER_ID)
    print("OK: Profile deleted successfully")
    
    test_directory = Path(TEST_DIRECTORY)
    if test_directory.exists():
        shutil.rmtree(test_directory)

def main():
    print("\n" + "="*50)
    print("VOICE PROFILE MANAGER TESTS")
    print("="*50)
    
    manager = test_create_profile()
    original_embeddings = create_test_embeddings()
    test_add_embeddings(manager, original_embeddings)
    test_load_all_embeddings(manager, original_embeddings)
    test_metadata(manager)
    test_single_embedding_loading(manager)
    test_list_profiles(manager)
    test_sixth_embedding_rejected(manager)
    test_delete_profile(manager)
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
