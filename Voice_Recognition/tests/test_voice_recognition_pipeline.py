"""
Test Incremental Voice Enrollment Pipeline

Pipeline:

    Audio File
        ↓
    SpeakerEncoder
        ↓
    Preprocessing
        ↓
    ECAPA-TDNN
        ↓
    192-D Embedding
        ↓
    Speaker Verification
        ↓
    Voice Profile Manager
        ↓
    reference_1.pt
    reference_2.pt
    ...
    reference_5.pt
"""

from pathlib import Path
import shutil

from Voice_Recognition.speaker_encoder import (
    SpeakerEncoder
)

from Voice_Recognition.voice_profile_manager import (
    VoiceProfileManager
)

from Voice_Recognition.speaker_identifier import (
    SpeakerIdentifier
)


# =========================================================
# TEST CONFIGURATION
# =========================================================

TEST_PROFILE_DIRECTORY = (
    "data/test_voice_profiles"
)

TEST_USER_ID = (
    "test_user_001"
)

TEST_PROFILE_NAME = (
    "Test User"
)

# Use five recordings from the SAME speaker.
#
# Replace these filenames with your actual files.

AUDIO_FILES = [

    Path(
        "data/audio/recording_20260727_092147.wav"
    ),

    Path(
        "data/audio/recording_20260727_092200.wav"
    ),

    Path(
        "data/audio/recording_20260727_092215.wav"
    ),

    Path(
        "data/audio/recording_20260727_092230.wav"
    ),

    Path(
        "data/audio/recording_20260727_092245.wav"
    )

]


# This threshold is only a starting point.
#
# It should eventually be calibrated using
# genuine and different-speaker recordings.

SIMILARITY_THRESHOLD = 0.70


# =========================================================
# CHECK AUDIO FILES
# =========================================================

def verify_audio_files():

    print(
        "\n"
        "========================================"
    )

    print(
        "TEST: VERIFY AUDIO FILES"
    )

    print(
        "========================================"
    )

    for audio_file in AUDIO_FILES:

        if not audio_file.exists():

            raise FileNotFoundError(

                f"Audio file not found:\n"
                f"{audio_file}"

            )

        print(
            f"✓ Found: {audio_file}"
        )

    print(
        "\n"
        "✓ All audio files found"
    )


# =========================================================
# CLEAN OLD TEST PROFILE
# =========================================================

def clean_previous_profile():

    test_directory = Path(
        TEST_PROFILE_DIRECTORY
    )

    if test_directory.exists():

        shutil.rmtree(
            test_directory
        )

        print(
            "\n"
            "✓ Previous test profile deleted"
        )


# =========================================================
# INITIALIZE COMPONENTS
# =========================================================

def initialize_components():

    print(
        "\n"
        "========================================"
    )

    print(
        "INITIALIZING VOICE RECOGNITION COMPONENTS"
    )

    print(
        "========================================"
    )

    print(
        "\n"
        "Loading speaker encoder..."
    )

    encoder = SpeakerEncoder()

    profile_manager = (
        VoiceProfileManager(

            profiles_directory=(
                TEST_PROFILE_DIRECTORY
            )

        )
    )

    identifier = (
        SpeakerIdentifier(

            profile_manager=(
                profile_manager
            ),

            similarity_threshold=(
                SIMILARITY_THRESHOLD
            )

        )
    )

    print(
        "\n"
        "✓ All components initialized"
    )

    return (

        encoder,

        profile_manager,

        identifier

    )


# =========================================================
# ENROLL ONE AUDIO FILE
# =========================================================

def enroll_audio(

    encoder,

    profile_manager,

    identifier,

    audio_file,

    reference_index

):

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        f"ENROLLMENT RECORDING "
        f"{reference_index}"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Audio: {audio_file}"
    )

    print(
        "\n"
        "Generating speaker embedding..."
    )

    embedding = (
        encoder.encode(
            audio_file
        )
    )

    print(
        f"Embedding shape: "
        f"{embedding.shape}"
    )

    print(
        f"Embedding device: "
        f"{embedding.device}"
    )

    assert embedding.shape == (
        192,
    )

    # =====================================================
    # FIRST REFERENCE
    # =====================================================

    if not profile_manager.profile_exists(
        TEST_USER_ID
    ):

        print(
            "\n"
            "No existing profile found."
        )

        print(
            "Creating new voice profile..."
        )

        profile_manager.create_profile(

            user_id=TEST_USER_ID,

            profile_name=TEST_PROFILE_NAME

        )

        saved_reference_number = (

            profile_manager
            .add_reference_embedding(

                user_id=TEST_USER_ID,

                embedding=embedding

            )

        )

        assert saved_reference_number == 1

        print(
            "\n"
            "✓ First reference embedding saved"
        )

        return True

    # =====================================================
    # PROFILE ALREADY EXISTS
    # =====================================================

    current_count = (

        profile_manager
        .get_reference_count(

            TEST_USER_ID

        )

    )

    print(
        "\n"
        f"Existing references: "
        f"{current_count}"
    )

    # =====================================================
    # PROFILE FULL
    # =====================================================

    if profile_manager.is_enrollment_complete(

        TEST_USER_ID

    ):

        print(
            "\n"
            "⚠ Profile already contains "
            "5 references."
        )

        return False

    # =====================================================
    # VERIFY NEW RECORDING
    # =====================================================

    print(
        "\n"
        "Comparing new recording with "
        "existing references..."
    )

    verification = (

        identifier
        .compare_with_profile(

            new_embedding=embedding,

            user_id=TEST_USER_ID

        )

    )

    print(
        "\n"
        "Verification results:"
    )

    print(
        f"Similarities: "
        f"{verification['similarities']}"
    )

    print(
        f"Match count: "
        f"{verification['match_count']}"
    )

    print(
        f"Required matches: "
        f"{verification['required_matches']}"
    )

    print(
        f"Average similarity: "
        f"{verification['average_similarity']:.4f}"
    )

    print(
        f"Maximum similarity: "
        f"{verification['maximum_similarity']:.4f}"
    )

    print(
        f"Is match: "
        f"{verification['is_match']}"
    )

    # =====================================================
    # REJECT DIFFERENT SPEAKER
    # =====================================================

    if not verification["is_match"]:

        print(
            "\n"
            "✗ Enrollment rejected."
        )

        print(
            "The new recording did not match "
            "the existing profile."
        )

        return False

    # =====================================================
    # SAVE NEW REFERENCE
    # =====================================================

    saved_reference_number = (

        profile_manager
        .add_reference_embedding(

            user_id=TEST_USER_ID,

            embedding=embedding

        )

    )

    expected_reference_number = (
        current_count + 1
    )

    assert saved_reference_number == (
        expected_reference_number
    )

    print(
        "\n"
        f"✓ Reference "
        f"{saved_reference_number} "
        f"saved successfully"
    )

    return True


# =========================================================
# VERIFY FINAL PROFILE
# =========================================================

def verify_final_profile(
    profile_manager
):

    print(
        "\n"
        "========================================"
    )

    print(
        "VERIFYING FINAL VOICE PROFILE"
    )

    print(
        "========================================"
    )

    metadata = (

        profile_manager
        .load_metadata(

            TEST_USER_ID

        )

    )

    reference_count = (

        profile_manager
        .get_reference_count(

            TEST_USER_ID

        )

    )

    print(
        f"\n"
        f"Final reference count: "
        f"{reference_count}"
    )

    print(
        f"Enrollment complete: "
        f"{metadata['enrollment_complete']}"
    )

    assert reference_count == 5

    assert metadata[
        "enrollment_complete"
    ] is True

    embeddings = (

        profile_manager
        .load_all_embeddings(

            TEST_USER_ID

        )

    )

    assert len(
        embeddings
    ) == 5

    for index, embedding in enumerate(
        embeddings,
        start=1
    ):

        print(
            f"✓ reference_{index}.pt "
            f"shape: {embedding.shape}"
        )

        assert embedding.shape == (
            192,
        )

    print(
        "\n"
        "✓ All 5 reference embeddings "
        "verified successfully"
    )


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "TEST: INCREMENTAL VOICE ENROLLMENT"
    )

    print(
        "========================================"
    )

    verify_audio_files()

    clean_previous_profile()

    (

        encoder,

        profile_manager,

        identifier

    ) = initialize_components()

    successful_enrollments = 0

    for index, audio_file in enumerate(

        AUDIO_FILES,

        start=1

    ):

        success = enroll_audio(

            encoder=encoder,

            profile_manager=profile_manager,

            identifier=identifier,

            audio_file=audio_file,

            reference_index=index

        )

        if success:

            successful_enrollments += 1

        else:

            print(
                "\n"
                f"⚠ Recording {index} "
                "was not enrolled."
            )

            break

    if successful_enrollments == 5:

        verify_final_profile(
            profile_manager
        )

        print(
            "\n"
            "========================================"
        )

        print(
            "VOICE ENROLLMENT TEST PASSED"
        )

        print(
            "========================================"
        )

    else:

        print(
            "\n"
            "========================================"
        )

        print(
            "VOICE ENROLLMENT TEST DID NOT "
            "COMPLETE 5 REFERENCES"
        )

        print(
            "========================================"
        )


if __name__ == "__main__":

    main()