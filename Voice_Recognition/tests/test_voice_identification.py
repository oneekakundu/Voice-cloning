"""
Test Speaker Identification

Pipeline:

    New Audio
        ↓
    SpeakerEncoder
        ↓
    Preprocessing
        ↓
    ECAPA-TDNN
        ↓
    New 192-D Embedding
        ↓
    SpeakerIdentifier
        ↓
    Compare with all 5 references
        ↓
    Majority Vote
        ↓
    Identified User / Unknown
"""

from pathlib import Path

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
# CONFIGURATION
# =========================================================

PROFILE_DIRECTORY = (
    "data/test_voice_profiles"
)

TEST_USER_ID = (
    "test_user_001"
)

TEST_AUDIO_FILE = Path(
    "data/audio/recording_20260727_092300.wav"
)

SIMILARITY_THRESHOLD = 0.70


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "TEST: SPEAKER IDENTIFICATION"
    )

    print(
        "========================================"
    )

    # =====================================================
    # CHECK AUDIO
    # =====================================================

    if not TEST_AUDIO_FILE.exists():

        raise FileNotFoundError(

            f"Identification audio not found:\n"
            f"{TEST_AUDIO_FILE}"

        )

    # =====================================================
    # LOAD PROFILE MANAGER
    # =====================================================

    profile_manager = (
        VoiceProfileManager(

            profiles_directory=(
                PROFILE_DIRECTORY
            )

        )
    )

    # =====================================================
    # CHECK PROFILE
    # =====================================================

    if not profile_manager.profile_exists(

        TEST_USER_ID

    ):

        raise RuntimeError(

            "Voice profile does not exist.\n"
            "Run the enrollment test first."

        )

    reference_count = (

        profile_manager
        .get_reference_count(

            TEST_USER_ID

        )

    )

    print(
        "\n"
        f"Stored references: "
        f"{reference_count}"
    )

    if reference_count != 5:

        raise RuntimeError(

            "Profile must contain exactly "
            "5 reference embeddings before "
            "identification testing."

        )

    # =====================================================
    # LOAD ENCODER
    # =====================================================

    print(
        "\n"
        "Loading ECAPA-TDNN encoder..."
    )

    encoder = (
        SpeakerEncoder()
    )

    # =====================================================
    # LOAD IDENTIFIER
    # =====================================================

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

    # =====================================================
    # GENERATE NEW EMBEDDING
    # =====================================================

    print(
        "\n"
        "Generating embedding for "
        "new audio..."
    )

    new_embedding = (

        encoder.encode(

            TEST_AUDIO_FILE

        )

    )

    print(
        f"Embedding shape: "
        f"{new_embedding.shape}"
    )

    print(
        f"Embedding device: "
        f"{new_embedding.device}"
    )

    assert new_embedding.shape == (
        192,
    )

    # =====================================================
    # IDENTIFY SPEAKER
    # =====================================================

    print(
        "\n"
        "Identifying speaker..."
    )

    result = (

        identifier.identify(

            new_embedding

        )

    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "IDENTIFICATION RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"\n"
        f"Identified: "
        f"{result['identified']}"
    )

    print(
        f"User ID: "
        f"{result['user_id']}"
    )

    print(
        f"Match count: "
        f"{result['match_count']}"
    )

    print(
        f"Reference count: "
        f"{result['reference_count']}"
    )

    print(
        f"Average similarity: "
        f"{result['average_similarity']:.4f}"
    )

    print(
        f"Maximum similarity: "
        f"{result['maximum_similarity']:.4f}"
    )

    # =====================================================
    # PROFILE DETAILS
    # =====================================================

    print(
        "\n"
        "Individual reference similarities:"
    )

    for profile_result in (

        result[
            "profile_results"
        ]

    ):

        print(
            f"\n"
            f"Profile: "
            f"{profile_result['user_id']}"
        )

        for index, similarity in enumerate(

            profile_result[
                "similarities"
            ],

            start=1

        ):

            print(

                f"  reference_{index}: "
                f"{similarity:.4f}"

            )

        print(

            f"  Matches: "
            f"{profile_result['match_count']}"

        )

    # =====================================================
    # ASSERT IDENTIFICATION
    # =====================================================

    assert result[
        "identified"
    ] is True

    assert result[
        "user_id"
    ] == TEST_USER_ID

    print(
        "\n"
        "✓ Speaker identified successfully"
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "SPEAKER IDENTIFICATION TEST PASSED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()