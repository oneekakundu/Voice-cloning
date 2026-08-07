from Voice_Cloning.voice_cloning_manager import (
    VoiceCloningManager
)


def test_voice_cloning_manager():

    print(
        "=" * 60
    )

    print(
        "STARTING VOICE CLONING MANAGER TEST"
    )

    print(
        "=" * 60
    )


    user_id = "user_001"


    test_text = (
        "This is a test of the "
        "user-based CARE Doll voice "
        "cloning system."
    )


    voice_cloning_manager = (
        VoiceCloningManager()
    )


    generated_audio = (
        voice_cloning_manager
        .generate_for_user(
            user_id=user_id,
            text=test_text,
            language="en"
        )
    )


    if not generated_audio.exists():

        raise RuntimeError(
            "Voice generation completed, "
            "but the generated audio "
            "file was not found."
        )


    print()

    print(
        "=" * 60
    )

    print(
        "VOICE CLONING MANAGER "
        "TEST PASSED"
    )

    print(
        f"User ID: {user_id}"
    )

    print(
        f"Generated audio: "
        f"{generated_audio}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    test_voice_cloning_manager()