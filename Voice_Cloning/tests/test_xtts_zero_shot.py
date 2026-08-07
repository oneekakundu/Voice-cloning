from pathlib import Path

from Voice_Cloning.config import (
    PROJECT_ROOT,
    GENERATED_AUDIO_DIRECTORY,
    create_required_directories
)

from Voice_Cloning.xtts_zero_shot import (
    XTTSZeroShotCloner
)


def test_xtts_zero_shot():

    print(
        "=" * 60
    )

    print(
        "STARTING XTTS-v2 ZERO-SHOT TEST"
    )

    print(
        "=" * 60
    )


    create_required_directories()


    reference_audio_path = (
        PROJECT_ROOT
        / "data"
        / "voice_cloning"
        / "references"
        / "user_001"
        /  "recording_20260727_092147.wav"
    )


    output_path = (
        GENERATED_AUDIO_DIRECTORY
        / "test_user_001_output.wav"
    )


    test_text = (
        "This is a test of the "
        "CARE Doll voice cloning module."
    )


    print(
        f"Reference audio: "
        f"{reference_audio_path}"
    )


    print(
        f"Output audio: "
        f"{output_path}"
    )


    if not reference_audio_path.exists():

        raise FileNotFoundError(
            "\nReference audio is missing.\n"
            "Place a clean WAV file at:\n"
            f"{reference_audio_path}"
        )


    cloner = XTTSZeroShotCloner()


    generated_file = cloner.generate(
        text=test_text,
        reference_audio_path=(
            reference_audio_path
        ),
        output_path=output_path,
        language="en"
    )


    print(
        "\n"
        "=" * 60
    )

    print(
        "XTTS-v2 ZERO-SHOT TEST PASSED"
    )

    print(
        f"Generated file: "
        f"{generated_file}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    test_xtts_zero_shot()