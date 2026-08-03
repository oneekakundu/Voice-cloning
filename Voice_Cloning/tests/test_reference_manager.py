from Voice_Cloning.reference_manager import (
    ReferenceManager
)


def test_reference_manager():

    print(
        "=" * 60
    )

    print(
        "STARTING REFERENCE MANAGER TEST"
    )

    print(
        "=" * 60
    )


    user_id = "user_001"


    reference_manager = (
        ReferenceManager()
    )


    reference_audio = (
        reference_manager.get_reference_audio(
            user_id
        )
    )


    print()

    print(
        "Reference audio successfully "
        "found."
    )

    print(
        f"User ID: {user_id}"
    )

    print(
        f"Reference audio: "
        f"{reference_audio}"
    )


    if not reference_audio.exists():

        raise RuntimeError(
            "Reference Manager returned "
            "a path, but the file "
            "does not exist."
        )


    print()

    print(
        "=" * 60
    )

    print(
        "REFERENCE MANAGER TEST PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    test_reference_manager()