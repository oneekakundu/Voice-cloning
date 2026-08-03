from pathlib import Path

from Voice_Cloning.config import (
    GENERATED_AUDIO_DIRECTORY,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_LANGUAGE,
    create_required_directories
)

from Voice_Cloning.xtts_zero_shot import (
    XTTSZeroShotCloner
)

from Voice_Cloning.reference_manager import (
    ReferenceManager
)

from Voice_Cloning.utils import (
    ensure_directory,
    validate_user_id,
    validate_text
)


class VoiceCloningManager:
    """
    Main manager for the Voice Cloning module.

    Current functionality:
        1. Accept a user ID.
        2. Find that user's reference audio.
        3. Generate speech using XTTS-v2 zero-shot cloning.
        4. Save and return the generated audio.

    Future functionality:
        1. Check for an approved trained user model.
        2. Use the trained model when available.
        3. Fall back to XTTS zero-shot cloning.
    """

    def __init__(self):

        print(
            "Initializing Voice Cloning Manager..."
        )

        create_required_directories()


        self.reference_manager = (
            ReferenceManager()
        )


        self.zero_shot_cloner = (
            XTTSZeroShotCloner()
        )


        print(
            "Voice Cloning Manager "
            "initialized successfully."
        )


    def get_default_output_path(
        self,
        user_id
    ):
        """
        Create the default output path for a user.

        Parameters
        ----------
        user_id : str
            User identifier.

        Returns
        -------
        Path
            Default generated-audio path.
        """

        user_id = validate_user_id(
            user_id
        )


        user_output_directory = (
            GENERATED_AUDIO_DIRECTORY
            / user_id
        )


        ensure_directory(
            user_output_directory
        )


        return (
            user_output_directory
            / DEFAULT_OUTPUT_FILENAME
        )


    def generate_for_user(
        self,
        user_id,
        text,
        output_path=None,
        language=DEFAULT_LANGUAGE
    ):
        """
        Generate speech using the voice
        associated with a specific user.

        Parameters
        ----------
        user_id : str
            Identified user ID.

        text : str
            Text to convert into speech.

        output_path : str or Path, optional
            Output path for generated audio.

            If None, the default user-specific
            output path is used.

        language : str
            XTTS language code.

        Returns
        -------
        Path
            Generated audio-file path.
        """

        user_id = validate_user_id(
            user_id
        )


        text = validate_text(
            text
        )


        print()

        print(
            "Starting user-based "
            "voice generation..."
        )

        print(
            f"User ID: {user_id}"
        )


        reference_audio = (
            self.reference_manager
            .get_reference_audio(
                user_id
            )
        )


        if output_path is None:

            output_path = (
                self.get_default_output_path(
                    user_id
                )
            )

        else:

            output_path = Path(
                output_path
            )

            ensure_directory(
                output_path.parent
            )


        generated_audio = (
            self.zero_shot_cloner.generate(
                text=text,
                reference_audio_path=(
                    reference_audio
                ),
                output_path=output_path,
                language=language
            )
        )


        print()

        print(
            "User-based voice generation "
            "completed successfully."
        )

        print(
            f"Generated audio: "
            f"{generated_audio}"
        )


        return generated_audio