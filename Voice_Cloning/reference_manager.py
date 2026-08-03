from pathlib import Path

from Voice_Cloning.config import (
    REFERENCE_AUDIO_DIRECTORY,
    SUPPORTED_AUDIO_EXTENSIONS
)

from Voice_Cloning.utils import (
    validate_user_id
)

from Voice_Recognition.voice_profile_manager import (
    VoiceProfileManager
)


class ReferenceManager:
    """
    Manages user-specific reference audio files.

    Current version:
        - One reference audio file per user.
        - The reference audio is found automatically
          inside the user's reference directory.

    Future version:
        - Multiple reference audio files.
        - Reference quality scoring.
        - Automatic best-reference selection.
    """

    def __init__(
        self,
        reference_directory=REFERENCE_AUDIO_DIRECTORY,
        profile_manager=None
    ):

        self.reference_directory = Path(
            reference_directory
        )

        self.profile_manager = (
            profile_manager
            or VoiceProfileManager()
        )

        print(
            "Reference Manager initialized."
        )

        print(
            "Reference directory: "
            f"{self.reference_directory}"
        )


    def get_user_reference_directory(
        self,
        user_id
    ):
        """
        Return the reference-audio directory
        for a specific user.

        Parameters
        ----------
        user_id : str
            User identifier.

        Returns
        -------
        Path
            User reference directory.
        """

        user_id = validate_user_id(
            user_id
        )

        return (
            self.reference_directory
            / user_id
        )


    def get_reference_audio(
        self,
        user_id
    ):
        """
        Find and return the single top-ranked reference audio for a user.
        Reuses VoiceProfileManager.get_active_reference_audio_paths(user_id)
        and selects only the first path (top-1 / longest valid recording).
        """

        user_id = validate_user_id(
            user_id
        )

        try:
            active_paths = (
                self.profile_manager
                .get_active_reference_audio_paths(
                    user_id
                )
            )
            if active_paths:
                top_reference_path = Path(active_paths[0])
                if not top_reference_path.exists():
                    raise FileNotFoundError(
                        f"Top reference audio file does not exist: {top_reference_path}"
                    )
                if not top_reference_path.is_file():
                    raise ValueError(
                        f"Top reference audio path is not a file: {top_reference_path}"
                    )
                print(
                    "Top-1 reference audio selected "
                    f"for user '{user_id}': {top_reference_path}"
                )
                return top_reference_path
        except Exception:
            pass

        user_directory = (
            self.get_user_reference_directory(
                user_id
            )
        )

        if not user_directory.exists():
            raise FileNotFoundError(
                "Reference directory was not "
                f"found for user '{user_id}'.\n"
                "Expected directory:\n"
                f"{user_directory}"
            )

        if not user_directory.is_dir():
            raise NotADirectoryError(
                "The user reference path exists "
                "but is not a directory:\n"
                f"{user_directory}"
            )

        audio_files = []

        for file_path in user_directory.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_AUDIO_EXTENSIONS
            ):
                audio_files.append(
                    file_path
                )

        if not audio_files:
            supported_extensions = (
                ", ".join(
                    sorted(
                        SUPPORTED_AUDIO_EXTENSIONS
                    )
                )
            )
            raise FileNotFoundError(
                "No supported reference audio "
                f"was found for user '{user_id}'.\n"
                f"Directory:\n"
                f"{user_directory}\n"
                "Supported extensions:\n"
                f"{supported_extensions}"
            )

        audio_files.sort(
            key=lambda path: path.name.lower()
        )

        selected_reference = (
            audio_files[0]
        )

        print(
            "Reference audio selected "
            f"for user '{user_id}':"
        )
        print(
            selected_reference
        )

        return selected_reference