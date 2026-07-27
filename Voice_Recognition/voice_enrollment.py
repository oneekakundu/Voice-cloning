"""
Voice Recognition Pipeline

Connects:

    Audio File
        ↓
    Preprocessing
        ↓
    ECAPA-TDNN Encoder
        ↓
    Voice Profile Manager
        ↓
    Speaker Identifier

Supports:

    1. Incremental voice enrollment
    2. Speaker identification
"""

from pathlib import Path

from Voice_Recognition.speaker_preprocessing import (
    SpeakerPreprocessor
)

from Voice_Recognition.speaker_encoder import (
    SpeakerEncoder
)

from Voice_Recognition.voice_profile_manager import (
    VoiceProfileManager
)

from Voice_Recognition.speaker_identifier import (
    SpeakerIdentifier
)


class VoiceRecognitionPipeline:

    def __init__(
        self,
        profiles_directory: str = (
            "data/voice_profiles"
        ),
        similarity_threshold: float = 0.70
    ):
        """
        Initialize the complete Voice Recognition system.
        """

        print(
            "Initializing Voice Recognition Pipeline..."
        )

        # -----------------------------------------
        # PREPROCESSOR
        # -----------------------------------------

        self.preprocessor = (
            SpeakerPreprocessor()
        )

        # -----------------------------------------
        # ECAPA-TDNN ENCODER
        # -----------------------------------------

        self.encoder = (
            SpeakerEncoder()
        )

        # -----------------------------------------
        # PROFILE MANAGER
        # -----------------------------------------

        self.profile_manager = (
            VoiceProfileManager(
                profiles_directory=(
                    profiles_directory
                )
            )
        )

        # -----------------------------------------
        # SPEAKER IDENTIFIER
        # -----------------------------------------

        self.identifier = (
            SpeakerIdentifier(

                profile_manager=(
                    self.profile_manager
                ),

                similarity_threshold=(
                    similarity_threshold
                )

            )
        )

        print(
            "Voice Recognition Pipeline initialized."
        )

    # =====================================================
    # AUDIO → EMBEDDING
    # =====================================================

    def generate_embedding(
        self,
        audio_path: str
    ):
        """
        Convert audio file into a 192-D embedding.

        Flow:

            Audio
              ↓
            Preprocessing
              ↓
            ECAPA-TDNN
              ↓
            Embedding
        """

        audio_path = Path(
            audio_path
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: "
                f"{audio_path}"
            )

        # Your existing encoder should internally
        # use the preprocessing pipeline if that
        # is how your current implementation is designed.

        embedding = (
            self.encoder.encode(
                str(audio_path)
            )
        )

        return embedding

    # =====================================================
    # ENROLLMENT
    # =====================================================

    def enroll(
        self,
        audio_path: str,
        user_id: str,
        profile_name: str = "Primary User"
    ) -> dict:
        """
        Process one audio recording for enrollment.

        Behavior:

            First audio:
                Save as reference_1

            Second audio:
                Compare with reference_1
                If match → save reference_2

            ...

            Fifth audio:
                Verify and save reference_5

            After five:
                Enrollment complete
        """

        # -----------------------------------------
        # GENERATE NEW EMBEDDING
        # -----------------------------------------

        new_embedding = (
            self.generate_embedding(
                audio_path
            )
        )

        # -----------------------------------------
        # FIRST ENROLLMENT
        # -----------------------------------------

        if not self.profile_manager.profile_exists(
            user_id
        ):

            self.profile_manager.create_profile(

                user_id=user_id,

                profile_name=profile_name

            )

            reference_number = (
                self.profile_manager
                .add_reference_embedding(

                    user_id=user_id,

                    embedding=new_embedding

                )
            )

            return {

                "success": True,

                "status": "first_reference_saved",

                "user_id": user_id,

                "reference_number": (
                    reference_number
                ),

                "enrollment_complete": False

            }

        # -----------------------------------------
        # CHECK IF PROFILE IS FULL
        # -----------------------------------------

        if self.profile_manager.is_enrollment_complete(
            user_id
        ):

            return {

                "success": False,

                "status": "enrollment_already_complete",

                "user_id": user_id,

                "message": (
                    "Profile already contains "
                    "5 reference embeddings."
                )

            }

        # -----------------------------------------
        # VERIFY NEW RECORDING
        # -----------------------------------------

        verification = (
            self.identifier
            .compare_with_profile(

                new_embedding=new_embedding,

                user_id=user_id

            )
        )

        # -----------------------------------------
        # REJECT DIFFERENT SPEAKER
        # -----------------------------------------

        if not verification[
            "is_match"
        ]:

            return {

                "success": False,

                "status": "speaker_verification_failed",

                "user_id": user_id,

                "verification": verification,

                "message": (
                    "New recording did not match "
                    "the existing voice profile."
                )

            }

        # -----------------------------------------
        # ACCEPT NEW REFERENCE
        # -----------------------------------------

        reference_number = (
            self.profile_manager
            .add_reference_embedding(

                user_id=user_id,

                embedding=new_embedding

            )
        )

        current_count = (
            self.profile_manager
            .get_reference_count(
                user_id
            )
        )

        return {

            "success": True,

            "status": "reference_saved",

            "user_id": user_id,

            "reference_number": (
                reference_number
            ),

            "reference_count": (
                current_count
            ),

            "enrollment_complete": (

                current_count
                >= 5

            ),

            "verification": verification

        }

    # =====================================================
    # IDENTIFICATION
    # =====================================================

    def identify(
        self,
        audio_path: str
    ) -> dict:
        """
        Identify the speaker from an audio file.

        Flow:

            Audio
              ↓
            Preprocessing
              ↓
            Encoder
              ↓
            Temporary Embedding
              ↓
            Speaker Identifier
              ↓
            User / Unknown
        """

        new_embedding = (
            self.generate_embedding(
                audio_path
            )
        )

        result = (
            self.identifier.identify(
                new_embedding
            )
        )

        # new_embedding is now no longer
        # needed after identification.
        #
        # It remains only as a local variable
        # during this method execution.

        return result