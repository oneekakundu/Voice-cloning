"""
Voice Cloning Pipeline

Thin orchestration layer for the Voice Cloning module.
Receives an identified user ID and text, and invokes VoiceCloningManager
to generate personalized voice output using the user's top-1 reference audio.
"""

from pathlib import Path

from Voice_Cloning.voice_cloning_manager import VoiceCloningManager
from Voice_Cloning.utils import validate_user_id, validate_text
from Voice_Cloning.config import DEFAULT_LANGUAGE


class VoiceCloningPipeline:
    """
    Thin pipeline orchestration layer for Voice Cloning.
    """

    def __init__(self):
        print("Initializing Voice Cloning Pipeline...")
        self.cloning_manager = VoiceCloningManager()
        print("Voice Cloning Pipeline initialized.")

    def generate_for_identified_user(
        self,
        user_id: str,
        text: str,
        output_path: str | Path | None = None,
        language: str = DEFAULT_LANGUAGE
    ) -> Path:
        """
        Generate speech for an already-identified user.

        Parameters
        ----------
        user_id : str
            Already-identified user ID.

        text : str
            Text to convert into speech.

        output_path : str or Path, optional
            Path where generated audio will be saved.

        language : str
            Language code for generated speech.

        Returns
        -------
        Path
            Path to the generated audio file.
        """
        user_id = validate_user_id(user_id)
        text = validate_text(text)

        return self.cloning_manager.generate_for_user(
            user_id=user_id,
            text=text,
            output_path=output_path,
            language=language
        )
