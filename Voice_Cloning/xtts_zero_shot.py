from pathlib import Path

from TTS.api import TTS

from Voice_Cloning.config import (
    XTTS_MODEL_NAME,
    DEVICE,
    DEFAULT_LANGUAGE,
    SPLIT_SENTENCES
)

from Voice_Cloning.utils import (
    ensure_directory,
    validate_text
)


class XTTSZeroShotCloner:
    """
    XTTS-v2 zero-shot voice-cloning engine.

    The model is loaded only when it is first needed.
    This is called lazy loading.
    """

    def __init__(self):
        self.model = None

        print(
            "XTTS Zero-Shot Cloner initialized."
        )

        print(
            f"Configured device: {DEVICE}"
        )


    def is_model_loaded(self):
        """
        Check whether the XTTS model is loaded.

        Returns
        -------
        bool
            True if the model is loaded.
        """

        return self.model is not None


    def load_model(self):
        """
        Load XTTS-v2 into memory.

        The model is loaded only once.
        """

        if self.model is not None:

            print(
                "XTTS-v2 model is already loaded."
            )

            return


        print(
            "Loading XTTS-v2 model..."
        )

        print(
            f"Model: {XTTS_MODEL_NAME}"
        )

        print(
            f"Device: {DEVICE}"
        )


        self.model = TTS(
            model_name=XTTS_MODEL_NAME,
            progress_bar=False
        )


        self.model = self.model.to(
            DEVICE
        )


        print(
            "XTTS-v2 model loaded successfully."
        )


    def generate(
        self,
        text,
        reference_audio_path,
        output_path,
        language=DEFAULT_LANGUAGE
    ):
        """
        Generate speech using XTTS-v2 zero-shot
        voice cloning.

        Parameters
        ----------
        text : str
            Text to convert into speech.

        reference_audio_path : str or Path
            Audio file containing the target voice.

        output_path : str or Path
            Path where generated audio will be saved.

        language : str
            Language code for the generated speech.

        Returns
        -------
        Path
            Path to the generated audio file.
        """

        text = validate_text(
            text
        )


        reference_audio_path = Path(
            reference_audio_path
        )


        output_path = Path(
            output_path
        )


        if not reference_audio_path.exists():

            raise FileNotFoundError(
                "Reference audio was not found: "
                f"{reference_audio_path}"
            )


        ensure_directory(
            output_path.parent
        )


        self.load_model()


        print(
            "Starting XTTS-v2 generation..."
        )

        print(
            f"Reference audio: "
            f"{reference_audio_path}"
        )

        print(
            f"Language: {language}"
        )

        print(
            f"Output path: {output_path}"
        )


        self.model.tts_to_file(
            text=text,
            speaker_wav=str(
                reference_audio_path
            ),
            language=language,
            file_path=str(
                output_path
            ),
            split_sentences=SPLIT_SENTENCES
        )


        if not output_path.exists():

            raise RuntimeError(
                "XTTS generation completed, "
                "but the output file was not created."
            )


        print(
            "XTTS-v2 generation completed."
        )

        return output_path