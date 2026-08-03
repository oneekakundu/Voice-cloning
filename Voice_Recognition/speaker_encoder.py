"""
ECAPA-TDNN Speaker Encoder

Responsibilities:
    Audio file → Preprocessed audio → Speaker embedding

This module does NOT:
    - Record audio
    - Enroll speakers
    - Identify speakers
    - Compare embeddings
    - Manage voice profiles
"""

from pathlib import Path
from typing import Union, Tuple

import torch
from speechbrain.inference.speaker import EncoderClassifier

from .config import SPEAKER_MODEL_SOURCE, DEVICE
from .speaker_preprocessing import SpeakerAudioPreprocessor


class SpeakerEncoder:
    """
    ECAPA-TDNN speaker encoder.

    Converts an audio file into a speaker embedding that can later
    be used for speaker verification and identification.
    """

    def __init__(
        self,
        model_source: str = SPEAKER_MODEL_SOURCE,
        device: str = DEVICE,
    ):
        """
        Initialize the ECAPA-TDNN speaker encoder.

        Args:
            model_source: Pretrained SpeechBrain model.
            device: "cuda:0" or "cpu".
        """

        self.device = device

        print("Loading ECAPA-TDNN speaker encoder...")
        print(f"Device: {self.device}")

        self.model = EncoderClassifier.from_hparams(
            source=model_source,
            run_opts={"device": self.device},
        )

        self.preprocessor = SpeakerAudioPreprocessor()

        print(
            "ECAPA-TDNN speaker encoder "
            "loaded successfully."
        )

    def encode_with_duration(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[torch.Tensor, float]:
        """
        Convert an audio file into a speaker embedding and return its usable speech duration in seconds.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        signal = self.preprocessor.process(
            audio_path,
            trim_silence=True,
            normalize=False,
        )

        duration = float(
            signal.shape[1] / self.preprocessor.TARGET_SAMPLE_RATE
        )

        signal = signal.to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_batch(
                signal
            )

        embedding = embedding.squeeze()

        return embedding, duration

    def encode(
        self,
        audio_path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Convert an audio file into a speaker embedding.

        Pipeline:
            Audio file
                ↓
            Audio preprocessing
                ↓
            ECAPA-TDNN
                ↓
            Speaker embedding

        Args:
            audio_path: Path to a WAV/audio file.

        Returns:
            torch.Tensor:
                1-dimensional speaker embedding.
        """

        embedding, _ = self.encode_with_duration(audio_path)
        return embedding

    def encode_and_normalize(
        self,
        audio_path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Generate an L2-normalized speaker embedding.
        """

        embedding = self.encode(audio_path)

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=0,
        )

        return embedding

    def get_embedding_dimension(
        self,
        audio_path: Union[str, Path],
    ) -> int:
        """
        Return the dimensionality of the generated embedding.
        """

        embedding = self.encode_and_normalize(
            audio_path
        )

        return embedding.shape[0]