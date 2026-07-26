"""
ECAPA-TDNN Speaker Encoder

Responsibilities:
    Audio file → Speaker embedding

This module does NOT:
    - Record audio
    - Enroll speakers
    - Identify speakers
    - Compare embeddings
    - Manage voice profiles

Those responsibilities belong to other modules.
"""

from pathlib import Path
from typing import Union

import torch
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier

from .config import SPEAKER_MODEL_SOURCE, DEVICE


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

        print("ECAPA-TDNN speaker encoder loaded successfully.")

    def encode(
        self,
        audio_path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Convert an audio file into a speaker embedding.

        Args:
            audio_path: Path to a WAV/audio file.

        Returns:
            torch.Tensor:
                1-dimensional speaker embedding.
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # Load audio
        audio_data, sample_rate = sf.read(str(audio_path))

        print(f"Audio sample rate: {sample_rate} Hz")

        # Convert NumPy array to PyTorch tensor
        signal = torch.from_numpy(audio_data).float()

        # Convert stereo audio to mono
        if signal.ndim == 2:
            signal = signal.mean(dim=1)

        # Move audio to selected device
        signal = signal.to(self.device)

        # Generate speaker embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(signal)

        # Convert [1, 1, embedding_dim] → [embedding_dim]
        embedding = embedding.squeeze()

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

        embedding = self.encode_and_normalize(audio_path)

        return embedding.shape[0]