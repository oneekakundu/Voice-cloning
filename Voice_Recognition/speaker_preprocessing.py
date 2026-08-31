"""
Speaker Audio Preprocessing

Responsibilities:
    - Load audio files
    - Convert audio to mono
    - Resample audio to 16 kHz
    - Convert audio to float32
    - Optionally trim leading/trailing silence
    - Optionally normalize amplitude
    - Validate audio before speaker embedding extraction

Output:
    Tensor shape: [1, num_samples]
    Sample rate: 16,000 Hz
    Data type: float32
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio


class SpeakerAudioPreprocessor:
    """
    Standardizes audio before speaker embedding extraction.

    Final output:
        - Mono audio
        - 16 kHz sample rate
        - float32 tensor
        - Shape: [1, num_samples]
    """

    TARGET_SAMPLE_RATE = 16000
    MINIMUM_DURATION_SECONDS = 3.0

    def load_audio(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file using SoundFile.
        SoundFile returns:
            Mono:
                [samples]
            Stereo:
                [samples, channels]
        This method converts the result to:
            [channels, samples]
        Returns:
            waveform:
                torch.Tensor with shape [channels, samples]
            sample_rate:
                Original sample rate
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # Load audio as float32
        audio_data, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
        )

        waveform = torch.from_numpy(
            audio_data
        )
        # Mono audio:
        # SoundFile shape:
        # [samples]
        # Convert to:
        # [1, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Multi-channel audio:
        # SoundFile shape:
        # [samples, channels]
        # Convert to:
        # [channels, samples]
        else:
            waveform = waveform.transpose(
                0,
                1,
            )

        return waveform, sample_rate

    def convert_to_mono(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert multi-channel audio to mono.

        Input:
            [channels, samples]

        Output:
            [1, samples]
        """

        if waveform.shape[0] > 1:
            waveform = torch.mean(
                waveform,
                dim=0,
                keepdim=True,
            )

        return waveform

    def resample_audio(
        self,
        waveform: torch.Tensor,
        original_sample_rate: int,
    ) -> torch.Tensor:
        """
        Resample audio to 16 kHz.
        """

        if (
            original_sample_rate
            == self.TARGET_SAMPLE_RATE
        ):
            return waveform

        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=self.TARGET_SAMPLE_RATE,
        )

        waveform = resampler(waveform)

        return waveform

    def normalize_audio(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize audio amplitude safely.

        This is optional and is disabled by default.
        """

        max_amplitude = torch.max(
            torch.abs(waveform)
        )

        if max_amplitude > 0:
            waveform = (
                waveform / max_amplitude
            )

        return waveform

    def remove_silence(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """
        Remove very quiet leading and trailing sections.

        This is basic amplitude-based silence trimming.

        It does NOT perform:
            - Noise cancellation
            - Background noise removal
            - Speech enhancement
            - VAD-based speech detection
        """

        # [1, samples] → [samples]
        audio = waveform.squeeze(0)
        amplitude = torch.abs(audio)
        active_samples = (
            amplitude > threshold
        )

        # If the entire recording is below
        # the threshold, return original audio
        if not torch.any(active_samples):
            return waveform

        active_indices = torch.where(
            active_samples
        )[0]

        start = active_indices[0]
        end = active_indices[-1] + 1

        trimmed_audio = audio[start:end]

        # [samples] → [1, samples]
        return trimmed_audio.unsqueeze(0)

    def validate_audio(
        self,
        waveform: torch.Tensor,
    ) -> None:
        """
        Validate processed audio.
        """

        # Must be:
        # [channels, samples]
        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform shape "
                "[1, samples], "
                f"got {waveform.shape}"
            )

        # Must be mono
        if waveform.shape[0] != 1:
            raise ValueError(
                "Audio must be mono."
            )

        # Must contain samples
        if waveform.shape[1] == 0:
            raise ValueError(
                "Audio contains no samples."
            )

        duration = (
            waveform.shape[1]
            / self.TARGET_SAMPLE_RATE
        )

        if (
            duration
            < self.MINIMUM_DURATION_SECONDS
        ):
            raise ValueError(
                f"Audio is too short: "
                f"{duration:.2f} seconds. "
                f"Minimum required: "
                f"{self.MINIMUM_DURATION_SECONDS:.1f} "
                f"seconds."
            )

    def high_pass_filter(
        self,
        waveform: torch.Tensor,
        cutoff_hz: float = 70.0,
        order: int = 2,
    ) -> torch.Tensor:
        """
        Apply a gentle 2nd-order Butterworth high-pass filter.
        Removes sub-audible DC offset, handling noise, and microphone rumble
        below human vocal fundamentals (F0) without altering speech characteristics.
        """
        audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        nyquist = 0.5 * self.TARGET_SAMPLE_RATE
        normalized_cutoff = cutoff_hz / nyquist

        sos = scipy.signal.butter(
            order,
            normalized_cutoff,
            btype="highpass",
            output="sos",
        )
        filtered = scipy.signal.sosfilt(sos, audio_np)

        return torch.from_numpy(filtered).unsqueeze(0).float()

    def process(
        self,
        audio_path: Union[str, Path],
        trim_silence: bool = True,
        normalize: bool = False,
        high_pass: bool = True,
        cutoff_hz: float = 70.0,
    ) -> torch.Tensor:
        """
        Complete audio preprocessing pipeline.

        Pipeline:
            Audio file
                ↓
            Load with SoundFile
                ↓
            Convert to mono
                ↓
            Resample to 16 kHz
                ↓
            Convert to float32
                ↓
            Gentle High-Pass Filter (70 Hz)
                ↓
            Optional silence trimming
                ↓
            Optional normalization
                ↓
            Validate
                ↓
            Return processed waveform

        Returns:
            torch.Tensor:
                Shape: [1, num_samples]

            Sample rate:
                16,000 Hz

            Data type:
                float32
        """

        print(
            f"\nProcessing audio: "
            f"{audio_path}"
        )

        # --------------------------------------------------
        # 1. Load audio
        # --------------------------------------------------

        waveform, original_sample_rate = (
            self.load_audio(audio_path)
        )

        print(
            f"Original sample rate: "
            f"{original_sample_rate} Hz"
        )

        print(
            f"Original channels: "
            f"{waveform.shape[0]}"
        )

        # --------------------------------------------------
        # 2. Convert to mono
        # --------------------------------------------------

        waveform = self.convert_to_mono(
            waveform
        )

        # --------------------------------------------------
        # 3. Resample to 16 kHz
        # --------------------------------------------------

        waveform = self.resample_audio(
            waveform,
            original_sample_rate,
        )

        # --------------------------------------------------
        # 4. Ensure float32
        # --------------------------------------------------

        waveform = waveform.float()

        # --------------------------------------------------
        # 5. Gentle High-Pass Filtering (70 Hz)
        # --------------------------------------------------

        if high_pass:
            waveform = self.high_pass_filter(
                waveform,
                cutoff_hz=cutoff_hz,
            )

        # --------------------------------------------------
        # 6. Optional silence trimming
        # --------------------------------------------------

        if trim_silence:
            waveform = self.remove_silence(
                waveform
            )

        # --------------------------------------------------
        # 7. Optional amplitude normalization
        # --------------------------------------------------

        if normalize:
            waveform = self.normalize_audio(
                waveform
            )

        # --------------------------------------------------
        # 8. Validate final audio
        # --------------------------------------------------

        self.validate_audio(waveform)

        duration = (
            waveform.shape[1]
            / self.TARGET_SAMPLE_RATE
        )

        print(
            f"Processed sample rate: "
            f"{self.TARGET_SAMPLE_RATE} Hz"
        )

        print(
            "Processed channels: 1"
        )

        print(
            f"Processed duration: "
            f"{duration:.2f} seconds"
        )

        print(
            "Audio preprocessing completed."
        )

        return waveform

    def get_usable_speech_duration(
        self,
        audio_path: Union[str, Path],
    ) -> float:
        """
        Calculate usable speech duration in seconds for an audio file
        after preprocessing and validation.
        """
        waveform = self.process(audio_path)
        return float(waveform.shape[1] / self.TARGET_SAMPLE_RATE)