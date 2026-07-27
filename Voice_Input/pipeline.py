#!/usr/bin/env python3
"""
Orchestrate the CARE DOLL voice input workflow.

Pipeline:

    Microphone
        ↓
    Audio Recording
        ↓
    Speaker Recognition
        ↓
    Speech-to-Text
        ↓
    Transcription

Current responsibilities:

    1. Record audio
    2. Identify the speaker
    3. Transcribe the audio
    4. Return the generated file paths and speaker result
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# =========================================================
# PATH CONFIGURATION
# =========================================================

CURRENT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    CURRENT_DIR
    .parent
)

for path in (

    PROJECT_ROOT,

    CURRENT_DIR

):

    if str(path) not in sys.path:

        sys.path.insert(

            0,

            str(path)

        )


# =========================================================
# VOICE INPUT MODULES
# =========================================================

from Voice_Input.record_audio import (
    record_audio
)

from Voice_Input.speech_to_text import (
    speech_to_text
)


# =========================================================
# VOICE RECOGNITION MODULE
# =========================================================

from Voice_Recognition.voice_enrollment import (
    VoiceRecognitionPipeline
)


# =========================================================
# GLOBAL VOICE RECOGNITION PIPELINE
# =========================================================

voice_recognition = (
    VoiceRecognitionPipeline()
)


# =========================================================
# MAIN VOICE INPUT PIPELINE
# =========================================================

def run_pipeline() -> tuple[
    Path,
    Path,
    dict[str, Any]
]:
    """
    Record audio, identify the speaker,
    transcribe the audio, and return results.

    Complete flow:

        Microphone
            ↓
        Audio File
            ↓
        Speaker Recognition
            ↓
        Speaker Result
            ↓
        Speech-to-Text
            ↓
        Transcription
    """

    # =====================================================
    # STEP 1: RECORD AUDIO
    # =====================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "STEP 1: RECORDING AUDIO"
    )

    print(
        "========================================"
    )

    audio_path = (

        record_audio()

    )

    print(
        f"\nAudio recorded:"
    )

    print(
        audio_path
    )


    # =====================================================
    # STEP 2: IDENTIFY SPEAKER
    # =====================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "STEP 2: IDENTIFYING SPEAKER"
    )

    print(
        "========================================"
    )

    speaker_result = (

        voice_recognition
        .identify(

            audio_path

        )

    )


    if speaker_result.get(

        "identified",

        False

    ):

        print(
            "\n✓ Known speaker identified"
        )

        print(
            f"User ID: "
            f"{speaker_result['user_id']}"
        )

    else:

        print(
            "\n⚠ Speaker unknown"
        )


    # =====================================================
    # STEP 3: SPEECH-TO-TEXT
    # =====================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "STEP 3: SPEECH-TO-TEXT"
    )

    print(
        "========================================"
    )

    text_path = (

        speech_to_text(

            audio_path

        )

    )


    # =====================================================
    # STEP 4: READ TRANSCRIPTION
    # =====================================================

    transcription = (

        text_path
        .read_text(

            encoding="utf-8"

        )

    )


    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "========================================"
    )

    print(
        "\nSpeaker:"
    )

    if speaker_result.get(

        "identified",

        False

    ):

        print(

            speaker_result[

                "user_id"

            ]

        )

    else:

        print(
            "UNKNOWN"
        )


    print(
        "\nTranscription:"
    )

    print(
        transcription
    )

    print(
        f"\nAudio saved at:"
    )

    print(
        audio_path
    )

    print(
        f"\nText saved at:"
    )

    print(
        text_path
    )


    return (

        audio_path,

        text_path,

        speaker_result

    )


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    run_pipeline()