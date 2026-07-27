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
    Save Audio to Speaker Dataset
        ↓
    Speech-to-Text
        ↓
    Transcription

Responsibilities:

    1. Record audio
    2. Identify the speaker
    3. Display similarity diagnostics
    4. Save original audio under the identified speaker
    5. Transcribe the same audio
    6. Return all results
"""

from __future__ import annotations

import sys
import shutil

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
# DATASET STORAGE
# =========================================================

VOICE_PROFILE_DIRECTORY = (

    PROJECT_ROOT
    / "data"
    / "voice_profiles"

)


def save_audio_to_speaker_profile(

    audio_path: Path,

    speaker_result: dict[str, Any]

) -> Path | None:
    """
    Save the original recorded WAV file
    inside the identified speaker's audio dataset.

    Example:

        data/voice_profiles/user_001/audio/
            recording_20260727_220522.wav

    The original audio file is copied.

    It is NOT converted into an embedding
    for dataset storage.

    Returns:

        Destination path if speaker identified.

        None if speaker unknown.
    """

    # -----------------------------------------
    # CHECK IDENTIFICATION RESULT
    # -----------------------------------------

    if not speaker_result.get(

        "identified",

        False

    ):

        return None


    user_id = (

        speaker_result.get(

            "user_id"

        )

    )


    if not user_id:

        return None


    # -----------------------------------------
    # SAVE AUDIO TO SPEAKER PROFILE
    # -----------------------------------------

    return (

        voice_recognition
        .profile_manager
        .save_audio_recording(

            user_id=user_id,

            audio_path=audio_path

        )

    )


# =========================================================
# DISPLAY SPEAKER SIMILARITY REPORT
# =========================================================

def display_similarity_report(

    speaker_result: dict[str, Any]

) -> None:
    """
    Display the complete speaker identification
    and similarity diagnostics.

    This function does not change
    identification logic.

    It only displays the information
    already returned by SpeakerIdentifier.
    """

    print(

        "\n"
        "========================================"

    )

    print(

        "VOICE IDENTIFICATION REPORT"

    )

    print(

        "========================================"

    )


    print(

        f"\nIdentified: "
        f"{speaker_result.get('identified', False)}"

    )


    if speaker_result.get(

        "user_id"

    ):

        print(

            f"User ID: "
            f"{speaker_result['user_id']}"

        )


    # -----------------------------------------
    # PROFILE RESULTS
    # -----------------------------------------

    profile_results = (

        speaker_result.get(

            "profile_results",

            []

        )

    )


    if not profile_results:

        print(

            "\nNo profile comparison results "
            "were returned."

        )

        return


    print(

        "\nProfile Comparison Results:"

    )


    for profile_result in profile_results:

        print(

            "\n----------------------------------------"

        )


        print(

            f"User ID: "
            f"{profile_result.get('user_id')}"

        )


        print(

            f"Is Match: "
            f"{profile_result.get('is_match')}"

        )


        print(

            f"Match Count: "
            f"{profile_result.get('match_count')}"

        )


        print(

            f"Required Matches: "
            f"{profile_result.get('required_matches')}"

        )


        print(

            f"Average Similarity: "
            f"{profile_result.get('average_similarity', 0.0):.4f}"

        )


        print(

            f"Maximum Similarity: "
            f"{profile_result.get('maximum_similarity', 0.0):.4f}"

        )


        similarities = (

            profile_result.get(

                "similarities",

                []

            )

        )


        print(

            f"Per-Reference Similarities: "
            f"{similarities}"

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
    Execute the complete voice input pipeline.

    Complete flow:

        Microphone
            ↓
        Record Audio
            ↓
        Current WAV File
            ↓
        Generate Temporary Embedding
            ↓
        Compare With Stored References
            ↓
        Identify Speaker
            ↓
        Save Original WAV to Speaker Dataset
            ↓
        Speech-to-Text
            ↓
        Final Result
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

        "\nAudio recorded:"

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


    # -----------------------------------------
    # DISPLAY COMPLETE SIMILARITY INFORMATION
    # -----------------------------------------

    display_similarity_report(

        speaker_result

    )


    # =====================================================
    # STEP 3: SAVE ORIGINAL AUDIO
    # =====================================================

    print(

        "\n"
        "========================================"

    )

    print(

        "STEP 3: SAVING SPEAKER AUDIO"

    )

    print(

        "========================================"

    )


    speaker_audio_path = (

        save_audio_to_speaker_profile(

            audio_path=audio_path,

            speaker_result=speaker_result

        )

    )


    incremental_enrollment = False

    if speaker_result.get("identified", False):

        user_id = speaker_result.get("user_id")

        if user_id and not voice_recognition.profile_manager.is_enrollment_complete(user_id):

            enrollment_res = voice_recognition.enroll(

                audio_path=str(audio_path),

                user_id=user_id

            )

            incremental_enrollment = enrollment_res.get("success", False)

            if incremental_enrollment:

                print(

                    f"✓ Incremental voice enrollment updated for {user_id} "

                    f"(references: {enrollment_res.get('reference_count')}/5)"

                )


    if speaker_audio_path:

        print(

            "\n✓ Audio saved to speaker dataset:"

        )

        print(

            speaker_audio_path

        )

    else:

        print(

            "\n⚠ Speaker unknown."

        )

        print(

            "Audio remains in the general audio directory."

        )


    # =====================================================
    # STEP 4: SPEECH-TO-TEXT
    # =====================================================

    print(

        "\n"
        "========================================"

    )

    print(

        "STEP 4: SPEECH-TO-TEXT"

    )

    print(

        "========================================"

    )


    text_path = (

        speech_to_text(

            audio_path

        )

    )


    transcription = (

        text_path
        .read_text(

            encoding="utf-8"

        )

    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

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

            speaker_result.get(

                "user_id",

                "UNKNOWN"

            )

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

        "\nOriginal audio saved at:"

    )

    print(

        audio_path

    )


    if speaker_audio_path:

        print(

            "\nSpeaker dataset audio saved at:"

        )

        print(

            speaker_audio_path

        )


    print(

        "\nText saved at:"

    )

    print(

        text_path

    )


    pipeline_result = {

        "status": "identified" if speaker_result.get("identified") else "unknown",

        "user_id": speaker_result.get("user_id"),

        "confidence": speaker_result.get("confidence", 0.0),

        "source_audio": str(audio_path),

        "profile_audio": str(speaker_audio_path) if speaker_audio_path else None,

        "text_path": str(text_path),

        "transcription": transcription,

        "incremental_enrollment": incremental_enrollment,

        "speaker_result": speaker_result

    }


    return (

        audio_path,

        text_path,

        pipeline_result

    )


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    run_pipeline()