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
# VOICE CLONING MODULE
# =========================================================

from Voice_Cloning.pipeline import (
    VoiceCloningPipeline
)


PREDEFINED_VOICE_CLONING_TEXT = (
    "Hello. This is a test of the CARE Doll personalized "
    "voice cloning system."
)


# Initialize once.
# XTTS itself remains lazily loaded inside the existing cloner.
voice_cloning_pipeline = (
    VoiceCloningPipeline()
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

    ) and not speaker_result.get("newly_enrolled", False):

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


    incremental_enrollment = speaker_result.get("embedding_added", False)

    if speaker_result.get("newly_enrolled", False):

        user_id = speaker_result.get("user_id")

        ref_count = speaker_result.get("reference_count", 1)

        print(

            "\nNo existing speaker matched."

        )

        print(

            "Creating a new speaker profile..."

        )

        print(

            "\nNew speaker enrolled successfully."

        )

        print(

            f"\nUser ID: {user_id}"

        )

        print(

            f"Stored reference embeddings: {ref_count}"

        )

        print(

            f"\nCurrent audio assigned to:\n{user_id}"

        )

    elif speaker_result.get("embedding_added", False):

        user_id = speaker_result.get("user_id")

        ref_count = speaker_result.get("reference_count")

        print(

            f"\n[OK] Incremental voice enrollment updated for {user_id} "

            f"(references: {ref_count}/5)"

        )


    if speaker_audio_path:

        print(

            "\n[OK] Audio saved to speaker dataset:"

        )

        print(

            speaker_audio_path

        )

    else:

        print(

            "\nSpeaker unknown."

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
    # STEP 5: VOICE CLONING (PERSONALIZED VOICE GENERATION)
    # =====================================================

    print(

        "\n"
        "========================================"

    )

    print(

        "STEP 5: VOICE CLONING"

    )

    print(

        "========================================"

    )


    generated_audio_path = None

    if speaker_result.get("identified") and speaker_result.get("user_id"):

        user_id = speaker_result.get("user_id")

        print(

            "\nStarting personalized voice generation..."

        )

        print(

            f"Recognized user ID: {user_id}"

        )

        print(

            "Predefined text: "
            f"{PREDEFINED_VOICE_CLONING_TEXT}"

        )

        try:

            generated_audio_path = (

                voice_cloning_pipeline.generate_for_identified_user(

                    user_id=user_id,

                    text=PREDEFINED_VOICE_CLONING_TEXT,

                    language="en",

                )

            )

            print(

                "Personalized voice generated successfully."

            )

            print(

                f"Generated audio: {generated_audio_path}"

            )

        except Exception as error:

            print(

                f"Voice cloning failed for user "
                f"'{user_id}': {error}"

            )

    else:

        print(

            "Voice cloning skipped because no user was "
            "successfully identified."

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


    user_id = speaker_result.get("user_id")

    if user_id:

        print(user_id)

    else:

        print("UNKNOWN")


    if speaker_result.get("newly_enrolled", False):

        print(

            "\nSpeaker Status:\nNEWLY ENROLLED"

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


    if generated_audio_path:

        print(

            "\nGenerated personalized voice saved at:"

        )

        print(

            generated_audio_path

        )


    pipeline_result = {

        "status": "newly_enrolled" if speaker_result.get("newly_enrolled") else ("identified" if speaker_result.get("identified") else "unknown"),


        "user_id": speaker_result.get("user_id"),

        "confidence": speaker_result.get("confidence", 0.0),

        "source_audio": str(audio_path),

        "profile_audio": str(speaker_audio_path) if speaker_audio_path else None,

        "text_path": str(text_path),

        "transcription": transcription,

        "incremental_enrollment": incremental_enrollment,

        "speaker_result": speaker_result,

        "generated_audio_path": generated_audio_path

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