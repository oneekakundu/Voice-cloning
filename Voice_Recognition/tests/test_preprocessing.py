from ..speaker_preprocessing import SpeakerAudioPreprocessor


def main():

    audio_path = (
        "data/audio/recording_20260727_092147.wav"
    )

    preprocessor = SpeakerAudioPreprocessor()

    waveform = preprocessor.process(
        audio_path
    )

    print("\nFinal waveform information:")
    print(f"Shape: {waveform.shape}")
    print(f"Data type: {waveform.dtype}")
    print(
        f"Final sample rate: "
        f"{preprocessor.TARGET_SAMPLE_RATE} Hz"
    )


if __name__ == "__main__":
    main()