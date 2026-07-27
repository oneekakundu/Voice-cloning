from pathlib import Path

from Voice_Recognition.speaker_encoder import SpeakerEncoder


AUDIO_FILE = Path(
    "data/audio/recording_20260727_092147.wav"
)


def main():
    print("Starting ECAPA-TDNN encoder test...")

    encoder = SpeakerEncoder()

    embedding = encoder.encode_and_normalize(
        AUDIO_FILE
    )

    print("\nEncoder test successful!")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {embedding.shape[0]}")
    print(f"Embedding device: {embedding.device}")


if __name__ == "__main__":
    main()