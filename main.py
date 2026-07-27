#!/usr/bin/env python3
"""Main entry point for the CARE DOLL voice input system."""

from Voice_Input.pipeline import run_pipeline


def main() -> None:
    """Start the CARE DOLL voice input pipeline."""
    audio_path, text_path, result = run_pipeline()
    return result


if __name__ == "__main__":
    main()