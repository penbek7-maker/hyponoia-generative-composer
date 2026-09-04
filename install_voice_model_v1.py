"""Download/load the small local Whisper model once before voice testing."""

from voice_feedback_v1 import LocalWhisperTranscriber


def main() -> None:
    transcriber = LocalWhisperTranscriber()
    transcriber._load()
    print(f"Local voice model ready: {transcriber.model_name}")
    print(f"Model folder: {transcriber.model_dir}")


if __name__ == "__main__":
    main()
