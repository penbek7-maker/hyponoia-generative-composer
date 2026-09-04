import numpy as np
import pytest

from feedback_input_v1 import build_feedback_preview
from voice_feedback_v1 import (
    LocalWhisperTranscriber,
    VoiceRecorder,
    VoiceUnavailable,
    feedback_question,
    speak_question,
)


def test_question_is_level_specific_and_read_with_local_greek_voice():
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return object()

    speak_question("D5", popen=fake_popen)
    assert "D5" in feedback_question("D5")
    assert calls[0][0][:4] == ["/usr/bin/say", "-v", "Melina", feedback_question("D5")]


def test_question_and_local_voice_can_switch_to_english():
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return object()

    speak_question("D3", language="en", popen=fake_popen)
    question = feedback_question("D3", "en")
    assert question.startswith("You listened to composition D3")
    assert calls[0][0] == ["/usr/bin/say", "-v", "Samantha", question]


def test_question_rejects_unknown_level():
    with pytest.raises(ValueError, match="D1, D3 or D5"):
        feedback_question("D2")


def test_question_rejects_unknown_interface_language():
    with pytest.raises(ValueError, match="el or en"):
        feedback_question("D1", "fr")


class FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        self.callback(np.ones((16_000, 1), dtype=np.float32) * 0.2, 16_000, None, None)

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self):
        self.stream = None

    def InputStream(self, **kwargs):
        assert kwargs["samplerate"] == 16_000
        assert kwargs["channels"] == 1
        self.stream = FakeStream(kwargs["callback"])
        return self.stream


def test_push_to_talk_records_only_between_explicit_start_and_stop():
    backend = FakeSoundDevice()
    recorder = VoiceRecorder(sounddevice_module=backend)
    assert recorder.is_recording is False
    recorder.start()
    assert recorder.is_recording is True
    audio = recorder.stop()
    assert recorder.is_recording is False
    assert audio.shape == (16_000,)
    assert backend.stream.closed is True


class FakeModel:
    def transcribe(self, audio, **kwargs):
        assert audio.dtype == np.float32
        assert kwargs["language"] is None
        assert kwargs["fp16"] is False
        return {"text": " Περισσότερο synth και ομαλότερες μεταβάσεις. ", "language": "el"}


class FakeWhisper:
    def load_model(self, model_name, **kwargs):
        assert model_name == "base"
        assert kwargs["device"] == "cpu"
        return FakeModel()


def test_local_whisper_transcript_uses_same_safe_voice_feedback_route(tmp_path):
    transcriber = LocalWhisperTranscriber(model_dir=tmp_path, whisper_module=FakeWhisper())
    result = transcriber.transcribe(np.ones(16_000, dtype=np.float32) * 0.1)
    preview = build_feedback_preview(
        result["text"],
        dream_level="D3",
        source="voice",
        locale=result["locale"],
    )
    assert result["locale"] == "el"
    assert preview["source"] == "voice"
    assert preview["target_levels"] == ["D3"]
    assert {action["intent"] for action in preview["actions"]} >= {
        "increase_synthetic_material",
        "increase_smoothness",
    }


def test_too_short_audio_is_rejected_without_transcription(tmp_path):
    transcriber = LocalWhisperTranscriber(model_dir=tmp_path, whisper_module=FakeWhisper())
    with pytest.raises(VoiceUnavailable, match="αρκετή φωνή"):
        transcriber.transcribe(np.zeros(100, dtype=np.float32))
