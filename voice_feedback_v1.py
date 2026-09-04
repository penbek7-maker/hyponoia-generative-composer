"""Small local speech layer for Hyponoia feedback on macOS.

The system voice reads the active question. Microphone audio is captured at
16 kHz and transcribed locally with multilingual Whisper. The resulting text
still has to pass through Hyponoia's normal preview and confirmation route.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


TARGET_SAMPLE_RATE = 16_000
MAX_RECORDING_SECONDS = 30
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "local_models" / "whisper"


class VoiceUnavailable(RuntimeError):
    """Raised when local speech input or output cannot be used safely."""


SUPPORTED_UI_LANGUAGES = {"el", "en"}


def feedback_question(dream_level: str, language: str = "el") -> str:
    level = str(dream_level).strip().upper()
    if level not in {"D1", "D3", "D5"}:
        raise ValueError("dream_level must be D1, D3 or D5")
    if language not in SUPPORTED_UI_LANGUAGES:
        raise ValueError("language must be el or en")
    if language == "en":
        return f"You listened to composition {level}. What would you like to keep and what would you like to change?"
    return f"Άκουσες τη σύνθεση {level}. Τι θα ήθελες να διατηρηθεί και τι να αλλάξει;"


def speak_question(
    dream_level: str,
    *,
    language: str = "el",
    voice: str | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
) -> Any:
    """Read the question locally with an installed Greek or English voice."""
    selected_voice = voice or ("Melina" if language == "el" else "Samantha")
    try:
        return popen(
            ["/usr/bin/say", "-v", selected_voice, feedback_question(dream_level, language)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise VoiceUnavailable("Δεν μπόρεσα να χρησιμοποιήσω την τοπική φωνή του macOS.") from exc


class VoiceRecorder:
    """Push-to-talk mono recorder with no hidden background capture."""

    def __init__(self, *, sample_rate: int = TARGET_SAMPLE_RATE, sounddevice_module: Any = None) -> None:
        if sounddevice_module is None:
            try:
                import sounddevice as sounddevice_module
            except ImportError as exc:
                raise VoiceUnavailable("Δεν έχει εγκατασταθεί ακόμη η τοπική εγγραφή φωνής.") from exc
        self.sample_rate = int(sample_rate)
        self._sounddevice = sounddevice_module
        self._stream: Any = None
        self._chunks: list[np.ndarray] = []
        self._captured_frames = 0

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        if status:
            # Non-fatal overflow/underflow information is intentionally not
            # printed from the real-time audio callback.
            pass
        remaining = MAX_RECORDING_SECONDS * self.sample_rate - self._captured_frames
        if remaining <= 0:
            return
        chunk = np.asarray(indata[:remaining, 0], dtype=np.float32).copy()
        self._chunks.append(chunk)
        self._captured_frames += len(chunk)

    def start(self) -> None:
        if self.is_recording:
            raise VoiceUnavailable("Η εγγραφή έχει ήδη ξεκινήσει.")
        self._chunks = []
        self._captured_frames = 0
        try:
            self._stream = self._sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise VoiceUnavailable(
                "Δεν άνοιξε το μικρόφωνο. Έλεγξε την άδεια μικροφώνου για το Terminal/Python."
            ) from exc

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            raise VoiceUnavailable("Δεν υπάρχει ενεργή εγγραφή.")
        stream = self._stream
        self._stream = None
        try:
            stream.stop()
            stream.close()
        finally:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
            self._chunks = []
        if len(audio) < int(0.35 * self.sample_rate):
            raise VoiceUnavailable("Η εγγραφή ήταν πολύ σύντομη. Μίλησε για τουλάχιστον ένα δευτερόλεπτο.")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def cancel(self) -> None:
        if not self.is_recording:
            return
        stream = self._stream
        self._stream = None
        self._chunks = []
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


class LocalWhisperTranscriber:
    """Lazy multilingual Whisper transcription; no audio leaves the computer."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_WHISPER_MODEL,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        whisper_module: Any = None,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self._whisper = whisper_module
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._whisper is None:
            try:
                import whisper
            except ImportError as exc:
                raise VoiceUnavailable("Δεν έχει εγκατασταθεί ακόμη το τοπικό Whisper.") from exc
            self._whisper = whisper
        self.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = self._whisper.load_model(
                self.model_name,
                device="cpu",
                download_root=str(self.model_dir),
            )
        except Exception as exc:
            raise VoiceUnavailable("Δεν μπόρεσα να φορτώσω το τοπικό μοντέλο φωνής.") from exc
        return self._model

    def transcribe(self, audio: np.ndarray) -> dict[str, str]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(samples) < int(0.35 * TARGET_SAMPLE_RATE):
            raise VoiceUnavailable("Η εγγραφή δεν περιέχει αρκετή φωνή.")
        try:
            result = self._load().transcribe(
                samples,
                task="transcribe",
                language=None,
                fp16=False,
                temperature=0,
                verbose=False,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise VoiceUnavailable("Η τοπική μεταγραφή δεν ολοκληρώθηκε.") from exc
        text = str(result.get("text", "")).strip()
        if not text:
            raise VoiceUnavailable("Δεν αναγνωρίστηκε ομιλία. Δοκίμασε ξανά λίγο πιο κοντά στο μικρόφωνο.")
        language = str(result.get("language", "und")).strip() or "und"
        return {"text": text, "locale": language}
