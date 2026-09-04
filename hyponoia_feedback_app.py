"""Non-technical preview-and-confirm window for Hyponoia free feedback."""

from __future__ import annotations

import copy
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from feedback_input_v1 import apply_feedback_preview, build_feedback_preview
from human_feedback_v1 import DEFAULT_LEARNING_PROFILE
from voice_feedback_v1 import (
    LocalWhisperTranscriber,
    VoiceRecorder,
    VoiceUnavailable,
    feedback_question,
    speak_question,
)


PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = PROJECT_DIR / "learning_profile.json"
EVIDENCE_DIR = PROJECT_DIR / "human_feedback"


LANGUAGE_NAMES = {"Ελληνικά": "el", "English": "en"}

UI_TEXT = {
    "el": {
        "intro": "Το Hyponoia θα σου δείξει πρώτα τι κατάλαβε. Τίποτα δεν αλλάζει πριν πατήσεις ‘Εφάρμοσε’.",
        "language": "Γλώσσα / Language:",
        "level": "Ποια σύνθεση άκουσες;",
        "listen": "🔊 Άκουσε την ερώτηση",
        "start_voice": "🎙 Έναρξη φωνής",
        "stop_voice": "■ Σταμάτα και γράψε",
        "transcribing": "Μεταγραφή…",
        "comment": "Ελεύθερο σχόλιο:",
        "preview": "1. Δείξε μου τι κατάλαβες",
        "apply": "2. Εφάρμοσε το feedback",
        "initial": "Γράψε ή πες ελεύθερα τι θέλεις να αλλάξει.",
        "changed": "Το σχόλιο άλλαξε. Δες ξανά τι κατάλαβε το Hyponoia.",
        "recording": "Σε ακούω. Μίλησε φυσικά και μετά πάτησε ‘Σταμάτα και γράψε’.",
        "transcribing_status": "Η φωνή μεταγράφεται τοπικά. Την πρώτη φορά μπορεί να αργήσει λίγο.",
        "voice_ready": "Η φωνή μεταγράφηκε. Έλεγξε το κείμενο και δες τι κατάλαβα.",
        "voice_failed": "Δεν άλλαξε τίποτα. Μπορείς να ξαναδοκιμάσεις ή να γράψεις το σχόλιο.",
        "ready": "Έτοιμο για έλεγχο. Αν συμφωνείς, πάτησε ‘Εφάρμοσε το feedback’.",
        "no_change": "Δεν έγινε καμία αλλαγή. Δοκίμασε πιο συγκεκριμένη διατύπωση.",
        "saved": "Το feedback αποθηκεύτηκε για {levels}. Θα επηρεάσει την επόμενη σύνθεση.",
    },
    "en": {
        "intro": "Hyponoia will first show you what it understood. Nothing changes until you press ‘Apply’.",
        "language": "Language / Γλώσσα:",
        "level": "Which composition did you listen to?",
        "listen": "🔊 Listen to the question",
        "start_voice": "🎙 Start voice input",
        "stop_voice": "■ Stop and transcribe",
        "transcribing": "Transcribing…",
        "comment": "Free comment:",
        "preview": "1. Show me what you understood",
        "apply": "2. Apply feedback",
        "initial": "Write or say freely what you would like to change.",
        "changed": "The comment changed. Check again what Hyponoia understood.",
        "recording": "I am listening. Speak naturally, then press ‘Stop and transcribe’.",
        "transcribing_status": "Your voice is being transcribed locally. The first time may take a moment.",
        "voice_ready": "Your voice was transcribed. Check the text, then review what I understood.",
        "voice_failed": "Nothing changed. You can try again or type your comment.",
        "ready": "Ready to review. If this is correct, press ‘Apply feedback’.",
        "no_change": "No change was proposed. Try a more specific description.",
        "saved": "Feedback was saved for {levels}. It will affect the next composition.",
    },
}


def format_preview(preview: dict, language: str = "el") -> str:
    if language not in UI_TEXT:
        raise ValueError("language must be el or en")
    if preview.get("status") == "empty":
        return "Δεν έχει γραφτεί ακόμη σχόλιο." if language == "el" else "No comment has been entered yet."
    if not preview.get("can_apply"):
        if language == "en":
            return "I did not recognise a specific musical change yet. Try e.g. ‘more synth, less repetition and smoother transitions’."
        return "Δεν αναγνώρισα ακόμη συγκεκριμένη μουσική αλλαγή. Δοκίμασε π.χ. ‘περισσότερο synth, λιγότερη επανάληψη και πιο ομαλές μεταβάσεις’."
    targets = ", ".join(preview.get("target_levels", []))
    mode = preview.get("interpreter", "rules")
    if language == "en":
        mode_label = "local language model" if mode == "local_llm" else "safe basic rules"
        lines = [f"Will affect: {targets}", f"Understanding: {mode_label}"]
    else:
        mode_label = "τοπικό γλωσσικό μοντέλο" if mode == "local_llm" else "ασφαλείς βασικοί κανόνες"
        lines = [f"Θα επηρεαστεί: {targets}", f"Κατανόηση: {mode_label}"]
    if language == "en" and preview.get("transcript"):
        # The legacy model contract names its explanatory field summary_el and
        # some local model versions still answer that field in Greek. Keep the
        # English UI consistently English by showing the listener's original
        # comment; the validated intent labels below provide the interpretation.
        lines.extend([f"Comment: {preview['transcript']}"])
    elif preview.get("summary_el"):
        lines.extend([f"Σύνοψη: {preview['summary_el']}"])
    lines.extend(["", "I understood:" if language == "en" else "Κατάλαβα:"])
    label_key = "label_en" if language == "en" else "label_el"
    lines.extend(f"• {action.get(label_key, action['intent'])}" for action in preview.get("actions", []))
    lines.extend(["", "Exact small changes:" if language == "en" else "Ακριβείς μικρές αλλαγές:"])
    for change in preview.get("control_changes", []):
        lines.append(
            f"• {change['target_level']} — {change['control']}: "
            f"{change['old_value']:.2f} → {change['new_value']:.2f}"
        )
    domains = preview.get("composition_influence", {}).get("domains", [])
    if domains:
        lines.extend(["", "Will influence in the composition:" if language == "en" else "Θα επηρεάσει στη σύνθεση:"])
        domain_label_key = "label_en" if language == "en" else "label_el"
        lines.extend(f"• {item.get(domain_label_key, item['domain'])}" for item in domains)
    if preview.get("ambiguities"):
        lines.extend(["", "Points that need attention:" if language == "en" else "Σημεία που χρειάζονται προσοχή:"])
        if language == "en":
            lines.append("• The model reported uncertainty. Check the proposed changes carefully.")
        else:
            lines.extend(f"• {item}" for item in preview["ambiguities"])
    return "\n".join(lines)


class FeedbackApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hyponoia — Feedback")
        self.root.geometry("780x720")
        self.root.minsize(650, 560)
        self.language_name = tk.StringVar(value="Ελληνικά")
        self.level = tk.StringVar(value="D1")
        self.status = tk.StringVar(value=UI_TEXT["el"]["initial"])
        self.source = "text"
        self.locale: str | None = None
        self.preview: dict | None = None
        self.voice_recorder: VoiceRecorder | None = None
        self.voice_transcriber = LocalWhisperTranscriber()
        self.voice_busy = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=24)
        frame.pack(fill="both", expand=True)
        title_row = ttk.Frame(frame)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="Hyponoia Feedback", font=("Helvetica", 24, "bold")).pack(side="left")
        self.language_menu = ttk.Combobox(
            title_row,
            textvariable=self.language_name,
            values=tuple(LANGUAGE_NAMES),
            width=10,
            state="readonly",
        )
        self.language_menu.pack(side="right")
        self.language_label = ttk.Label(title_row, text=UI_TEXT["el"]["language"])
        self.language_label.pack(side="right", padx=(0, 8))
        self.language_menu.bind("<<ComboboxSelected>>", lambda _event: self._language_changed())
        self.intro_label = ttk.Label(
            frame,
            text=UI_TEXT["el"]["intro"],
            wraplength=700,
            justify="left",
        )
        self.intro_label.pack(anchor="w", pady=(8, 18))

        level_row = ttk.Frame(frame)
        level_row.pack(fill="x")
        self.level_label = ttk.Label(level_row, text=UI_TEXT["el"]["level"])
        self.level_label.pack(side="left")
        level_menu = ttk.Combobox(
            level_row,
            textvariable=self.level,
            values=("D1", "D3", "D5"),
            width=6,
            state="readonly",
        )
        level_menu.pack(side="left", padx=(10, 0))
        level_menu.bind("<<ComboboxSelected>>", lambda _event: self._level_changed())

        self.question = tk.StringVar(value=feedback_question(self.level.get()))
        ttk.Label(
            frame,
            textvariable=self.question,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(14, 4))
        voice_buttons = ttk.Frame(frame)
        voice_buttons.pack(fill="x", pady=(2, 4))
        self.listen_button = ttk.Button(
            voice_buttons,
            text=UI_TEXT["el"]["listen"],
            command=self.read_question,
        )
        self.listen_button.pack(side="left")
        self.voice_button = ttk.Button(
            voice_buttons,
            text=UI_TEXT["el"]["start_voice"],
            command=self.toggle_voice,
        )
        self.voice_button.pack(side="left", padx=(10, 0))

        self.comment_label = ttk.Label(frame, text=UI_TEXT["el"]["comment"])
        self.comment_label.pack(anchor="w", pady=(18, 6))
        self.comment = tk.Text(frame, height=7, wrap="word")
        self.comment.pack(fill="x")
        self.comment.bind("<KeyRelease>", lambda _event: self._invalidate())

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=16)
        self.preview_button = ttk.Button(buttons, text=UI_TEXT["el"]["preview"], command=self.show_preview)
        self.preview_button.pack(side="left")
        self.apply_button = ttk.Button(
            buttons,
            text=UI_TEXT["el"]["apply"],
            command=self.apply,
            state="disabled",
        )
        self.apply_button.pack(side="left", padx=(10, 0))

        ttk.Label(frame, textvariable=self.status, wraplength=700, justify="left").pack(anchor="w")
        self.details = tk.Text(frame, height=18, wrap="word", state="disabled")
        self.details.pack(fill="both", expand=True, pady=(10, 0))

    @property
    def ui_language(self) -> str:
        return LANGUAGE_NAMES[self.language_name.get()]

    def _t(self, key: str) -> str:
        return UI_TEXT[self.ui_language][key]

    def _language_changed(self) -> None:
        self.language_label.configure(text=self._t("language"))
        self.intro_label.configure(text=self._t("intro"))
        self.level_label.configure(text=self._t("level"))
        self.listen_button.configure(text=self._t("listen"))
        self.comment_label.configure(text=self._t("comment"))
        self.preview_button.configure(text=self._t("preview"))
        self.apply_button.configure(text=self._t("apply"))
        self.question.set(feedback_question(self.level.get(), self.ui_language))
        if self.voice_busy:
            self.voice_button.configure(text=self._t("transcribing"))
            self.status.set(self._t("transcribing_status"))
        elif self.voice_recorder is not None and self.voice_recorder.is_recording:
            self.voice_button.configure(text=self._t("stop_voice"))
            self.status.set(self._t("recording"))
        else:
            self.voice_button.configure(text=self._t("start_voice"))
            if self.preview is None:
                self.status.set(self._t("initial"))
            elif self.preview.get("can_apply"):
                self.status.set(self._t("ready"))
            else:
                self.status.set(self._t("no_change"))
        if self.preview is not None:
            self._write_details(format_preview(self.preview, self.ui_language))

    def _invalidate(self) -> None:
        self.preview = None
        self.source = "text"
        self.locale = None
        self.apply_button.configure(state="disabled")
        self.status.set(self._t("changed"))

    def _level_changed(self) -> None:
        self.question.set(feedback_question(self.level.get(), self.ui_language))
        self._invalidate()

    def read_question(self) -> None:
        try:
            speak_question(self.level.get(), language=self.ui_language)
        except VoiceUnavailable as exc:
            messagebox.showinfo("Δεν μπόρεσα να διαβάσω την ερώτηση", str(exc))

    def toggle_voice(self) -> None:
        if self.voice_busy:
            return
        if self.voice_recorder is not None and self.voice_recorder.is_recording:
            self._stop_and_transcribe()
            return
        try:
            self.voice_recorder = VoiceRecorder()
            self.voice_recorder.start()
        except VoiceUnavailable as exc:
            messagebox.showinfo("Δεν άνοιξε το μικρόφωνο", str(exc))
            return
        self.voice_button.configure(text=self._t("stop_voice"))
        self.status.set(self._t("recording"))

    def _stop_and_transcribe(self) -> None:
        assert self.voice_recorder is not None
        try:
            audio = self.voice_recorder.stop()
        except VoiceUnavailable as exc:
            messagebox.showinfo("Δεν ολοκληρώθηκε η εγγραφή", str(exc))
            self.voice_button.configure(text=self._t("start_voice"))
            return
        self.voice_busy = True
        self.voice_button.configure(text=self._t("transcribing"), state="disabled")
        self.status.set(self._t("transcribing_status"))
        threading.Thread(target=self._transcribe_worker, args=(audio,), daemon=True).start()

    def _transcribe_worker(self, audio) -> None:
        try:
            result = self.voice_transcriber.transcribe(audio)
        except VoiceUnavailable as exc:
            self.root.after(0, self._voice_failed, str(exc))
            return
        self.root.after(0, self._voice_ready, result)

    def _voice_ready(self, result: dict) -> None:
        self.voice_busy = False
        self.voice_button.configure(text=self._t("start_voice"), state="normal")
        self.accept_voice_transcript(result["text"], result["locale"])

    def _voice_failed(self, message: str) -> None:
        self.voice_busy = False
        self.voice_button.configure(text=self._t("start_voice"), state="normal")
        self.status.set(self._t("voice_failed"))
        messagebox.showinfo("Δεν ολοκληρώθηκε η μεταγραφή", message)

    def accept_voice_transcript(self, transcript: str, locale: str) -> None:
        """Entry point used by the upcoming push-to-talk capture layer."""
        self.comment.delete("1.0", "end")
        self.comment.insert("1.0", transcript)
        self.source = "voice"
        self.locale = locale
        self.preview = None
        self.apply_button.configure(state="disabled")
        self.status.set(self._t("voice_ready"))

    def _write_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def show_preview(self) -> None:
        text = self.comment.get("1.0", "end").strip()
        profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
        if PROFILE_PATH.exists():
            try:
                import json

                profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                messagebox.showerror(
                    "Δεν μπορώ να διαβάσω το feedback",
                    "Το υπάρχον learning_profile.json δεν είναι έγκυρο. Δεν άλλαξα τίποτα.",
                )
                return
        try:
            self.preview = build_feedback_preview(
                text,
                dream_level=self.level.get(),
                source=self.source,
                locale=self.locale,
                profile=profile,
                interpreter="auto",
            )
        except ValueError as exc:
            messagebox.showinfo("Έλεγξε το σχόλιο", str(exc))
            return
        self._write_details(format_preview(self.preview, self.ui_language))
        if self.preview["can_apply"]:
            self.status.set(self._t("ready"))
            self.apply_button.configure(state="normal")
        else:
            self.status.set(self._t("no_change"))
            self.apply_button.configure(state="disabled")

    def apply(self) -> None:
        if not self.preview or not self.preview.get("can_apply"):
            messagebox.showinfo("Πρώτα προεπισκόπηση", "Δες πρώτα τι κατάλαβε το Hyponoia.")
            return
        try:
            _profile, event = apply_feedback_preview(
                self.preview,
                profile_path=PROFILE_PATH,
                evidence_dir=EVIDENCE_DIR,
                confirmed=True,
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Δεν αποθηκεύτηκε", f"Δεν άλλαξα τίποτα.\n\n{exc}")
            return
        self.apply_button.configure(state="disabled")
        self.status.set(self._t("saved").format(levels=", ".join(event["target_levels"])))

    def close(self) -> None:
        if self.voice_recorder is not None:
            self.voice_recorder.cancel()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FeedbackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
