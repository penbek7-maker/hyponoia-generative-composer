"""Unified ratings, text and local-voice feedback for the Hyponoia app."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from adaptive_composition_preference_v1 import preference_score, update_preference_from_review
from composition_feedback_v1 import apply_composition_feedback, build_composition_feedback
from feedback_input_v1 import INTENT_LABELS_EL, INTENT_LABELS_EN, load_feedback_profile
from hyponoia_runtime import update_user_config
from hyponoia_stability import atomic_write_json
from learning_backup_v1 import archive_and_reset_learning, export_learning_backup
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
RENDER_REPORT_PATH = PROJECT_DIR / "render_report.json"
BASE_PREFERENCE_PATH = PROJECT_DIR / "phase2_artifacts" / "composition_preference_gold.json"
USER_PREFERENCE_PATH = PROJECT_DIR / "composition_preference_v1.json"


LANGUAGE_NAMES = {"Ελληνικά": "el", "English": "en"}
BASELINE_NAMES = {
    "el": {"Ναι": "yes", "Όχι": "no", "Δεν είμαι σίγουρη": "unsure"},
    "en": {"Yes": "yes", "No": "no", "Not sure": "unsure"},
}
RATING_LABELS = {
    "el": {
        "musicality": "Μουσικότητα",
        "material_coherence": "Συνοχή υλικών",
        "transition_smoothness": "Ομαλότητα μεταβάσεων",
        "variety_without_disconnection": "Ποικιλία με συνοχή",
        "synth_material_presence": "Παρουσία synth υλικού",
        "development_over_repetition": "Ανάπτυξη αντί επανάληψης",
        "overall_artistic_impression": "Συνολική εντύπωση",
    },
    "en": {
        "musicality": "Musicality",
        "material_coherence": "Material coherence",
        "transition_smoothness": "Transition smoothness",
        "variety_without_disconnection": "Variety with coherence",
        "synth_material_presence": "Synth material presence",
        "development_over_repetition": "Development over repetition",
        "overall_artistic_impression": "Overall impression",
    },
}

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
        "ratings": "Βαθμολόγησε τη σύνθεση (1–5)",
        "baseline": "Θα την κρατούσες ως βάση;",
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
        "saved": "Το feedback αποθηκεύτηκε για {levels}. Η επόμενη σύνθεση θα χρησιμοποιήσει τη νέα μάθηση.",
        "export": "Εξαγωγή learning backup",
        "reset": "Επιστροφή στην αρχική βάση",
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
        "ratings": "Rate the composition (1–5)",
        "baseline": "Would you keep it as a baseline?",
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
        "saved": "Feedback was saved for {levels}. The next composition will use the updated learning.",
        "export": "Export learning backup",
        "reset": "Return to release baseline",
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


def format_composition_review(event: dict, changes: list[dict], language: str = "el") -> str:
    """Explain ratings, language interpretation and whole-render learning together."""
    score = preference_score(event["ratings_1_to_5"])
    decision = event["baseline_decision"]
    if language == "en":
        direction = {"yes": "positive example", "no": "contrast example", "unsure": "no neural movement"}[decision]
        lines = [
            f"Whole-composition score: {score:.2f} / 5",
            f"Preference learning: {direction}",
            "",
            "I understood:",
        ]
        labels = INTENT_LABELS_EN
    else:
        direction = {"yes": "θετικό παράδειγμα", "no": "αρνητικό παράδειγμα", "unsure": "χωρίς μετακίνηση του neural model"}[decision]
        lines = [
            f"Συνολική μαθησιακή βαθμολογία: {score:.2f} / 5",
            f"Μάθηση προτίμησης: {direction}",
            "",
            "Κατάλαβα:",
        ]
        labels = INTENT_LABELS_EL
    actions = event.get("comment_interpretation", {}).get("actions", [])
    if actions:
        lines.extend(f"• {labels.get(action.get('intent'), action.get('intent'))}" for action in actions)
    else:
        lines.append("• " + ("the numeric ratings" if language == "en" else "τις αριθμητικές βαθμολογίες"))
    lines.extend(["", "Exact bounded changes:" if language == "en" else "Ακριβείς περιορισμένες αλλαγές:"])
    for change in changes:
        lines.append(
            f"• {change['target_level']} — {change['control']}: "
            f"{change['old_value']:.2f} → {change['new_value']:.2f}"
        )
    domains = event.get("composition_influence", {}).get("domains", [])
    if domains:
        key = "label_en" if language == "en" else "label_el"
        lines.extend(["", "Will influence:" if language == "en" else "Θα επηρεάσει:"])
        lines.extend(f"• {domain.get(key, domain['domain'])}" for domain in domains)
    return "\n".join(lines)


class FeedbackApp:
    def __init__(
        self,
        container: tk.Misc,
        *,
        embedded: bool = False,
        level_var: tk.StringVar | None = None,
        on_applied=None,
    ) -> None:
        self.container = container
        self.root = container.winfo_toplevel()
        self.embedded = embedded
        self.on_applied = on_applied
        if not embedded:
            self.root.title("Hyponoia — Feedback")
            self.root.geometry("860x820")
            self.root.minsize(720, 650)
        self.language_name = tk.StringVar(master=container, value="Ελληνικά")
        self.level = level_var if level_var is not None else tk.StringVar(master=container, value="D1")
        self.baseline_name = tk.StringVar(master=container, value="Δεν είμαι σίγουρη")
        self.rating_values = {
            name: tk.DoubleVar(master=container, value=3.0) for name in RATING_LABELS["el"]
        }
        self.status = tk.StringVar(value=UI_TEXT["el"]["initial"])
        self.source = "text"
        self.locale: str | None = None
        self.preview: dict | None = None
        self.voice_recorder: VoiceRecorder | None = None
        self.voice_transcriber = LocalWhisperTranscriber()
        self.voice_busy = False
        self._build()
        if not embedded:
            self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        frame = ttk.Frame(self.container, padding=18 if self.embedded else 24)
        frame.pack(fill="both", expand=True)
        title_row = ttk.Frame(frame)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="Teach Hyponoia", font=("Helvetica", 20, "bold")).pack(side="left")
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
        self.intro_label.pack(anchor="w", pady=(6, 12))

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

        self.ratings_frame = ttk.LabelFrame(frame, text=UI_TEXT["el"]["ratings"], padding=10)
        self.ratings_frame.pack(fill="x", pady=(12, 4))
        self.rating_labels = {}
        for index, (name, label) in enumerate(RATING_LABELS["el"].items()):
            row, pair = divmod(index, 2)
            column = pair * 2
            widget = ttk.Label(self.ratings_frame, text=label)
            widget.grid(row=row, column=column, sticky="w", padx=(0, 8), pady=3)
            self.rating_labels[name] = widget
            control = ttk.Spinbox(
                self.ratings_frame,
                from_=1.0,
                to=5.0,
                increment=0.5,
                width=5,
                textvariable=self.rating_values[name],
                command=self._invalidate,
            )
            control.grid(row=row, column=column + 1, sticky="w", padx=(0, 18), pady=3)
            control.bind("<FocusOut>", lambda _event: self._invalidate())
        self.ratings_frame.columnconfigure(0, weight=1)
        self.ratings_frame.columnconfigure(2, weight=1)

        baseline_row = ttk.Frame(frame)
        baseline_row.pack(fill="x", pady=(8, 0))
        self.baseline_label = ttk.Label(baseline_row, text=UI_TEXT["el"]["baseline"])
        self.baseline_label.pack(side="left")
        self.baseline_menu = ttk.Combobox(
            baseline_row,
            textvariable=self.baseline_name,
            values=tuple(BASELINE_NAMES["el"]),
            width=19,
            state="readonly",
        )
        self.baseline_menu.pack(side="left", padx=(10, 0))
        self.baseline_menu.bind("<<ComboboxSelected>>", lambda _event: self._invalidate())

        self.comment_label = ttk.Label(frame, text=UI_TEXT["el"]["comment"])
        self.comment_label.pack(anchor="w", pady=(10, 4))
        self.comment = tk.Text(frame, height=4, wrap="word")
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
        self.details = tk.Text(frame, height=12, wrap="word", state="disabled")
        self.details.pack(fill="both", expand=True, pady=(10, 0))
        learning_actions = ttk.Frame(frame)
        learning_actions.pack(fill="x", pady=(10, 0))
        self.export_button = ttk.Button(
            learning_actions, text=UI_TEXT["el"]["export"], command=self.export_learning
        )
        self.export_button.pack(side="left")
        self.reset_button = ttk.Button(
            learning_actions, text=UI_TEXT["el"]["reset"], command=self.reset_learning
        )
        self.reset_button.pack(side="left", padx=(8, 0))

    @property
    def ui_language(self) -> str:
        return LANGUAGE_NAMES[self.language_name.get()]

    def _t(self, key: str) -> str:
        return UI_TEXT[self.ui_language][key]

    @property
    def baseline_decision(self) -> str:
        for choices in BASELINE_NAMES.values():
            if self.baseline_name.get() in choices:
                return choices[self.baseline_name.get()]
        return "unsure"

    def ratings(self) -> dict[str, float]:
        return {name: float(value.get()) for name, value in self.rating_values.items()}

    def _language_changed(self) -> None:
        decision = self.baseline_decision
        self.language_label.configure(text=self._t("language"))
        self.intro_label.configure(text=self._t("intro"))
        self.level_label.configure(text=self._t("level"))
        self.listen_button.configure(text=self._t("listen"))
        self.comment_label.configure(text=self._t("comment"))
        self.ratings_frame.configure(text=self._t("ratings"))
        self.baseline_label.configure(text=self._t("baseline"))
        for name, widget in self.rating_labels.items():
            widget.configure(text=RATING_LABELS[self.ui_language][name])
        choices = BASELINE_NAMES[self.ui_language]
        self.baseline_menu.configure(values=tuple(choices))
        self.baseline_name.set(next(label for label, code in choices.items() if code == decision))
        self.preview_button.configure(text=self._t("preview"))
        self.apply_button.configure(text=self._t("apply"))
        self.export_button.configure(text=self._t("export"))
        self.reset_button.configure(text=self._t("reset"))
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
            self._write_details(
                format_composition_review(
                    self.preview["event"], self.preview["changes"], self.ui_language
                )
            )

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
        if not RENDER_REPORT_PATH.exists():
            messagebox.showinfo(
                "Generate first",
                "Create and listen to a composition before giving feedback.",
            )
            return
        try:
            report = json.loads(RENDER_REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            messagebox.showerror("Feedback unavailable", "The latest render report could not be read.")
            return
        expected_level = int(self.level.get()[1:])
        if int(report.get("dream_level", -1)) != expected_level:
            messagebox.showinfo(
                "Choose the latest composition",
                f"The latest composition is D{report.get('dream_level')}. Select that level or generate {self.level.get()} first.",
            )
            return
        try:
            profile = load_feedback_profile(PROFILE_PATH)
        except ValueError:
            messagebox.showerror(
                "Δεν μπορώ να διαβάσω το feedback",
                "Το υπάρχον learning_profile.json δεν είναι έγκυρο. Δεν άλλαξα τίποτα.",
            )
            return
        try:
            event = build_composition_feedback(
                self.ratings(),
                dream_level=self.level.get(),
                keep_as_baseline=self.baseline_decision == "yes",
                baseline_decision=self.baseline_decision,
                comment=text,
                render_name=Path(str(report.get("audio_file", "current.wav"))).name,
                source=self.source,
                locale=self.locale,
                interpreter="auto",
            )
            _proposed, changes = apply_composition_feedback(profile, event)
        except ValueError as exc:
            messagebox.showinfo("Έλεγξε το σχόλιο", str(exc))
            return
        self.preview = {"event": event, "changes": changes}
        self._write_details(format_composition_review(event, changes, self.ui_language))
        if changes or self.baseline_decision != "unsure":
            self.status.set(self._t("ready"))
            self.apply_button.configure(state="normal")
        else:
            self.status.set(self._t("no_change"))
            self.apply_button.configure(state="disabled")

    def apply(self) -> None:
        if not self.preview:
            messagebox.showinfo("Πρώτα προεπισκόπηση", "Δες πρώτα τι κατάλαβε το Hyponoia.")
            return
        event = self.preview["event"]
        try:
            profile = load_feedback_profile(PROFILE_PATH)
            updated, _changes = apply_composition_feedback(profile, event)
            atomic_write_json(PROFILE_PATH, updated)
            atomic_write_json(EVIDENCE_DIR / f"{event['event_id']}_composition_feedback.json", event)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Δεν αποθηκεύτηκε", f"Δεν άλλαξα τίποτα.\n\n{exc}")
            return
        try:
            learning = update_preference_from_review(
                base_model_path=BASE_PREFERENCE_PATH,
                user_model_path=USER_PREFERENCE_PATH,
                render_report_path=RENDER_REPORT_PATH,
                event=event,
            )
            if learning["updated"]:
                update_user_config(PROJECT_DIR, composition_preference=USER_PREFERENCE_PATH)
        except (OSError, ValueError) as exc:
            learning = {"updated": False, "error": str(exc)}
            messagebox.showwarning(
                "Το feedback αποθηκεύτηκε",
                "Οι αλλαγές σύνθεσης αποθηκεύτηκαν, αλλά το neural preference head δεν ενημερώθηκε.\n\n"
                + str(exc),
            )
        self.apply_button.configure(state="disabled")
        self.status.set(self._t("saved").format(levels=event["dream_level"]))
        if callable(self.on_applied):
            self.on_applied(event, learning)

    def export_learning(self) -> None:
        destination = filedialog.asksaveasfilename(
            title=self._t("export"),
            defaultextension=".zip",
            initialfile="Hyponoia_Learning_Backup.zip",
            filetypes=(("ZIP archive", "*.zip"),),
        )
        if not destination:
            return
        try:
            result = export_learning_backup(PROJECT_DIR, destination)
        except OSError as exc:
            messagebox.showerror("Backup failed", str(exc))
            return
        messagebox.showinfo("Learning backup", f"Saved:\n{result['path']}")

    def reset_learning(self) -> None:
        message = (
            "Η προσωπική μάθηση θα αρχειοθετηθεί και το Hyponoia θα επιστρέψει στην εγκεκριμένη αρχική βάση. Η βιβλιοθήκη δεν θα αλλάξει."
            if self.ui_language == "el"
            else "Personal learning will be archived and Hyponoia will return to the approved release baseline. The library will not change."
        )
        if not messagebox.askyesno(self._t("reset"), message):
            return
        try:
            result = archive_and_reset_learning(PROJECT_DIR)
        except OSError as exc:
            messagebox.showerror("Reset failed", str(exc))
            return
        self.preview = None
        self.apply_button.configure(state="disabled")
        self.status.set(
            ("Η αρχική βάση επανήλθε. Backup: " if self.ui_language == "el" else "Release baseline restored. Backup: ")
            + result["archive"]
        )
        if callable(self.on_applied):
            self.on_applied({}, {"updated": False})

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
