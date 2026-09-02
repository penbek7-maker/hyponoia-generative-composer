"""Non-technical preview-and-confirm window for Hyponoia free feedback."""

from __future__ import annotations

import copy
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from feedback_input_v1 import apply_feedback_preview, build_feedback_preview
from human_feedback_v1 import DEFAULT_LEARNING_PROFILE


PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = PROJECT_DIR / "learning_profile.json"
EVIDENCE_DIR = PROJECT_DIR / "human_feedback"


def format_preview(preview: dict) -> str:
    if preview.get("status") == "empty":
        return "Δεν έχει γραφτεί ακόμη σχόλιο."
    if not preview.get("can_apply"):
        return (
            "Δεν αναγνώρισα ακόμη συγκεκριμένη μουσική αλλαγή. "
            "Δοκίμασε π.χ. ‘περισσότερο synth, λιγότερη επανάληψη και πιο ομαλές μεταβάσεις’."
        )
    targets = ", ".join(preview.get("target_levels", []))
    lines = [f"Θα επηρεαστεί: {targets}", "", "Κατάλαβα:"]
    lines.extend(f"• {action['label_el']}" for action in preview.get("actions", []))
    lines.extend(["", "Ακριβείς μικρές αλλαγές:"])
    for change in preview.get("control_changes", []):
        lines.append(
            f"• {change['target_level']} — {change['control']}: "
            f"{change['old_value']:.2f} → {change['new_value']:.2f}"
        )
    return "\n".join(lines)


class FeedbackApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hyponoia — Feedback")
        self.root.geometry("760x650")
        self.root.minsize(650, 560)
        self.level = tk.StringVar(value="D1")
        self.status = tk.StringVar(value="Γράψε ελεύθερα τι θέλεις να αλλάξει.")
        self.source = "text"
        self.locale: str | None = None
        self.preview: dict | None = None
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Hyponoia Feedback", font=("Helvetica", 24, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=(
                "Το Hyponoia θα σου δείξει πρώτα τι κατάλαβε. "
                "Τίποτα δεν αλλάζει πριν πατήσεις ‘Εφάρμοσε’."
            ),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(8, 18))

        level_row = ttk.Frame(frame)
        level_row.pack(fill="x")
        ttk.Label(level_row, text="Ποια σύνθεση άκουσες;").pack(side="left")
        level_menu = ttk.Combobox(
            level_row,
            textvariable=self.level,
            values=("D1", "D3", "D5"),
            width=6,
            state="readonly",
        )
        level_menu.pack(side="left", padx=(10, 0))
        level_menu.bind("<<ComboboxSelected>>", lambda _event: self._invalidate())

        ttk.Label(frame, text="Ελεύθερο σχόλιο:").pack(anchor="w", pady=(18, 6))
        self.comment = tk.Text(frame, height=7, wrap="word")
        self.comment.pack(fill="x")
        self.comment.bind("<KeyRelease>", lambda _event: self._invalidate())

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=16)
        ttk.Button(buttons, text="1. Δείξε μου τι κατάλαβες", command=self.show_preview).pack(side="left")
        self.apply_button = ttk.Button(
            buttons,
            text="2. Εφάρμοσε το feedback",
            command=self.apply,
            state="disabled",
        )
        self.apply_button.pack(side="left", padx=(10, 0))

        ttk.Label(frame, textvariable=self.status, wraplength=700, justify="left").pack(anchor="w")
        self.details = tk.Text(frame, height=18, wrap="word", state="disabled")
        self.details.pack(fill="both", expand=True, pady=(10, 0))

    def _invalidate(self) -> None:
        self.preview = None
        self.source = "text"
        self.locale = None
        self.apply_button.configure(state="disabled")
        self.status.set("Το σχόλιο άλλαξε. Δες ξανά τι κατάλαβε το Hyponoia.")

    def accept_voice_transcript(self, transcript: str, locale: str) -> None:
        """Entry point used by the upcoming push-to-talk capture layer."""
        self.comment.delete("1.0", "end")
        self.comment.insert("1.0", transcript)
        self.source = "voice"
        self.locale = locale
        self.preview = None
        self.apply_button.configure(state="disabled")
        self.status.set("Η φωνή μεταγράφηκε. Έλεγξε το κείμενο και δες τι κατάλαβα.")

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
            )
        except ValueError as exc:
            messagebox.showinfo("Έλεγξε το σχόλιο", str(exc))
            return
        self._write_details(format_preview(self.preview))
        if self.preview["can_apply"]:
            self.status.set("Έτοιμο για έλεγχο. Αν συμφωνείς, πάτησε ‘Εφάρμοσε το feedback’.")
            self.apply_button.configure(state="normal")
        else:
            self.status.set("Δεν έγινε καμία αλλαγή. Δοκίμασε πιο συγκεκριμένη διατύπωση.")
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
        self.status.set(
            f"Το feedback αποθηκεύτηκε για {', '.join(event['target_levels'])}. "
            "Θα επηρεάσει την επόμενη σύνθεση."
        )


def main() -> None:
    root = tk.Tk()
    FeedbackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
