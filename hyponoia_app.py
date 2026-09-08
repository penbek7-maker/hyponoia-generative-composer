"""One front door for the Hyponoia library, composer, player and feedback."""

from __future__ import annotations

import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from hyponoia_feedback_app import FeedbackApp
from hyponoia_runtime import PROJECT_DIR, generator_command, runtime_status
from update_library_v1 import update_library


class HyponoiaApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hyponoia")
        self.root.geometry("1000x850")
        self.root.minsize(820, 700)
        self.library_folder = tk.StringVar()
        self.level = tk.StringVar(value="D1")
        self.scale = tk.StringVar(value="free")
        self.root_pitch = tk.IntVar(value=0)
        self.confidence = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Checking Hyponoia…")
        self._build()
        self.refresh_status()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Hyponoia", font=("Helvetica", 28, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Library → Generate → Listen → Feedback → Generate again",
        ).pack(anchor="w", pady=(2, 14))
        ttk.Label(frame, textvariable=self.status, wraplength=830, justify="left").pack(anchor="w")

        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill="both", expand=True, pady=(16, 0))
        library_tab = ttk.Frame(self.notebook, padding=18)
        compose_tab = ttk.Frame(self.notebook, padding=18)
        feedback_tab = ttk.Frame(self.notebook)
        self.notebook.add(library_tab, text="1. Library")
        self.notebook.add(compose_tab, text="2. Generate & Listen")
        self.notebook.add(feedback_tab, text="3. Teach Hyponoia")
        self._build_library(library_tab)
        self._build_composer(compose_tab)
        self._build_feedback(feedback_tab)

    def _build_library(self, tab: ttk.Frame) -> None:
        ttk.Label(
            tab, text="Choose a folder with WAV files", font=("Helvetica", 18, "bold")
        ).pack(anchor="w")
        ttk.Label(
            tab,
            text=(
                "About 100 recordings are recommended, but fewer work too. "
                "You can add or remove sounds whenever you want."
            ),
            wraplength=780,
        ).pack(anchor="w", pady=(6, 14))
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.library_folder).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choose…", command=self.choose_library).pack(side="left", padx=(8, 0))
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=14)
        self.preview_library_button = ttk.Button(
            buttons, text="Preview changes", command=lambda: self.run_library(False)
        )
        self.preview_library_button.pack(side="left")
        self.update_library_button = ttk.Button(
            buttons, text="Update library", command=lambda: self.run_library(True)
        )
        self.update_library_button.pack(side="left", padx=(8, 0))
        self.library_details = tk.Text(tab, height=16, wrap="word", state="disabled")
        self.library_details.pack(fill="both", expand=True)

    def _build_composer(self, tab: ttk.Frame) -> None:
        ttk.Label(tab, text="Create a composition", font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(tab, text="Dream level").grid(row=1, column=0, sticky="w", pady=(18, 4))
        ttk.Combobox(
            tab, textvariable=self.level, values=("D1", "D3", "D5"), state="readonly", width=8
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(tab, text="Scale").grid(row=1, column=1, sticky="w", padx=(18, 0), pady=(18, 4))
        ttk.Combobox(
            tab, textvariable=self.scale, values=("free", "major", "minor"), state="readonly", width=10
        ).grid(row=2, column=1, sticky="w", padx=(18, 0))
        ttk.Label(tab, text="Root (0–11)").grid(row=1, column=2, sticky="w", padx=(18, 0), pady=(18, 4))
        ttk.Spinbox(tab, from_=0, to=11, textvariable=self.root_pitch, width=7).grid(
            row=2, column=2, sticky="w", padx=(18, 0)
        )
        ttk.Label(tab, text="Harmony confidence").grid(
            row=1, column=3, sticky="w", padx=(18, 0), pady=(18, 4)
        )
        ttk.Spinbox(
            tab, from_=0.0, to=1.0, increment=0.05, textvariable=self.confidence, width=8
        ).grid(row=2, column=3, sticky="w", padx=(18, 0))
        buttons = ttk.Frame(tab)
        buttons.grid(row=3, column=0, columnspan=4, sticky="w", pady=24)
        self.generate_button = ttk.Button(buttons, text="Generate", command=self.generate)
        self.generate_button.pack(side="left")
        ttk.Button(buttons, text="Listen to latest", command=self.listen).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Show output folder", command=self.show_output).pack(side="left", padx=(8, 0))
        self.render_details = tk.Text(tab, height=19, wrap="word", state="disabled")
        self.render_details.grid(row=4, column=0, columnspan=4, sticky="nsew")
        tab.rowconfigure(4, weight=1)
        tab.columnconfigure(3, weight=1)

    def _build_feedback(self, tab: ttk.Frame) -> None:
        self.feedback_app = FeedbackApp(
            tab,
            embedded=True,
            level_var=self.level,
            on_applied=self.feedback_applied,
        )

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def refresh_status(self) -> None:
        info = runtime_status(PROJECT_DIR)
        deep = info["representation"]
        preference = info["composition_preference"]
        self.status.set(
            f"Library: {info['recordings']} recordings / {info['sound_objects']} objects  •  "
            f"Source WAVs: {'ON' if info['source_audio_ready'] else 'MISSING'} ({info['source_wav_count']})  •  "
            f"Deep embeddings: {'ON' if deep['active'] else 'OFF'} ({deep['embedding_count']})  •  "
            f"Preference learning: {'ON' if preference['active'] else 'OFF'} "
            f"({info['preference_review_count']} personal reviews)"
        )

    def choose_library(self) -> None:
        selected = filedialog.askdirectory(title="Choose your Hyponoia WAV folder")
        if selected:
            self.library_folder.set(selected)

    def run_library(self, apply: bool) -> None:
        path = Path(self.library_folder.get()).expanduser()
        if not path.is_dir():
            messagebox.showinfo("Choose a folder", "Choose a folder containing WAV files first.")
            return
        self.preview_library_button.configure(state="disabled")
        self.update_library_button.configure(state="disabled")
        self._set_text(self.library_details, "Updating…" if apply else "Checking…")

        def work() -> None:
            try:
                result = update_library(path, PROJECT_DIR, apply=apply)
                plan = result.get("plan", {}).get("summary", {})
                embedding = result.get("embedding_refresh") or {}
                lines = [
                    f"Status: {result.get('status')}",
                    f"Active WAVs: {plan.get('active_wavs', 0)}",
                    f"Added: {plan.get('added', 0)}  Changed: {plan.get('modified', 0)}  Removed: {plan.get('removed', 0)}",
                    f"Unchanged: {plan.get('unchanged', 0)}",
                    f"Embeddings reused: {embedding.get('reused_embeddings', 0)}",
                    f"Embeddings created: {embedding.get('created_embeddings', 0)}",
                ]
                if result.get("error"):
                    lines.append(f"Problem: {result['error']}")
                text = "\n".join(lines)
            except Exception as exc:
                text = f"Nothing changed.\n\n{type(exc).__name__}: {exc}"
            self.root.after(0, lambda: self._library_done(text))

        threading.Thread(target=work, daemon=True).start()

    def _library_done(self, text: str) -> None:
        self._set_text(self.library_details, text)
        self.preview_library_button.configure(state="normal")
        self.update_library_button.configure(state="normal")
        self.refresh_status()

    def generate(self) -> None:
        info = runtime_status(PROJECT_DIR)
        if not info["ready_to_generate"]:
            messagebox.showinfo(
                "Hyponoia is not ready",
                "Choose and update the sound library first, then check that Source WAVs and both learning models show ON.",
            )
            return
        level = int(self.level.get()[1:])
        try:
            command = generator_command(
                level,
                root_pitch=self.root_pitch.get(),
                scale=self.scale.get(),
                confidence=self.confidence.get(),
                project_dir=PROJECT_DIR,
            )
        except ValueError as exc:
            messagebox.showinfo("Check the settings", str(exc))
            return
        self.generate_button.configure(state="disabled")
        self._set_text(self.render_details, f"Generating {self.level.get()}…")

        def work() -> None:
            completed = subprocess.run(command, cwd=PROJECT_DIR, text=True, capture_output=True)
            output = completed.stdout
            if completed.stderr:
                output += "\n" + completed.stderr
            self.root.after(0, lambda: self._generation_done(completed.returncode, output))

        threading.Thread(target=work, daemon=True).start()

    def _generation_done(self, returncode: int, output: str) -> None:
        self.generate_button.configure(state="normal")
        self._set_text(
            self.render_details, output.strip() or f"Generator exited with code {returncode}."
        )
        self.refresh_status()
        if returncode == 0:
            messagebox.showinfo("Hyponoia", "The composition is ready. Press ‘Listen to latest’.")

    def listen(self) -> None:
        path = PROJECT_DIR / "output" / "current.wav"
        if not path.exists():
            messagebox.showinfo("No composition yet", "Generate a composition first.")
            return
        subprocess.Popen(["open", str(path)])

    def show_output(self) -> None:
        output = PROJECT_DIR / "output"
        output.mkdir(exist_ok=True)
        subprocess.Popen(["open", str(output)])

    def open_feedback(self) -> None:
        child = tk.Toplevel(self.root)
        FeedbackApp(child)

    def feedback_applied(self, _event: dict, learning: dict) -> None:
        self.refresh_status()
        if learning.get("updated"):
            messagebox.showinfo(
                "Hyponoia learned",
                "The ratings and comment were saved. The whole-composition preference model was updated safely.",
            )


def main() -> None:
    root = tk.Tk()
    HyponoiaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
