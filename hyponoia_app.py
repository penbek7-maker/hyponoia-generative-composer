"""One front door for the Hyponoia library, composer, player and feedback."""

from __future__ import annotations

import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from hyponoia_feedback_app import FeedbackApp
from hyponoia_runtime import PROJECT_DIR, generator_command, runtime_status
from max_live_v1 import MaxLiveController, find_max_project
from update_library_v1 import update_library


ROOT_NOTE_PITCHES = {
    "C": 0,
    "C♯ / D♭": 1,
    "D": 2,
    "D♯ / E♭": 3,
    "E": 4,
    "F": 5,
    "F♯ / G♭": 6,
    "G": 7,
    "G♯ / A♭": 8,
    "A": 9,
    "A♯ / B♭": 10,
    "B": 11,
}


def format_runtime_summary(info: dict) -> str:
    """Turn technical readiness into one useful next-step message."""
    if info.get("ready_to_generate"):
        return (
            f"Ready to compose · {info.get('recordings', 0)} recordings"
            " · deep listening and learning active"
        )
    if not info.get("source_audio_ready"):
        return "Start here: choose a folder containing your WAV sounds, then update the library."
    if not info.get("memory_ready"):
        return "Your sounds are selected. Update the library to prepare them for composition."
    if not info.get("representation", {}).get("active"):
        return "The sound library is ready, but the deep-listening model needs attention."
    if not info.get("composition_preference", {}).get("active"):
        return "The sound library is ready, but the preference model needs attention."
    return "Hyponoia is checking the local composition environment."


class HyponoiaApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hyponoia")
        self.root.geometry("1080x820")
        self.root.minsize(900, 680)
        self.library_folder = tk.StringVar()
        self.level = tk.StringVar(value="D1")
        self.scale = tk.StringVar(value="minor")
        self.root_note = tk.StringVar(value="C")
        self.confidence = tk.DoubleVar(value=0.8)
        self.status = tk.StringVar(value="Checking Hyponoia…")
        self.render_status = tk.StringVar(
            value="Choose a dream level and musical scale, then create your composition."
        )
        self.render_log_visible = False
        self.max_status = tk.StringVar(value="Live connection is stopped.")
        self.max_controller = MaxLiveController(PROJECT_DIR)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_status()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Hyponoia", font=("Helvetica", 30, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Your sounds → Your composition → Your feedback → A new composition",
        ).pack(anchor="w", pady=(2, 12))
        status_box = ttk.LabelFrame(frame, text="Hyponoia status", padding=(12, 8))
        status_box.pack(fill="x")
        ttk.Label(status_box, textvariable=self.status, justify="left").pack(anchor="w")

        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill="both", expand=True, pady=(16, 0))
        library_tab = ttk.Frame(self.notebook, padding=18)
        compose_tab = ttk.Frame(self.notebook, padding=18)
        feedback_tab = ttk.Frame(self.notebook)
        live_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(library_tab, text="1. Library")
        self.notebook.add(compose_tab, text="2. Generate & Listen")
        self.notebook.add(feedback_tab, text="3. Teach Hyponoia")
        self.notebook.add(live_tab, text="4. Live / Max")
        self.compose_tab = compose_tab
        self.feedback_tab = feedback_tab
        self._build_library(library_tab)
        self._build_composer(compose_tab)
        self._build_feedback(feedback_tab)
        self._build_live(live_tab)

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
        self._set_text(
            self.library_details,
            "Choose your WAV folder, preview the changes, then update the library. "
            "Hyponoia never moves or edits your original sounds.",
        )

    def _build_composer(self, tab: ttk.Frame) -> None:
        ttk.Label(tab, text="Create a composition", font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(
            tab,
            text="D1 is focused, D3 develops further, and D5 creates the fullest form.",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 8))
        ttk.Label(tab, text="Dream level").grid(row=2, column=0, sticky="w", pady=(8, 4))
        ttk.Combobox(
            tab, textvariable=self.level, values=("D1", "D3", "D5"), state="readonly", width=8
        ).grid(row=3, column=0, sticky="w")
        ttk.Label(tab, text="Scale").grid(row=2, column=1, sticky="w", padx=(18, 0), pady=(8, 4))
        ttk.Combobox(
            tab, textvariable=self.scale, values=("minor", "major", "free"), state="readonly", width=10
        ).grid(row=3, column=1, sticky="w", padx=(18, 0))
        ttk.Label(tab, text="Root note").grid(row=2, column=2, sticky="w", padx=(18, 0), pady=(8, 4))
        ttk.Combobox(
            tab,
            textvariable=self.root_note,
            values=tuple(ROOT_NOTE_PITCHES),
            state="readonly",
            width=11,
        ).grid(row=3, column=2, sticky="w", padx=(18, 0))
        ttk.Label(tab, text="Scale strength").grid(
            row=2, column=3, sticky="w", padx=(18, 0), pady=(8, 4)
        )
        ttk.Spinbox(
            tab, from_=0.0, to=1.0, increment=0.05, textvariable=self.confidence, width=8
        ).grid(row=3, column=3, sticky="w", padx=(18, 0))
        buttons = ttk.Frame(tab)
        buttons.grid(row=4, column=0, columnspan=4, sticky="w", pady=20)
        self.generate_button = ttk.Button(buttons, text="Create composition", command=self.generate)
        self.generate_button.pack(side="left")
        ttk.Button(buttons, text="Listen", command=self.listen).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons,
            text="Continue to feedback",
            command=lambda: self.notebook.select(self.feedback_tab),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Output folder", command=self.show_output).pack(side="left", padx=(8, 0))

        result_box = ttk.LabelFrame(tab, text="Composition", padding=12)
        result_box.grid(row=5, column=0, columnspan=4, sticky="ew")
        ttk.Label(result_box, textvariable=self.render_status, wraplength=720, justify="left").pack(
            side="left", fill="x", expand=True
        )
        self.render_details_button = ttk.Button(
            result_box, text="Show technical details", command=self._toggle_render_details
        )
        self.render_details_button.pack(side="right", padx=(12, 0))
        self.render_details = tk.Text(tab, height=13, wrap="word", state="disabled")
        self.render_details.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        self.render_details.grid_remove()
        tab.rowconfigure(6, weight=1)
        tab.columnconfigure(3, weight=1)

    def _build_feedback(self, tab: ttk.Frame) -> None:
        canvas = tk.Canvas(tab, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = ttk.Frame(canvas)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")

        def refresh_scroll_region(_event=None) -> None:
            bounds = canvas.bbox("all")
            if bounds is not None:
                canvas.configure(scrollregion=bounds)

        def fit_content_width(event) -> None:
            canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", refresh_scroll_region)
        canvas.bind("<Configure>", fit_content_width)
        self.feedback_canvas = canvas
        self.feedback_app = FeedbackApp(
            content,
            embedded=True,
            level_var=self.level,
            on_applied=self.feedback_applied,
        )
        self.root.bind("<MouseWheel>", self._scroll_feedback, add="+")

    def _scroll_feedback(self, event) -> None:
        if self.notebook.select() != str(self.feedback_tab) or not event.delta:
            return
        self.feedback_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _build_live(self, tab: ttk.Frame) -> None:
        ttk.Label(tab, text="Live Performance / Max", font=("Helvetica", 20, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text=(
                "Optional. Start this only when you want Max/MSP to request new D1, D3 or D5 renders. "
                "Normal Hyponoia composition works without Max."
            ),
            wraplength=820,
            justify="left",
        ).pack(anchor="w", pady=(6, 18))
        status_box = ttk.LabelFrame(tab, text="Connection status", padding=14)
        status_box.pack(fill="x")
        ttk.Label(status_box, textvariable=self.max_status, wraplength=760, justify="left").pack(anchor="w")
        ttk.Label(
            status_box,
            text="Max sends to 127.0.0.1:7401  •  Hyponoia replies to 127.0.0.1:7402",
        ).pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=18)
        self.start_max_button = ttk.Button(buttons, text="Start live connection", command=self.start_max)
        self.start_max_button.pack(side="left")
        self.stop_max_button = ttk.Button(buttons, text="Stop", command=self.stop_max)
        self.stop_max_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Open Max project", command=self.open_max_project).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Send test to Max", command=self.test_max).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Open connection log", command=self.open_max_log).pack(side="left", padx=(8, 0))
        ttk.Label(
            tab,
            text=(
                "In Max, send /generator/render D1, D3 or D5. Hyponoia returns the complete WAV path "
                "before /generator/ready, so Max can preload it safely."
            ),
            wraplength=820,
            justify="left",
        ).pack(anchor="w")
        self.refresh_max_status()

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def refresh_status(self) -> None:
        info = runtime_status(PROJECT_DIR)
        self.status.set(format_runtime_summary(info))

    def _toggle_render_details(self) -> None:
        self.render_log_visible = not self.render_log_visible
        if self.render_log_visible:
            self.render_details.grid()
            self.render_details_button.configure(text="Hide technical details")
        else:
            self.render_details.grid_remove()
            self.render_details_button.configure(text="Show technical details")

    def refresh_max_status(self) -> None:
        info = self.max_controller.snapshot()
        self.max_status.set(info["message"])
        self.start_max_button.configure(state="disabled" if info["receiver_active"] else "normal")
        self.stop_max_button.configure(state="normal" if info["owned_by_app"] else "disabled")

    def start_max(self) -> None:
        try:
            info = self.max_controller.start()
        except OSError as exc:
            messagebox.showerror("Live connection did not start", str(exc))
            return
        self.max_status.set(info["message"])
        self.refresh_max_status()

    def stop_max(self) -> None:
        self.max_controller.stop()
        self.refresh_max_status()

    def test_max(self) -> None:
        result = self.max_controller.send_test()
        self.max_status.set(
            f"Test sent to Max at {result['destination']}. Check the Max console for /hyponoia/test 1."
        )

    def open_max_project(self) -> None:
        path = find_max_project(PROJECT_DIR)
        if path is None:
            selected = filedialog.askopenfilename(
                title="Choose the Hyponoia Max project",
                filetypes=(("Max project", "*.maxproj"),),
            )
            if not selected:
                return
            path = Path(selected).resolve()
            from hyponoia_runtime import update_user_config

            update_user_config(PROJECT_DIR, max_project=path)
        subprocess.Popen(["open", str(path)])

    def open_max_log(self) -> None:
        path = PROJECT_DIR / "logs" / "max_live.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        subprocess.Popen(["open", str(path)])

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
                root_pitch=ROOT_NOTE_PITCHES[self.root_note.get()],
                scale=self.scale.get(),
                confidence=self.confidence.get(),
                project_dir=PROJECT_DIR,
            )
        except ValueError as exc:
            messagebox.showinfo("Check the settings", str(exc))
            return
        self.generate_button.configure(state="disabled")
        self.render_status.set(f"Creating {self.level.get()}… This may take a moment.")
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
            self.render_status.set(
                f"{self.level.get()} is ready in {self.root_note.get()} {self.scale.get()}. "
                "Listen, then tell Hyponoia what should stay and what should change."
            )
            messagebox.showinfo("Hyponoia", "Your composition is ready. Press ‘Listen’.")
        else:
            self.render_status.set("The composition could not be created. Open the technical details below.")
            if not self.render_log_visible:
                self._toggle_render_details()

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
                "Your ratings and comment were saved. Create again to hear the new preference.",
            )
            self.notebook.select(self.compose_tab)

    def close(self) -> None:
        self.max_controller.stop()
        if hasattr(self, "feedback_app") and self.feedback_app.voice_recorder is not None:
            self.feedback_app.voice_recorder.cancel()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    HyponoiaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
