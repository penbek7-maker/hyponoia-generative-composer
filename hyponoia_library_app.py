"""Small non-technical desktop window for updating a Hyponoia WAV library."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from update_library_v1 import update_library


PROJECT_DIR = Path(__file__).resolve().parent


class LibraryApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hyponoia — Update Library")
        self.root.geometry("720x560")
        self.root.minsize(620, 500)
        self.folder = tk.StringVar()
        self.status = tk.StringVar(value="Choose a folder containing your WAV files.")
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Hyponoia Library", font=("Helvetica", 24, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=(
                "Choose the folder that contains the sounds you want Hyponoia to use. "
                "About 100 recordings are recommended, but you can start with fewer and add more later."
            ),
            wraplength=650,
            justify="left",
        ).pack(anchor="w", pady=(8, 20))

        chooser = ttk.Frame(frame)
        chooser.pack(fill="x")
        ttk.Entry(chooser, textvariable=self.folder).pack(side="left", fill="x", expand=True)
        ttk.Button(chooser, text="Choose folder…", command=self.choose_folder).pack(side="left", padx=(10, 0))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=18)
        self.preview_button = ttk.Button(buttons, text="1. Preview changes", command=lambda: self.run(False))
        self.preview_button.pack(side="left")
        self.update_button = ttk.Button(buttons, text="2. Update library", command=lambda: self.run(True))
        self.update_button.pack(side="left", padx=(10, 0))

        ttk.Separator(frame).pack(fill="x", pady=(0, 16))
        ttk.Label(frame, textvariable=self.status, wraplength=650, justify="left").pack(anchor="w")
        self.details = tk.Text(frame, height=16, wrap="word", state="disabled")
        self.details.pack(fill="both", expand=True, pady=(12, 0))

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="Choose your Hyponoia WAV folder")
        if selected:
            self.folder.set(selected)
            self.status.set("Folder selected. Preview the changes before updating.")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.preview_button.configure(state=state)
        self.update_button.configure(state=state)

    def _show(self, result: dict) -> None:
        plan = result.get("plan", {}).get("summary", {})
        lines = [
            f"Status: {result.get('status', 'unknown')}",
            f"Active WAV files: {plan.get('active_wavs', 0)}",
            f"Added: {plan.get('added', 0)}",
            f"Changed: {plan.get('modified', 0)}",
            f"Removed from active memory: {plan.get('removed', 0)}",
            f"Renamed: {plan.get('renamed', 0)}",
            f"Unchanged: {plan.get('unchanged', 0)}",
        ]
        report = result.get("build_report")
        if report:
            lines.extend(
                [
                    "",
                    f"Memory recordings: {report.get('recordings', 0)}",
                    f"Sound objects: {report.get('sound_objects', 0)}",
                    f"Reused analyses: {len(report.get('reused_recordings', []))}",
                    f"New analyses: {len(report.get('analysed_recordings', []))}",
                ]
            )
        if result.get("backup_folder"):
            lines.extend(["", f"Previous version backup: {result['backup_folder']}"])
        if result.get("error"):
            lines.extend(["", f"Problem: {result['error']}"])
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")
        if result.get("status") == "updated":
            self.status.set("Library updated. Hyponoia will use this folder on the next composition.")
        elif result.get("status") == "preview":
            self.status.set("Preview ready. If the counts look right, press Update library.")
        else:
            self.status.set("Nothing was changed. Read the problem below and try again.")

    def run(self, apply: bool) -> None:
        path = Path(self.folder.get()).expanduser()
        if not path.is_dir():
            messagebox.showinfo("Choose a folder", "Please choose a folder containing WAV files first.")
            return
        self._set_busy(True)
        self.status.set("Updating the musical memory…" if apply else "Checking the folder…")

        def work() -> None:
            try:
                result = update_library(path, PROJECT_DIR, apply=apply)
            except Exception as exc:  # Keep the window alive and preserve current data.
                result = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
            self.root.after(0, lambda: (self._show(result), self._set_busy(False)))

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    LibraryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
