from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from iso_tool import BuildPipeline


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ISO-Tool — GitHub source to ISO / IMG')
        self.geometry('980x700')
        self.repo = tk.StringVar()
        self.out = tk.StringVar(value='output.iso')
        self.progress = tk.DoubleVar()
        self.status = tk.StringVar(value='Ready')
        self.events = queue.Queue()

        ttk.Label(self, text='GitHub repository / local checkout').pack(anchor='w', padx=12, pady=(12, 2))
        ttk.Entry(self, textvariable=self.repo).pack(fill='x', padx=12)
        ttk.Label(self, text='Output image').pack(anchor='w', padx=12, pady=(8, 2))
        ttk.Entry(self, textvariable=self.out).pack(fill='x', padx=12)

        buttons = ttk.Frame(self)
        buttons.pack(fill='x', padx=12, pady=10)
        self.analyze_button = ttk.Button(buttons, text='Inventory / Build Plan', command=self.inventory)
        self.analyze_button.pack(side='left')
        self.clear_button = ttk.Button(buttons, text='Clear details', command=self.clear_log)
        self.clear_button.pack(side='left', padx=8)

        ttk.Label(self, textvariable=self.status).pack(anchor='w', padx=12)
        ttk.Progressbar(self, variable=self.progress, maximum=100).pack(fill='x', padx=12, pady=(4, 8))
        ttk.Label(self, text='Live operation details').pack(anchor='w', padx=12)
        self.log = tk.Text(self, height=24, state='disabled')
        self.log.pack(fill='both', expand=True, padx=12, pady=(4, 12))
        self.after(75, self._drain_events)

    def clear_log(self):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')

    def _append(self, message):
        self.log.configure(state='normal')
        self.log.insert('end', message + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'progress':
                    event = payload
                    percent = 100.0 if event.total == 0 else (event.completed / event.total) * 100.0
                    self.progress.set(max(self.progress.get(), percent))
                    self.status.set(event.message)
                    self._append(f'[{event.stage}] {event.message}')
                elif kind == 'log':
                    self._append(payload)
                elif kind == 'done':
                    self.analyze_button.configure(state='normal')
                    self.status.set(payload)
                    self._append(payload)
        except queue.Empty:
            pass
        self.after(75, self._drain_events)

    def inventory(self):
        root = Path(self.repo.get()).expanduser()
        if not root.is_dir():
            messagebox.showerror('ISO-Tool', 'Choose a local checkout for this reference build.')
            return
        self.analyze_button.configure(state='disabled')
        self.progress.set(0)
        self._append('Starting inventory. Runtime errors will be recorded and processing will continue.')
        threading.Thread(target=self._inventory_worker, args=(root,), daemon=True).start()

    def _inventory_worker(self, root: Path):
        pipeline = BuildPipeline(root)
        try:
            files = pipeline.inventory()
            total = len(files)
            self.events.put(('progress', type('P', (), {
                'stage': 'inventory', 'completed': 0, 'total': max(total, 1),
                'message': f'Found {total} source files; inspecting entries...'
            })()))
            for index, path in enumerate(files, 1):
                try:
                    relative = path.relative_to(root)
                    self.events.put(('log', f'[{index}/{total}] inventory: {relative}'))
                except Exception as exc:
                    self.events.put(('log', f'[error] inventory entry skipped: {type(exc).__name__}: {exc}'))
                self.events.put(('progress', type('P', (), {
                    'stage': 'inventory', 'completed': index, 'total': max(total, 1),
                    'message': f'Inspected {index}/{total}'
                })()))
            self.events.put(('done', f'Inventory complete: {total} source files discovered.'))
        except Exception as exc:
            self.events.put(('log', f'[error] top-level inventory failure: {type(exc).__name__}: {exc}'))
            self.events.put(('done', 'Inventory ended with recoverable errors; application remains available.'))


if __name__ == '__main__':
    App().mainloop()
