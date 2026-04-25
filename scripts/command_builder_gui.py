import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from weakref import WeakKeyDictionary


SCRIPT_OPTIONS = (
    "scripts/tapo_opencv_test.py",
    "scripts/tapo_opencv_classifier_test.py",
)

DETECT_MODES = ("always", "motion", "auto")
DEVICE_OPTIONS = ("cuda", "cpu")
PREPROCESS_OPTIONS = ("none", "night-lite", "night")


@dataclass(frozen=True)
class ValueField:
    key: str
    label: str
    default: str = ""
    browse: str | None = None
    width: int = 48
    enabled_by_default: bool = True


@dataclass(frozen=True)
class BoolField:
    key: str
    label: str
    default: bool = False


COMMON_VALUE_FIELDS = (
    ValueField("video", "Video file", browse="file", enabled_by_default=False),
    ValueField("url", "RTSP URL", enabled_by_default=False),
    ValueField("zone_polygon", "Zone polygon", "0.3651,0.5207;0.4317,0.5238;0.4328,0.6251;0.3644,0.6189"),
    ValueField("zone", "Zone rect", "0.35,0.45,0.3,0.3"),
    ValueField("motion_threshold", "Motion threshold", "1.4"),
    ValueField("process_fps", "Process FPS", "5"),
    ValueField("snapshot_cooldown", "Snapshot cooldown", "0"),
    ValueField("alert_seconds", "Alert seconds", "4"),
    ValueField("clip_seconds", "Clip seconds", "10"),
    ValueField("cat_model", "Cat model", "models\\yolov8m.pt", browse="file"),
    ValueField("cat_confidence", "Cat confidence", "0.08"),
    ValueField("cat_enter_frames", "Cat enter frames", "1"),
    ValueField("cat_hold_seconds", "Cat hold seconds", "1.5"),
    ValueField("possible_goblin_seconds", "Possible Goblin seconds", "2.0"),
    ValueField("cat_zone_overlap", "Cat zone overlap", "0.25"),
    ValueField("cat_imgsz", "Cat imgsz", "1920"),
    ValueField("identity_debug_csv", "Identity debug CSV", "", browse="save", enabled_by_default=False),
    ValueField("launch_origin", "Launch origin", "manual"),
    ValueField("extra_args", "Extra args", enabled_by_default=False),
)

ADVANCED_VALUE_FIELDS = (
    ValueField("id_goblin_support_conf", "Goblin support conf", "", enabled_by_default=False),
    ValueField("id_goblin_support_margin", "Goblin support margin", "", enabled_by_default=False),
    ValueField("id_goblin_torso_white_min", "Goblin torso white min", "", enabled_by_default=False),
    ValueField("id_goblin_periphery_margin_max", "Goblin periphery max", "", enabled_by_default=False),
    ValueField("id_lock_margin", "Identity lock margin", "", enabled_by_default=False),
    ValueField("id_switch_margin", "Identity switch margin", "", enabled_by_default=False),
    ValueField("id_confirmed_goblin_support_count", "Confirmed support count", "", enabled_by_default=False),
    ValueField("id_confirmed_goblin_hold_seconds", "Confirmed hold seconds", "", enabled_by_default=False),
)

BOOL_FIELDS = (
    BoolField("use_venv_python", "Use .venv Python", True),
    BoolField("use_video", "Use video source", False),
    BoolField("use_url", "Pass explicit --url", False),
    BoolField("use_zone_polygon", "Use zone polygon", True),
    BoolField("zone_edit", "Enable zone editor", True),
    BoolField("headless", "Headless", False),
    BoolField("loop_video", "Loop video", False),
    BoolField("no_snapshots", "No snapshots", True),
    BoolField("save_clip_on_alert", "Save clip on alert", False),
)


class CommandBuilderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.repo_root = Path(__file__).resolve().parents[1]
        self.dotenv_path = self.repo_root / ".env"
        self.tooltip_path = self.repo_root / "scripts" / "i18n" / "tooltips.json"
        self.value_vars: dict[str, tk.StringVar] = {}
        self.value_enabled_vars: dict[str, tk.BooleanVar] = {}
        self.bool_vars: dict[str, tk.BooleanVar] = {}
        self.script_var = tk.StringVar(value=SCRIPT_OPTIONS[1])
        self.detect_mode_var = tk.StringVar(value="always")
        self.detect_mode_enabled_var = tk.BooleanVar(value=True)
        self.device_var = tk.StringVar(value="cuda")
        self.device_enabled_var = tk.BooleanVar(value=True)
        self.preprocess_var = tk.StringVar(value="none")
        self.preprocess_enabled_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")
        self.command_var = tk.StringVar()
        self.tooltips = self._load_tooltips()
        self.tooltip_window: tk.Toplevel | None = None
        self.tooltip_after_id: str | None = None
        self.is_closing = False
        self.entry_history: WeakKeyDictionary[tk.Entry, list[str]] = WeakKeyDictionary()
        self.entry_history_index: WeakKeyDictionary[tk.Entry, int] = WeakKeyDictionary()
        self.entry_history_job: WeakKeyDictionary[tk.Entry, str] = WeakKeyDictionary()
        self.colors = {
            "bg": "#f6efe5",
            "panel": "#fffaf4",
            "panel_alt": "#fff4e6",
            "border": "#dfc7aa",
            "accent": "#d96b2b",
            "accent_dark": "#a84a1b",
            "text": "#2f241b",
            "muted": "#7a6451",
            "preview_bg": "#2b211b",
            "preview_fg": "#f9efe2",
            "tick_bg": "#fffaf4",
            "tooltip_bg": "#fff7ec",
        }

        self.root.title("Tapo Command Builder")
        self.root.geometry("1040x880")
        self.root.configure(bg=self.colors["bg"])

        self._build_vars()
        self._configure_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_watchdog_from_env()
        self.update_command_preview()

    def _build_vars(self) -> None:
        for field in COMMON_VALUE_FIELDS + ADVANCED_VALUE_FIELDS:
            self.value_vars[field.key] = tk.StringVar(value=field.default)
            self.value_enabled_vars[field.key] = tk.BooleanVar(value=field.enabled_by_default)
            self.value_vars[field.key].trace_add("write", self._on_form_change)
            self.value_enabled_vars[field.key].trace_add("write", self._on_form_change)

        for field in BOOL_FIELDS:
            self.bool_vars[field.key] = tk.BooleanVar(value=field.default)
            self.bool_vars[field.key].trace_add("write", self._on_form_change)

        for variable in (
            self.script_var,
            self.detect_mode_var,
            self.detect_mode_enabled_var,
            self.device_var,
            self.device_enabled_var,
            self.preprocess_var,
            self.preprocess_enabled_var,
        ):
            variable.trace_add("write", self._on_form_change)

    def _load_tooltips(self) -> dict[str, str]:
        if not self.tooltip_path.exists():
            return {}
        try:
            data = json.loads(self.tooltip_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "xpnative" in style.theme_names():
            style.theme_use("xpnative")

        style.configure(
            ".",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["panel"],
        )
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure(
            "Card.TLabelframe",
            background=self.colors["panel"],
            bordercolor=self.colors["border"],
            relief="solid",
            borderwidth=1,
            padding=10,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self.colors["panel"],
            foreground=self.colors["accent_dark"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Header.TFrame",
            background=self.colors["panel_alt"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "HeaderTitle.TLabel",
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            font=("Segoe UI Semibold", 18),
        )
        style.configure(
            "HeaderBody.TLabel",
            background=self.colors["panel_alt"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "SectionHint.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "TEntry",
            padding=6,
            fieldbackground=self.colors["panel"],
            bordercolor=self.colors["border"],
        )
        style.configure(
            "TCombobox",
            padding=4,
            fieldbackground=self.colors["panel"],
            bordercolor=self.colors["border"],
        )
        style.configure(
            "Action.TButton",
            background=self.colors["accent"],
            foreground="#ffffff",
            borderwidth=0,
            padding=(12, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Action.TButton",
            background=[
                ("active", self.colors["accent_dark"]),
                ("pressed", self.colors["accent_dark"]),
            ],
            foreground=[("disabled", "#f2d6c7")],
        )
        style.configure(
            "Secondary.TButton",
            background=self.colors["panel_alt"],
            foreground=self.colors["accent_dark"],
            bordercolor=self.colors["border"],
            padding=(10, 8),
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#f8e2c9")],
            foreground=[("active", self.colors["accent_dark"])],
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, padding=18, style="Header.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Tapo Command Builder", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Build the OpenCV and classifier commands by ticking exactly what you want, then copy or save the final watchdog command.",
            style="HeaderBody.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        canvas = tk.Canvas(
            container,
            highlightthickness=0,
            bg=self.colors["bg"],
            bd=0,
        )
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="App.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.root.bind_all(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
        )

        top_frame = ttk.LabelFrame(scroll_frame, text="Preset", style="Card.TLabelframe")
        top_frame.pack(fill="x", pady=(0, 10))
        top_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(top_frame, text="Script").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        script_combo = ttk.Combobox(
            top_frame,
            textvariable=self.script_var,
            values=SCRIPT_OPTIONS,
            state="readonly",
            width=42,
        )
        script_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        self._attach_tooltip(script_combo, "script")

        load_button = ttk.Button(
            top_frame,
            text="Load .env watchdog",
            command=self._load_watchdog_from_env,
            style="Secondary.TButton",
        )
        load_button.grid(row=0, column=2, sticky="w", padx=8, pady=8)
        self._attach_tooltip(load_button, "load_env_watchdog")
        ttk.Label(
            top_frame,
            text="Start from your current production watchdog command, then adjust only the pieces you want.",
            style="SectionHint.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        flags_frame = ttk.LabelFrame(scroll_frame, text="Quick Toggles", style="Card.TLabelframe")
        flags_frame.pack(fill="x", pady=(0, 10))
        self._build_bool_grid(flags_frame, BOOL_FIELDS, columns=3)
        ttk.Label(
            flags_frame,
            text="These are the fast on/off switches. The checkboxes beside individual fields below control whether those flags are emitted.",
            style="SectionHint.TLabel",
        ).grid(row=(len(BOOL_FIELDS) // 3) + 1, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 2))

        source_frame = ttk.LabelFrame(scroll_frame, text="Source And Zone", style="Card.TLabelframe")
        source_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            source_frame,
            text="Use the left tick on each row to include or exclude that option without losing the saved value.",
            style="SectionHint.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        self._build_value_rows(
            source_frame,
            (
                self._field_by_key("video"),
                self._field_by_key("url"),
                self._field_by_key("zone_polygon"),
                self._field_by_key("zone"),
            ),
        )

        detection_frame = ttk.LabelFrame(scroll_frame, text="Detection", style="Card.TLabelframe")
        detection_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            detection_frame,
            text="Core runtime knobs for motion, snapshots, alert timing, YOLO detection, and replay debugging.",
            style="SectionHint.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        self._build_value_rows(
            detection_frame,
            (
                self._field_by_key("motion_threshold"),
                self._field_by_key("process_fps"),
                self._field_by_key("snapshot_cooldown"),
                self._field_by_key("alert_seconds"),
                self._field_by_key("clip_seconds"),
                self._field_by_key("cat_model"),
                self._field_by_key("cat_confidence"),
                self._field_by_key("cat_enter_frames"),
                self._field_by_key("cat_hold_seconds"),
                self._field_by_key("possible_goblin_seconds"),
                self._field_by_key("cat_zone_overlap"),
                self._field_by_key("cat_imgsz"),
                self._field_by_key("identity_debug_csv"),
                self._field_by_key("launch_origin"),
            ),
        )

        option_frame = ttk.Frame(detection_frame, style="App.TFrame")
        option_frame.grid_columnconfigure(2, weight=1)
        option_frame.pack(fill="x", padx=8, pady=(4, 8))

        detect_check = self._make_tickbox(option_frame, self.detect_mode_enabled_var)
        detect_check.grid(row=0, column=0, sticky="w", pady=4)
        self._attach_tooltip(detect_check, "cat_detect_mode")
        detect_label = ttk.Label(option_frame, text="Cat detect mode", style="SectionHint.TLabel")
        detect_label.grid(row=0, column=1, sticky="w", pady=4)
        self._attach_tooltip(detect_label, "cat_detect_mode")
        detect_combo = ttk.Combobox(
            option_frame,
            textvariable=self.detect_mode_var,
            values=DETECT_MODES,
            state="readonly",
            width=18,
        )
        detect_combo.grid(row=0, column=2, sticky="w", pady=4)
        self._attach_tooltip(detect_combo, "cat_detect_mode")

        preprocess_check = self._make_tickbox(option_frame, self.preprocess_enabled_var)
        preprocess_check.grid(row=1, column=0, sticky="w", pady=4)
        self._attach_tooltip(preprocess_check, "cat_preprocess")
        preprocess_label = ttk.Label(option_frame, text="Cat preprocess", style="SectionHint.TLabel")
        preprocess_label.grid(row=1, column=1, sticky="w", pady=4)
        self._attach_tooltip(preprocess_label, "cat_preprocess")
        preprocess_combo = ttk.Combobox(
            option_frame,
            textvariable=self.preprocess_var,
            values=PREPROCESS_OPTIONS,
            state="readonly",
            width=18,
        )
        preprocess_combo.grid(row=1, column=2, sticky="w", pady=4)
        self._attach_tooltip(preprocess_combo, "cat_preprocess")

        device_check = self._make_tickbox(option_frame, self.device_enabled_var)
        device_check.grid(row=2, column=0, sticky="w", pady=4)
        self._attach_tooltip(device_check, "device")
        device_label = ttk.Label(option_frame, text="Device", style="SectionHint.TLabel")
        device_label.grid(row=2, column=1, sticky="w", pady=4)
        self._attach_tooltip(device_label, "device")
        device_combo = ttk.Combobox(
            option_frame,
            textvariable=self.device_var,
            values=DEVICE_OPTIONS,
            state="readonly",
            width=18,
        )
        device_combo.grid(row=2, column=2, sticky="w", pady=4)
        self._attach_tooltip(device_combo, "device")

        advanced_frame = ttk.LabelFrame(scroll_frame, text="Advanced Identity Tuning", style="Card.TLabelframe")
        advanced_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            advanced_frame,
            text="Keep these unticked for normal use. Turn them on only when you want to override the current identity tuning.",
            style="SectionHint.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        self._build_value_rows(advanced_frame, ADVANCED_VALUE_FIELDS)

        extra_frame = ttk.LabelFrame(scroll_frame, text="Extra Args", style="Card.TLabelframe")
        extra_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            extra_frame,
            text="Anything uncommon can still be appended here as raw CLI arguments.",
            style="SectionHint.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        self._build_value_rows(extra_frame, (self._field_by_key("extra_args"),))

        command_frame = ttk.LabelFrame(scroll_frame, text="Generated Command", style="Card.TLabelframe")
        command_frame.pack(fill="both", expand=True)
        ttk.Label(
            command_frame,
            text="Preview the exact command string that will be copied or written into `.env`.",
            style="SectionHint.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        command_text = tk.Text(
            command_frame,
            height=10,
            wrap="word",
            bg=self.colors["preview_bg"],
            fg=self.colors["preview_fg"],
            insertbackground=self.colors["preview_fg"],
            relief="flat",
            padx=12,
            pady=12,
            font=("Consolas", 10),
        )
        command_text.pack(fill="both", expand=True, padx=8, pady=8)
        command_text.configure(state="disabled")
        self.command_text = command_text
        self._attach_tooltip(command_text, "generated_command")

        action_frame = ttk.Frame(scroll_frame, style="App.TFrame")
        action_frame.pack(fill="x", pady=(10, 4))

        refresh_button = ttk.Button(
            action_frame,
            text="Refresh",
            command=self.update_command_preview,
            style="Secondary.TButton",
        )
        refresh_button.pack(side="left", padx=(0, 8))
        self._attach_tooltip(refresh_button, "refresh")

        copy_button = ttk.Button(
            action_frame,
            text="Copy Command",
            command=self.copy_command,
            style="Action.TButton",
        )
        copy_button.pack(side="left", padx=(0, 8))
        self._attach_tooltip(copy_button, "copy_command")

        write_button = ttk.Button(
            action_frame,
            text="Write To .env",
            command=self.write_watchdog_command,
            style="Action.TButton",
        )
        write_button.pack(side="left", padx=(0, 8))
        self._attach_tooltip(write_button, "write_to_env")

        status_card = tk.Label(
            action_frame,
            textvariable=self.status_var,
            bg=self.colors["panel_alt"],
            fg=self.colors["accent_dark"],
            padx=12,
            pady=8,
            font=("Segoe UI Semibold", 9),
            relief="solid",
            bd=1,
        )
        status_card.pack(side="right")

    def _bind_shortcuts(self) -> None:
        self.root.bind_class("TEntry", "<Control-z>", self._handle_entry_undo, add="+")
        self.root.bind_class("TEntry", "<Control-y>", self._handle_entry_redo, add="+")
        self.root.bind_class("Entry", "<Control-z>", self._handle_entry_undo, add="+")
        self.root.bind_class("Entry", "<Control-y>", self._handle_entry_redo, add="+")
        self.root.bind_class("Text", "<Control-z>", self._handle_text_undo, add="+")
        self.root.bind_class("Text", "<Control-y>", self._handle_text_redo, add="+")

    def _handle_entry_undo(self, event: tk.Event) -> str:
        widget = event.widget
        if isinstance(widget, tk.Entry):
            self._entry_undo(widget)
        return "break"

    def _handle_entry_redo(self, event: tk.Event) -> str:
        widget = event.widget
        if isinstance(widget, tk.Entry):
            self._entry_redo(widget)
        return "break"

    def _handle_text_undo(self, event: tk.Event) -> str:
        widget = event.widget
        if isinstance(widget, tk.Text):
            try:
                widget.edit_undo()
            except tk.TclError:
                pass
        return "break"

    def _handle_text_redo(self, event: tk.Event) -> str:
        widget = event.widget
        if isinstance(widget, tk.Text):
            try:
                widget.edit_redo()
            except tk.TclError:
                pass
        return "break"

    def _register_entry_history(self, entry: tk.Entry) -> None:
        current = entry.get()
        self.entry_history[entry] = [current]
        self.entry_history_index[entry] = 0
        entry.bind("<KeyRelease>", lambda _event, widget=entry: self._schedule_entry_snapshot(widget), add="+")
        entry.bind("<FocusIn>", lambda _event, widget=entry: self._ensure_entry_state(widget), add="+")

    def _ensure_entry_state(self, entry: tk.Entry) -> None:
        if entry not in self.entry_history:
            current = entry.get()
            self.entry_history[entry] = [current]
            self.entry_history_index[entry] = 0

    def _schedule_entry_snapshot(self, entry: tk.Entry) -> None:
        self._ensure_entry_state(entry)
        pending = self.entry_history_job.get(entry)
        if pending:
            try:
                entry.after_cancel(pending)
            except tk.TclError:
                pass
        job = entry.after(250, lambda widget=entry: self._snapshot_entry_state(widget))
        self.entry_history_job[entry] = job

    def _snapshot_entry_state(self, entry: tk.Entry) -> None:
        self.entry_history_job.pop(entry, None)
        self._ensure_entry_state(entry)
        current = entry.get()
        history = self.entry_history[entry]
        index = self.entry_history_index[entry]
        if history[index] == current:
            return
        if index < len(history) - 1:
            del history[index + 1 :]
        history.append(current)
        self.entry_history_index[entry] = len(history) - 1

    def _entry_undo(self, entry: tk.Entry) -> None:
        self._flush_entry_snapshot(entry)
        index = self.entry_history_index[entry]
        if index <= 0:
            return
        index -= 1
        self.entry_history_index[entry] = index
        self._set_entry_text(entry, self.entry_history[entry][index])

    def _entry_redo(self, entry: tk.Entry) -> None:
        self._flush_entry_snapshot(entry)
        history = self.entry_history[entry]
        index = self.entry_history_index[entry]
        if index >= len(history) - 1:
            return
        index += 1
        self.entry_history_index[entry] = index
        self._set_entry_text(entry, history[index])

    def _flush_entry_snapshot(self, entry: tk.Entry) -> None:
        self._ensure_entry_state(entry)
        pending = self.entry_history_job.pop(entry, None)
        if pending:
            try:
                entry.after_cancel(pending)
            except tk.TclError:
                pass
        self._snapshot_entry_state(entry)

    def _set_entry_text(self, entry: tk.Entry, value: str) -> None:
        try:
            entry.delete(0, tk.END)
            entry.insert(0, value)
            entry.icursor(tk.END)
        except tk.TclError:
            pass

    def _field_by_key(self, key: str) -> ValueField:
        for field in COMMON_VALUE_FIELDS + ADVANCED_VALUE_FIELDS:
            if field.key == key:
                return field
        raise KeyError(key)

    def _build_bool_grid(self, parent: ttk.Widget, fields: tuple[BoolField, ...], columns: int) -> None:
        for index, field in enumerate(fields):
            row = index // columns
            column = index % columns
            widget = self._make_tickbox(parent, self.bool_vars[field.key], text=field.label)
            widget.grid(row=row, column=column, sticky="w", padx=8, pady=6)
            self._attach_tooltip(widget, field.key)

    def _build_value_rows(self, parent: ttk.Widget, fields: tuple[ValueField, ...]) -> None:
        frame = ttk.Frame(parent, style="App.TFrame")
        frame.pack(fill="x", padx=8, pady=8)
        frame.grid_columnconfigure(2, weight=1)

        for row, field in enumerate(fields):
            checkbox = self._make_tickbox(frame, self.value_enabled_vars[field.key])
            checkbox.grid(row=row, column=0, sticky="w", pady=4)
            self._attach_tooltip(checkbox, field.key)

            label = ttk.Label(frame, text=field.label, style="SectionHint.TLabel")
            label.grid(row=row, column=1, sticky="w", pady=4, padx=(4, 0))
            self._attach_tooltip(label, field.key)

            entry = tk.Entry(
                frame,
                textvariable=self.value_vars[field.key],
                width=field.width,
                bg=self.colors["panel"],
                fg=self.colors["text"],
                insertbackground=self.colors["text"],
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground=self.colors["border"],
                highlightcolor=self.colors["accent"],
                font=("Segoe UI", 10),
            )
            entry.grid(row=row, column=2, sticky="ew", pady=4, padx=(8, 8))
            self._attach_tooltip(entry, field.key)
            self._register_entry_history(entry)

            if field.browse == "file":
                browse_button = ttk.Button(
                    frame,
                    text="Browse",
                    command=lambda key=field.key: self.browse_file(key),
                    style="Secondary.TButton",
                )
                browse_button.grid(row=row, column=3, sticky="w", pady=4)
                self._attach_tooltip(browse_button, field.key)
            elif field.browse == "save":
                browse_button = ttk.Button(
                    frame,
                    text="Browse",
                    command=lambda key=field.key: self.browse_save_file(key),
                    style="Secondary.TButton",
                )
                browse_button.grid(row=row, column=3, sticky="w", pady=4)
                self._attach_tooltip(browse_button, field.key)

    def _make_tickbox(
        self,
        parent: tk.Misc,
        variable: tk.BooleanVar,
        text: str = "",
    ) -> tk.Checkbutton:
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            bg=self.colors["panel"],
            activebackground=self.colors["panel"],
            fg=self.colors["text"],
            activeforeground=self.colors["accent_dark"],
            selectcolor=self.colors["tick_bg"],
            font=("Segoe UI", 9),
            highlightthickness=0,
            bd=0,
            padx=2,
            pady=2,
            anchor="w",
        )

    def _attach_tooltip(self, widget: tk.Misc, key: str) -> None:
        text = self.tooltips.get(key, "").strip()
        if not text:
            return
        widget.bind("<Enter>", lambda event, tooltip_text=text: self._schedule_tooltip(event, tooltip_text), add="+")
        widget.bind("<Leave>", lambda _event: self._hide_tooltip(), add="+")
        widget.bind("<ButtonPress>", lambda _event: self._hide_tooltip(), add="+")

    def _schedule_tooltip(self, event: tk.Event, text: str) -> None:
        if self.is_closing:
            return
        self._hide_tooltip()
        self.tooltip_after_id = event.widget.after(450, lambda: self._show_tooltip(event.widget, text))

    def _show_tooltip(self, widget: tk.Misc, text: str) -> None:
        if self.is_closing:
            return
        self._hide_tooltip()
        if not widget.winfo_exists() or not self.root.winfo_exists():
            return
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.attributes("-topmost", True)

        x = widget.winfo_rootx() + 18
        y = widget.winfo_rooty() + widget.winfo_height() + 10
        tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tooltip,
            text=text,
            justify="left",
            wraplength=360,
            bg=self.colors["tooltip_bg"],
            fg=self.colors["text"],
            relief="solid",
            bd=1,
            padx=10,
            pady=8,
            font=("Segoe UI", 9),
        )
        label.pack()
        self.tooltip_window = tooltip

    def _hide_tooltip(self) -> None:
        if self.tooltip_after_id is not None:
            try:
                self.root.after_cancel(self.tooltip_after_id)
            except (ValueError, tk.TclError):
                pass
            self.tooltip_after_id = None
        if self.tooltip_window is not None:
            try:
                if self.tooltip_window.winfo_exists():
                    self.tooltip_window.destroy()
            except tk.TclError:
                pass
            self.tooltip_window = None

    def _on_close(self) -> None:
        self.is_closing = True
        self._hide_tooltip()
        try:
            self.root.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass
        try:
            self.root.withdraw()
        except tk.TclError:
            pass
        try:
            self.root.after_idle(self._destroy_root_safely)
        except tk.TclError:
            pass

    def _destroy_root_safely(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _on_form_change(self, *_args: object) -> None:
        self.update_command_preview()

    def browse_file(self, key: str) -> None:
        selected = filedialog.askopenfilename(initialdir=str(self.repo_root))
        if selected:
            path = Path(selected)
            try:
                value = str(path.relative_to(self.repo_root))
            except ValueError:
                value = str(path)
            self.value_vars[key].set(value)

    def browse_save_file(self, key: str) -> None:
        selected = filedialog.asksaveasfilename(initialdir=str(self.repo_root))
        if selected:
            path = Path(selected)
            try:
                value = str(path.relative_to(self.repo_root))
            except ValueError:
                value = str(path)
            self.value_vars[key].set(value)

    def build_command_tokens(self) -> list[str]:
        tokens: list[str] = []

        if self.bool_vars["use_venv_python"].get():
            tokens.append(r".\.venv\Scripts\python.exe")
        else:
            tokens.append("python")

        tokens.append(self.script_var.get())

        if self.bool_vars["use_video"].get():
            self._append_value_arg(tokens, "--video", "video", require_enabled=False)
            if self.bool_vars["loop_video"].get():
                tokens.append("--loop-video")
        elif self.bool_vars["use_url"].get():
            self._append_value_arg(tokens, "--url", "url", require_enabled=False)

        if self.bool_vars["headless"].get():
            tokens.append("--headless")

        if self.bool_vars["use_zone_polygon"].get():
            self._append_value_arg(tokens, "--zone-polygon", "zone_polygon", require_enabled=False)
        else:
            self._append_value_arg(tokens, "--zone", "zone", require_enabled=False)

        if self.bool_vars["zone_edit"].get():
            tokens.append("--zone-edit")

        self._append_value_arg(tokens, "--motion-threshold", "motion_threshold")
        self._append_value_arg(tokens, "--process-fps", "process_fps")
        self._append_value_arg(tokens, "--snapshot-cooldown", "snapshot_cooldown")

        if self.bool_vars["no_snapshots"].get():
            tokens.append("--no-snapshots")

        self._append_value_arg(tokens, "--alert-seconds", "alert_seconds")

        if self.bool_vars["save_clip_on_alert"].get():
            tokens.append("--save-clip-on-alert")
            self._append_value_arg(tokens, "--clip-seconds", "clip_seconds")

        self._append_value_arg(tokens, "--cat-model", "cat_model")
        self._append_value_arg(tokens, "--cat-confidence", "cat_confidence")
        self._append_value_arg(tokens, "--cat-enter-frames", "cat_enter_frames")
        self._append_value_arg(tokens, "--cat-hold-seconds", "cat_hold_seconds")
        self._append_value_arg(tokens, "--possible-goblin-seconds", "possible_goblin_seconds")
        self._append_value_arg(tokens, "--cat-zone-overlap", "cat_zone_overlap")
        self._append_value_arg(tokens, "--cat-imgsz", "cat_imgsz")

        if self.preprocess_enabled_var.get() and self.preprocess_var.get():
            tokens.extend(["--cat-preprocess", self.preprocess_var.get()])
        if self.detect_mode_enabled_var.get() and self.detect_mode_var.get():
            tokens.extend(["--cat-detect-mode", self.detect_mode_var.get()])
        if self.device_enabled_var.get() and self.device_var.get():
            tokens.extend(["--device", self.device_var.get()])

        self._append_value_arg(tokens, "--identity-debug-csv", "identity_debug_csv")
        self._append_value_arg(tokens, "--launch-origin", "launch_origin")

        advanced_map = (
            ("--id-goblin-support-conf", "id_goblin_support_conf"),
            ("--id-goblin-support-margin", "id_goblin_support_margin"),
            ("--id-goblin-torso-white-min", "id_goblin_torso_white_min"),
            ("--id-goblin-periphery-margin-max", "id_goblin_periphery_margin_max"),
            ("--id-lock-margin", "id_lock_margin"),
            ("--id-switch-margin", "id_switch_margin"),
            ("--id-confirmed-goblin-support-count", "id_confirmed_goblin_support_count"),
            ("--id-confirmed-goblin-hold-seconds", "id_confirmed_goblin_hold_seconds"),
        )
        for flag, key in advanced_map:
            self._append_value_arg(tokens, flag, key)

        extra_args = self.value_vars["extra_args"].get().strip()
        if extra_args:
            tokens.extend(shlex.split(extra_args, posix=False))

        return tokens

    def _append_value_arg(
        self,
        tokens: list[str],
        flag: str,
        key: str,
        *,
        require_enabled: bool = True,
    ) -> None:
        if require_enabled and not self.value_enabled_vars[key].get():
            return
        value = self.value_vars[key].get().strip()
        if value:
            tokens.extend([flag, value])

    def build_command_string(self) -> str:
        return " ".join(self._quote_powershell_arg(token) for token in self.build_command_tokens())

    def _quote_powershell_arg(self, arg: str) -> str:
        if not arg:
            return "''"
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/\\:=+")
        if all(char in safe_chars for char in arg):
            return arg
        return "'" + arg.replace("'", "''") + "'"

    def update_command_preview(self) -> None:
        command = self.build_command_string()
        self.command_var.set(command)
        self.command_text.configure(state="normal")
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert("1.0", command)
        self.command_text.configure(state="disabled")

    def copy_command(self) -> None:
        command = self.build_command_string()
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        self.status_var.set("Command copied to clipboard.")

    def write_watchdog_command(self) -> None:
        command = self.build_command_string()
        lines: list[str] = []
        replaced = False

        if self.dotenv_path.exists():
            lines = self.dotenv_path.read_text(encoding="utf-8").splitlines()

        updated_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("WATCHDOG_TAPO_COMMAND="):
                updated_lines.append(f"WATCHDOG_TAPO_COMMAND={command}")
                replaced = True
            else:
                updated_lines.append(line)

        if not replaced:
            if updated_lines and updated_lines[-1].strip():
                updated_lines.append("")
            updated_lines.append(f"WATCHDOG_TAPO_COMMAND={command}")

        self.dotenv_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        self.status_var.set("Saved to .env as WATCHDOG_TAPO_COMMAND.")
        messagebox.showinfo("Saved", "WATCHDOG_TAPO_COMMAND was written to .env")

    def _load_watchdog_from_env(self) -> None:
        if not self.dotenv_path.exists():
            self.status_var.set(".env not found. Using defaults.")
            return

        command = ""
        for raw_line in self.dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("WATCHDOG_TAPO_COMMAND="):
                command = line.split("=", 1)[1].strip()
                break

        if not command:
            self.status_var.set("WATCHDOG_TAPO_COMMAND not found in .env. Using defaults.")
            return

        try:
            self._apply_command_string(command)
            self.status_var.set("Loaded WATCHDOG_TAPO_COMMAND from .env.")
        except Exception as exc:
            self.status_var.set(f"Could not fully parse .env command: {exc}")

    def _apply_command_string(self, command: str) -> None:
        tokens = shlex.split(command, posix=False)
        if not tokens:
            return

        token_index = 0
        first = tokens[0].lower()
        use_venv = first.endswith("python.exe") and ".venv" in first
        if use_venv or first == "python":
            self.bool_vars["use_venv_python"].set(use_venv)
            token_index = 1

        if token_index < len(tokens) and tokens[token_index] in SCRIPT_OPTIONS:
            self.script_var.set(tokens[token_index])
            token_index += 1

        parsed: dict[str, str | bool] = {}
        while token_index < len(tokens):
            token = tokens[token_index]
            if not token.startswith("--"):
                token_index += 1
                continue

            key = token[2:].replace("-", "_")
            next_index = token_index + 1
            if next_index < len(tokens) and not tokens[next_index].startswith("--"):
                parsed[key] = tokens[next_index]
                token_index += 2
            else:
                parsed[key] = True
                token_index += 1

        self.bool_vars["use_video"].set("video" in parsed)
        self.bool_vars["use_url"].set("url" in parsed)
        self.bool_vars["loop_video"].set(bool(parsed.get("loop_video", False)))
        self.bool_vars["headless"].set(bool(parsed.get("headless", False)))
        self.bool_vars["use_zone_polygon"].set("zone_polygon" in parsed)
        self.bool_vars["zone_edit"].set(bool(parsed.get("zone_edit", False)))
        self.bool_vars["no_snapshots"].set(bool(parsed.get("no_snapshots", False)))
        self.bool_vars["save_clip_on_alert"].set(bool(parsed.get("save_clip_on_alert", False)))
        self.detect_mode_enabled_var.set("cat_detect_mode" in parsed)
        self.preprocess_enabled_var.set("cat_preprocess" in parsed)
        self.device_enabled_var.set("device" in parsed)

        simple_value_keys = {
            "video",
            "url",
            "zone_polygon",
            "zone",
            "motion_threshold",
            "process_fps",
            "snapshot_cooldown",
            "alert_seconds",
            "clip_seconds",
            "cat_model",
            "cat_confidence",
            "cat_enter_frames",
            "cat_hold_seconds",
            "possible_goblin_seconds",
            "cat_zone_overlap",
            "cat_imgsz",
            "identity_debug_csv",
            "launch_origin",
            "id_goblin_support_conf",
            "id_goblin_support_margin",
            "id_goblin_torso_white_min",
            "id_goblin_periphery_margin_max",
            "id_lock_margin",
            "id_switch_margin",
            "id_confirmed_goblin_support_count",
            "id_confirmed_goblin_hold_seconds",
            "extra_args",
        }

        for key in simple_value_keys:
            self.value_enabled_vars[key].set(key in parsed)
            if key in parsed and isinstance(parsed[key], str) and key in self.value_vars:
                self.value_vars[key].set(str(parsed[key]))

        if isinstance(parsed.get("cat_detect_mode"), str):
            self.detect_mode_var.set(str(parsed["cat_detect_mode"]))
        if isinstance(parsed.get("cat_preprocess"), str):
            self.preprocess_var.set(str(parsed["cat_preprocess"]))
        if isinstance(parsed.get("device"), str):
            self.device_var.set(str(parsed["device"]))


def main() -> None:
    root = tk.Tk()
    app = CommandBuilderApp(root)
    app.update_command_preview()
    root.mainloop()


if __name__ == "__main__":
    main()
