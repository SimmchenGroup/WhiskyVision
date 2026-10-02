"""
stitch_gui.py — point-and-click stitcher for tiled images of circular objects.

Double-click "Start Stitcher.bat" (or run `python stitch_gui.py`), then:
  1. choose the folder that holds the tiles
  2. click each sample: "Before" shows its tiles in stitch order,
     "After" shows a quick preview of the stitch
  3. adjust Rows / Columns / Order under Options if the preview looks wrong
  4. press "Stitch"
Results are written to <folder>/Stitched/<sample>_stitched.tif.

Look: the "Organic" design (cream + terracotta, Caprasimo / Figtree), layout 1c
"Before | after" from the Claude Design mockups. Built on CustomTkinter.
"""
import glob
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageTk

import stitcher_core as core

APP_TITLE = "Whisky Webs Stitcher"
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
TILE_THUMB = 1000      # max edge of cached tile thumbnails (px)
RESULT_THUMB = 2000    # max edge of cached saved-result thumbnails (px)
QUICK_EDGE = 480       # tiles are shrunk to this for the quick preview stitch
ZOOMS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

# Organic palette (from the mockup's design system)
C = dict(
    bg="#f5ead8", surface="#ebddc5", well="#2e2b25", text="#201e1d", soft="#474238",
    muted="#645c50", line="#d3c5ad", accent="#c67139", accent_hover="#b0602c",
    accent_deep="#8c491a", accent_tint="#ffe1d0", accent_ink="#643312",
    sage="#7a8a5e", sage_ink="#3d472b", order="#f6a06b", band="#ccdbb2",
    well_text="#c0b6a5", well_faint="#a19786", seg="#474238", track="#dcd3c4",
    error="#b3261e", hover_tint="#e2d1b4",
)

HELP_TEXT = """\
HOW TO USE

1. Click "Browse…" and choose the folder that holds your image tiles.

2. Every sample in the folder is listed under Samples. Click one:
     Before – its tiles, numbered and joined in the order they were taken,
              with the overlapping strips shaded green
     After  – a quick preview of what the stitched image will look like
   Use − / + (or the mouse wheel) to zoom, and drag to move around. Both
   pictures zoom and move together.

3. If the After picture doesn't look like your object, change Rows,
   Columns or Order under Options › This sample. Both pictures update
   straight away. Changes apply to every selected sample (Ctrl/Shift-click
   to pick several).

4. Press "Stitch". Finished images are saved in a new "Stitched" folder
   inside your tile folder. Your original tiles are never changed.
   Switch After to "Saved" to look at a finished stitch.

FILE NAMES

Tiles must end in "_<number>", e.g.
    S24-1273_4X_1uL_01.tiff, S24-1273_4X_1uL_02.tiff, …
Everything before the last "_" is the sample name, so all tiles of one
object must share it.

TILE ORDER  (always starts at the top-left tile)

  Snake by rows     → → →  then  ← ← ←  then  → → → …  (most microscopes)
  Row by row        → → →  then  → → →  …
  Snake by columns  ↓ ↓ ↓  then  ↑ ↑ ↑  …
  Column by column  ↓ ↓ ↓  then  ↓ ↓ ↓  …
For 2-tile samples the direction (side by side or one above the other)
is detected automatically.

ENGINE

  ImageJ (Fiji) – the standard ImageJ "Grid/Collection" and "Pairwise"
                  stitching plugins, run in the background. Needs Fiji.
  Python        – a built-in stitcher that doesn't need Fiji.

If a stitched image looks wrong, try a different Order, or set Overlap to
roughly how much neighbouring tiles overlap (20 % is typical)."""


# --------------------------------------------------------------------------- #
#  Fonts
# --------------------------------------------------------------------------- #
def load_bundled_fonts():
    """Make the bundled Caprasimo / Figtree fonts available to this process only
    (nothing is installed on the computer). Must run before the window opens."""
    paths = glob.glob(os.path.join(FONT_DIR, "*.ttf"))
    try:
        import ctypes
        if sys.platform.startswith("win"):
            for path in paths:
                ctypes.windll.gdi32.AddFontResourceExW(path, 0x10, 0)   # FR_PRIVATE
        elif sys.platform == "darwin":
            # CoreText: register each file for this process only.
            ct = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreText.framework/CoreText")
            cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
            cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
            ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
            ct.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            for path in paths:
                raw = path.encode("utf-8")
                url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)
                if url:
                    ct.CTFontManagerRegisterFontsForURL(url, 1, None)        # kCTFontManagerScopeProcess
                    cf.CFRelease(url)
    except Exception:
        pass  # the app falls back to standard fonts


class Fonts:
    """Font tuples, falling back to standard fonts if the bundled ones are missing."""
    def __init__(self, root):
        fams = set(tkfont.families(root))

        def pick(*names):
            return next((n for n in names if n in fams), names[-1])
        self.head_family = pick("Caprasimo", "Georgia")
        self.body_family = pick("Figtree", "Segoe UI", "Helvetica Neue", "Helvetica")
        # Windows lists "Figtree SemiBold" as its own family; macOS only lists "Figtree",
        # so there the semibold voice is Figtree in bold.
        self.semi_family = pick("Figtree SemiBold", "Segoe UI Semibold", self.body_family)
        self.semi_weight = "normal" if self.semi_family in ("Figtree SemiBold", "Segoe UI Semibold") else "bold"

    def head(self, size):
        return ctk.CTkFont(family=self.head_family, size=size)

    def body(self, size):
        return ctk.CTkFont(family=self.body_family, size=size)

    def semi(self, size):
        return ctk.CTkFont(family=self.semi_family, size=size, weight=self.semi_weight)


def badge_image(n, px=52):
    """Cream circle with a terracotta number, for the Stitch button."""
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((0, 0, px - 1, px - 1), fill=C["bg"])
    try:
        f = ImageFont.truetype(os.path.join(FONT_DIR, "Caprasimo-Regular.ttf"), int(px * 0.5))
    except OSError:
        f = ImageFont.load_default()
    d.text((px / 2, px / 2 + 1), str(n), fill=C["accent"], font=f, anchor="mm")
    return ctk.CTkImage(light_image=im, size=(px // 2, px // 2))


# --------------------------------------------------------------------------- #
#  App
# --------------------------------------------------------------------------- #
class StitcherApp(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=C["bg"])
        self.title(APP_TITLE)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1600, sw - 40)}x{min(940, sh - 90)}+20+20")   # fits a laptop screen
        self.minsize(1100, 680)
        self.F = Fonts(self)

        self.settings = core.load_settings()
        self.samples = {}           # name -> core.Sample
        self.selected = {}          # name -> bool (stitch this one?)
        self.status = {}            # name -> (text, tag) while/after stitching
        self.thumbs = {}            # path -> PIL thumbnail (None if unreadable)
        self.thumb_errors = {}      # path -> why it couldn't be opened
        self.loading = set()        # paths being decoded in the background
        self.done_cache = {}        # (path, mtime, size) -> result file is complete
        self.quick_cache = {}       # settings key -> {"image", "rows", "cols"} or error text
        self.quick_pending = set()
        self.before_images, self.after_images = [], []   # keep PhotoImages alive
        self.logs = []
        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None
        self.zoom = 1.0
        self.pan = [0.0, 0.0]       # shared by both previews (pixels)
        self._drag = None
        self._redraw_job = None
        self.list_expanded = True

        s = self.settings
        self.folder = tk.StringVar(value=s.get("folder", ""))
        self.engine = tk.StringVar(value=s.get("engine", "imagej"))
        self.fiji = tk.StringVar(value=s.get("fiji", "") or core.find_fiji())
        self.overlap = tk.IntVar(value=int(s.get("overlap", 20)))
        self.refine = tk.BooleanVar(value=s.get("refine", True))
        self.skip_done = tk.BooleanVar(value=s.get("skip_done", True))
        self.after_mode = tk.StringVar(value="Quick")
        self.cur_rows = tk.IntVar(value=1)
        self.cur_cols = tk.IntVar(value=2)
        self.cur_order = tk.StringVar(value=core.DEFAULT_ORDER)

        self._build()
        self._on_engine()
        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if sys.platform.startswith("win"):
            self.after(0, lambda: self.state("zoomed"))
        if self.folder.get() and os.path.isdir(self.folder.get()):
            self.after(300, self.scan)
        else:
            self.after(300, self._draw_previews)

    # ================================================================== layout
    def _badge(self, parent, n, size=26):
        return ctk.CTkLabel(parent, text=str(n), width=size, height=size, corner_radius=size // 2,
                            fg_color=C["accent"], text_color=C["bg"], font=self.F.head(13))

    def _link(self, parent, text, command, size=13):
        return ctk.CTkButton(parent, text=text, command=command, width=10, height=30,
                             fg_color="transparent", hover_color=C["hover_tint"],
                             text_color=C["accent_deep"], font=self.F.semi(size), corner_radius=15)

    def _pill_button(self, parent, text, command, height=40, size=14, **kw):
        opts = dict(fg_color="transparent", hover_color=C["hover_tint"], border_width=1,
                    border_color=C["line"], text_color=C["text"], corner_radius=height // 2)
        opts.update(kw)
        return ctk.CTkButton(parent, text=text, command=command, height=height,
                             font=self.F.head(size), **opts)

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        body = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=28, pady=20)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self._build_left(body)
        self._build_previews(body)
        self._build_runbar()
        self._style_tree()

    def _build_header(self):
        bar = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0, height=78)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1, minsize=500)
        ctk.CTkLabel(bar, text=APP_TITLE, font=self.F.head(22), text_color=C["text"]).grid(
            row=0, column=0, padx=(28, 24), pady=18)
        pill = ctk.CTkFrame(bar, fg_color=C["bg"], corner_radius=21, height=42)
        pill.grid(row=0, column=1, sticky="ew")
        pill.grid_columnconfigure(1, weight=1)
        self._badge(pill, 1, 30).grid(row=0, column=0, padx=(6, 8), pady=6)
        ctk.CTkEntry(pill, textvariable=self.folder, border_width=0, fg_color=C["bg"],
                     text_color=C["soft"], font=self.F.body(14), height=32,
                     placeholder_text="Choose the folder with your tiles").grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(pill, text="Browse…", command=self.browse, height=32, width=96, corner_radius=16,
                      fg_color=C["text"], hover_color=C["soft"], text_color=C["bg"],
                      font=self.F.head(13)).grid(row=0, column=2, padx=6)
        self._link(bar, "Rescan", self.scan, 14).grid(row=0, column=2, padx=(14, 0))
        bar.grid_columnconfigure(3, weight=1)
        self._link(bar, "Help", self.show_help, 14).grid(row=0, column=4, padx=(0, 22))

    def _build_left(self, body):
        self.left = ctk.CTkFrame(body, fg_color=C["bg"], corner_radius=0, width=540)
        self.left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        self.left.grid_propagate(False)
        self.left.grid_rowconfigure(0, weight=1)
        self.left.grid_columnconfigure(0, weight=1)

        # -- Samples card
        card = ctk.CTkFrame(self.left, fg_color=C["surface"], corner_radius=28)
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=(16, 12), pady=(14, 6))
        head.grid_columnconfigure(3, weight=1)
        self._badge(head, 2).grid(row=0, column=0)
        self.samples_title = ctk.CTkLabel(head, text="Samples", font=self.F.semi(15), text_color=C["text"])
        self.samples_title.grid(row=0, column=1, padx=(10, 8))
        self.count_label = ctk.CTkLabel(head, text="", font=self.F.body(13), text_color=C["muted"])
        self.count_label.grid(row=0, column=2)
        self.all_none = ctk.CTkFrame(head, fg_color="transparent")
        self.all_none.grid(row=0, column=4)
        self._link(self.all_none, "All", lambda: self._set_all(True)).pack(side="left")
        self._link(self.all_none, "None", lambda: self._set_all(False)).pack(side="left")
        self.chevron = ctk.CTkButton(head, text="‹", width=32, height=32, corner_radius=16,
                                     fg_color=C["bg"], hover_color=C["hover_tint"],
                                     text_color=C["text"], font=self.F.semi(16),
                                     command=self.toggle_list)
        self.chevron.grid(row=0, column=5, padx=(6, 0))

        holder = tk.Frame(card, bg=C["surface"], highlightthickness=0)
        holder.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 14))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        cols = ("use", "sample", "tiles", "grid", "order", "status")
        self.tree = ttk.Treeview(holder, columns=cols, show="headings", selectmode="extended",
                                 style="WW.Treeview")
        for col, text, w, anchor, stretch in (
                ("use", "", 34, "center", False), ("sample", "SAMPLE", 120, "w", True),
                ("tiles", "TILES", 50, "center", False), ("grid", "R×C", 72, "center", False),
                ("order", "ORDER", 120, "w", False), ("status", "STATUS", 130, "w", False)):
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=w, minwidth=w if not stretch else 60, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(holder, command=self.tree.yview, fg_color=C["surface"],
                              button_color=C["line"], button_hover_color=C["muted"])
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        for tag, colour in (("ready", C["sage_ink"]), ("done", C["muted"]), ("redo", C["accent_deep"]),
                            ("bad", C["error"]), ("run", C["accent"]), ("off", C["well_faint"])):
            self.tree.tag_configure(tag, foreground=colour)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.tree.bind("<Button-1>", self._on_tree_click)

        # -- Options card
        self.options = ctk.CTkFrame(self.left, fg_color=C["surface"], corner_radius=28)
        self.options.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self.options.grid_columnconfigure(1, weight=1)
        oh = ctk.CTkFrame(self.options, fg_color="transparent")
        oh.grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))
        self._badge(oh, 3).pack(side="left")
        ctk.CTkLabel(oh, text="Options", font=self.F.semi(15), text_color=C["text"]).pack(side="left", padx=10)

        def label(row, text):
            ctk.CTkLabel(self.options, text=text, font=self.F.body(13), text_color=C["muted"]).grid(
                row=row, column=0, sticky="w", padx=(20, 10), pady=5)

        label(1, "This sample")
        this = ctk.CTkFrame(self.options, fg_color="transparent")
        this.grid(row=1, column=1, sticky="w", padx=(0, 18))
        self._stepper(this, self.cur_rows, "rows").pack(side="left")
        self._stepper(this, self.cur_cols, "cols").pack(side="left", padx=8)
        self.order_menu = ctk.CTkOptionMenu(
            this, values=list(core.ORDERS), variable=self.cur_order, width=160, height=30,
            corner_radius=15, fg_color=C["bg"], button_color=C["bg"], button_hover_color=C["hover_tint"],
            text_color=C["text"], font=self.F.body(13), dropdown_font=self.F.body(13),
            dropdown_fg_color=C["bg"], dropdown_hover_color=C["accent_tint"], dropdown_text_color=C["text"],
            command=lambda v: self._set_field("order", v))
        self.order_menu.pack(side="left")

        label(2, "Engine")
        eng = ctk.CTkFrame(self.options, fg_color="transparent")
        eng.grid(row=2, column=1, sticky="w")
        for text, value in (("ImageJ (Fiji)", "imagej"), ("Python", "native")):
            ctk.CTkRadioButton(eng, text=text, value=value, variable=self.engine, command=self._on_engine,
                               fg_color=C["accent"], hover_color=C["accent_hover"],
                               border_color=C["muted"], text_color=C["text"], font=self.F.body(13),
                               radiobutton_width=18, radiobutton_height=18).pack(side="left", padx=(0, 18))

        self.fiji_label = ctk.CTkLabel(self.options, text="Fiji", font=self.F.body(13), text_color=C["muted"])
        self.fiji_label.grid(row=3, column=0, sticky="w", padx=(20, 10), pady=5)
        self.fiji_row = ctk.CTkFrame(self.options, fg_color="transparent")
        self.fiji_row.grid(row=3, column=1, sticky="ew", padx=(0, 18))
        self.fiji_row.grid_columnconfigure(0, weight=1)
        self.fiji_path_label = ctk.CTkLabel(self.fiji_row, text="", font=self.F.body(12), anchor="w",
                                            text_color=C["soft"])
        self.fiji_path_label.grid(row=0, column=0, sticky="ew")
        self._link(self.fiji_row, "Change", self.browse_fiji).grid(row=0, column=1)

        label(4, "Overlap")
        ov = ctk.CTkFrame(self.options, fg_color="transparent")
        ov.grid(row=4, column=1, sticky="ew", padx=(0, 18))
        ov.grid_columnconfigure(0, weight=1)
        self.overlap_slider = ctk.CTkSlider(
            ov, from_=1, to=60, number_of_steps=59, command=self._on_overlap, height=18,
            fg_color=C["track"], progress_color=C["accent"], button_color=C["accent"],
            button_hover_color=C["accent_hover"])
        self.overlap_slider.set(self.overlap.get())
        self.overlap_slider.grid(row=0, column=0, sticky="ew")
        self.overlap_label = ctk.CTkLabel(ov, text=f"{self.overlap.get()}%", width=44,
                                          font=self.F.semi(13), text_color=C["text"])
        self.overlap_label.grid(row=0, column=1, padx=(10, 0))

        checks = ctk.CTkFrame(self.options, fg_color="transparent")
        checks.grid(row=5, column=1, sticky="w", pady=(4, 18))
        for text, var, cmd in (("Fine-tune positions", self.refine, self._schedule_redraw),
                               ("Skip already stitched", self.skip_done, self._refresh_table)):
            ctk.CTkCheckBox(checks, text=text, variable=var, command=cmd, onvalue=True, offvalue=False,
                            fg_color=C["sage"], hover_color=C["sage_ink"], border_color=C["muted"],
                            checkmark_color=C["bg"], text_color=C["text"], font=self.F.body(13),
                            checkbox_width=18, checkbox_height=18, corner_radius=5).pack(side="left", padx=(0, 18))

    def _stepper(self, parent, var, field):
        """Pill with − value + for rows/columns."""
        box = ctk.CTkFrame(parent, fg_color=C["bg"], corner_radius=15, height=30)
        name = "rows" if field == "rows" else "cols"

        def step(d):
            try:
                v = int(var.get()) + d
            except (tk.TclError, ValueError):
                v = 1
            if 1 <= v <= 50:
                var.set(v)
                self._set_field(field, v)
        for text, d in (("−", -1), (None, 0), ("+", 1)):
            if text is None:
                ctk.CTkLabel(box, textvariable=var, width=22, font=self.F.semi(13),
                             text_color=C["text"]).pack(side="left")
                ctk.CTkLabel(box, text=name, font=self.F.body(12), text_color=C["muted"]).pack(side="left", padx=(2, 2))
            else:
                ctk.CTkButton(box, text=text, width=26, height=26, corner_radius=13, fg_color=C["bg"],
                              hover_color=C["hover_tint"], text_color=C["text"], font=self.F.semi(15),
                              command=lambda d=d: step(d)).pack(side="left", padx=2, pady=2)
        return box

    def _build_previews(self, body):
        right = ctk.CTkFrame(body, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure((0, 1), weight=1, uniform="wells")
        right.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(right, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        head.grid_columnconfigure(2, weight=1)
        self.cur_title = ctk.CTkLabel(head, text="", font=self.F.head(30), text_color=C["accent"])
        self.cur_title.grid(row=0, column=0, padx=(4, 14))
        self.cur_summary = ctk.CTkLabel(head, text="", font=self.F.body(15), text_color=C["soft"])
        self.cur_summary.grid(row=0, column=1)
        ctk.CTkLabel(head, text="Zoom is linked", font=self.F.body(13), text_color=C["muted"]).grid(
            row=0, column=3, padx=12)
        zoom = ctk.CTkFrame(head, fg_color=C["surface"], corner_radius=21)
        zoom.grid(row=0, column=4)
        for text, cmd in (("−", lambda: self.zoom_step(-1)), (None, None), ("+", lambda: self.zoom_step(1))):
            if text is None:
                self.zoom_label = ctk.CTkLabel(zoom, text="100%", width=56, font=self.F.semi(14),
                                               text_color=C["text"])
                self.zoom_label.pack(side="left")
            else:
                ctk.CTkButton(zoom, text=text, width=34, height=34, corner_radius=17, fg_color=C["surface"],
                              hover_color=C["hover_tint"], text_color=C["text"], font=self.F.semi(18),
                              command=cmd).pack(side="left", padx=2, pady=4)
        ctk.CTkButton(zoom, text="Fit", width=50, height=34, corner_radius=17, fg_color=C["bg"],
                      hover_color=C["hover_tint"], text_color=C["text"], font=self.F.semi(13),
                      command=self.zoom_fit).pack(side="left", padx=(2, 4), pady=4)

        # Before well
        bw = ctk.CTkFrame(right, fg_color=C["well"], corner_radius=28)
        bw.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        bw.grid_rowconfigure(1, weight=1)
        bw.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(bw, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 0))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Before", font=self.F.head(17), text_color=C["bg"]).grid(row=0, column=0)
        self.before_info = ctk.CTkLabel(top, text="", font=self.F.body(12), text_color=C["well_text"])
        self.before_info.grid(row=0, column=2)
        self.before = tk.Canvas(bw, background=C["well"], highlightthickness=0, cursor="fleur")
        self.before.grid(row=1, column=0, sticky="nsew", padx=18, pady=(8, 8))
        legend = ctk.CTkFrame(bw, fg_color="transparent")
        legend.grid(row=2, column=0, sticky="w", padx=24, pady=(0, 16))
        sw = tk.Canvas(legend, width=18, height=12, bg=C["well"], highlightthickness=0)
        sw.create_rectangle(1, 1, 17, 11, outline=C["band"], dash=(2, 2), fill="#5d6252")
        sw.pack(side="left")
        self.legend_overlap = ctk.CTkLabel(legend, text="Overlap", font=self.F.body(12), text_color=C["well_text"])
        self.legend_overlap.pack(side="left", padx=(6, 16))
        dot = tk.Canvas(legend, width=12, height=12, bg=C["well"], highlightthickness=0)
        dot.create_oval(1, 1, 11, 11, fill=C["order"], outline="")
        dot.pack(side="left")
        ctk.CTkLabel(legend, text="Stitch order", font=self.F.body(12), text_color=C["well_text"]).pack(
            side="left", padx=6)

        # After well
        aw = ctk.CTkFrame(right, fg_color=C["well"], corner_radius=28)
        aw.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        aw.grid_rowconfigure(1, weight=1)
        aw.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(aw, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=(24, 16), pady=(14, 0))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="After", font=self.F.head(17), text_color=C["bg"]).grid(row=0, column=0)
        ctk.CTkSegmentedButton(top, values=["Quick", "Saved"], variable=self.after_mode,
                               command=lambda v: self._draw_after(), font=self.F.body(13), height=30,
                               corner_radius=15, fg_color=C["seg"], selected_color=C["accent"],
                               selected_hover_color=C["accent_hover"], unselected_color=C["seg"],
                               unselected_hover_color=C["soft"], text_color=C["bg"],
                               text_color_disabled=C["well_faint"]).grid(row=0, column=2)
        self.after_canvas = tk.Canvas(aw, background=C["well"], highlightthickness=0, cursor="fleur")
        self.after_canvas.grid(row=1, column=0, sticky="nsew", padx=18, pady=(8, 8))
        self.after_note = ctk.CTkLabel(aw, text="", font=self.F.body(12), text_color=C["well_faint"])
        self.after_note.grid(row=2, column=0, sticky="w", padx=24, pady=(0, 16))

        for cv in (self.before, self.after_canvas):
            cv.bind("<Configure>", lambda e: self._schedule_redraw(delay=60))
            cv.bind("<ButtonPress-1>", self._drag_start)
            cv.bind("<B1-Motion>", self._drag_move)
            cv.bind("<MouseWheel>", lambda e: self.zoom_step(1 if e.delta > 0 else -1))
            cv.bind("<Double-Button-1>", lambda e: self.zoom_fit())

    def _build_runbar(self):
        bar = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_columnconfigure(2, weight=1)
        self.run_btn = ctk.CTkButton(bar, text="Stitch", image=badge_image(4), compound="left",
                                     command=self.start, height=56, corner_radius=28,
                                     fg_color=C["accent"], hover_color=C["accent_hover"],
                                     text_color=C["bg"], text_color_disabled=C["bg"],
                                     font=self.F.head(19), width=300)
        self.run_btn.grid(row=0, column=0, padx=(28, 12), pady=16)
        self.stop_btn = self._pill_button(bar, "Stop", self.stop, height=56, size=15, width=96,
                                          text_color_disabled=C["line"])
        self.stop_btn.grid(row=0, column=1)
        self.stop_btn.configure(state="disabled")
        mid = ctk.CTkFrame(bar, fg_color="transparent")
        mid.grid(row=0, column=2, sticky="ew", padx=18)
        mid.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(mid, height=8, corner_radius=4, fg_color=C["track"],
                                           progress_color=C["track"])   # accent once a run starts
        self.progress.set(0)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.status_label = ctk.CTkLabel(mid, text="Choose a folder to begin", anchor="w",
                                         font=self.F.body(13), text_color=C["soft"])
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._link(mid, "Show log", self.show_log, 12).grid(row=1, column=1, pady=(4, 0))
        self._pill_button(bar, "Open results folder", self.open_results, height=44, size=14).grid(
            row=0, column=3, padx=(0, 28))

    def _style_tree(self):
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        st = ttk.Style(self)
        st.theme_use("clam")
        st.layout("WW.Treeview", [("WW.Treeview.treearea", {"sticky": "nswe"})])
        st.configure("WW.Treeview", background=C["surface"], fieldbackground=C["surface"],
                     foreground=C["text"], borderwidth=0, rowheight=int(30 * scale),
                     font=(self.F.body_family, 11))
        st.map("WW.Treeview", background=[("selected", C["accent_tint"])],
               foreground=[("selected", C["accent_ink"])])
        st.configure("WW.Treeview.Heading", background=C["surface"], foreground=C["muted"],
                     borderwidth=0, relief="flat", font=(self.F.semi_family, 8), padding=(4, 6))
        st.map("WW.Treeview.Heading", background=[("active", C["surface"])])

    # ================================================================= actions
    def browse(self):
        d = filedialog.askdirectory(title="Choose the folder with your tiles",
                                    initialdir=self.folder.get() or os.path.expanduser("~"))
        if d:
            self.folder.set(os.path.normpath(d))
            self.scan()

    def browse_fiji(self):
        if sys.platform.startswith("win"):
            f = filedialog.askopenfilename(
                title="Choose the Fiji program (fiji-windows-x64.exe or ImageJ-win64.exe in the Fiji folder)",
                filetypes=[("Programs", "*.exe"), ("All files", "*.*")])
        else:  # macOS / Linux: pick Fiji.app or the Fiji folder
            f = filedialog.askdirectory(title="Choose Fiji.app (or the Fiji folder)")
        if not f:
            return
        exe = core.resolve_fiji(os.path.normpath(f))
        if not exe:
            messagebox.showwarning(APP_TITLE, "That doesn't look like Fiji. Choose the Fiji program, "
                                              "Fiji.app, or the folder Fiji was unzipped into.")
            return
        self.fiji.set(exe)
        self._on_engine()

    def _text_window(self, title, text, width=720, height=760):
        win = ctk.CTkToplevel(self, fg_color=C["bg"])
        win.title(title)
        win.geometry(f"{width}x{height}")
        box = ctk.CTkTextbox(win, fg_color=C["surface"], text_color=C["text"], corner_radius=20,
                             font=ctk.CTkFont(family="Consolas", size=13), wrap="word")
        box.pack(fill="both", expand=True, padx=16, pady=16)
        box.insert("1.0", text)
        box.configure(state="disabled")
        win.after(50, win.lift)
        return box

    def show_help(self):
        self._text_window("How to use", HELP_TEXT)

    def show_log(self):
        box = self._text_window("Log", "\n".join(self.logs) or "Nothing yet.", 820, 520)
        box.see("end")

    def scan(self):
        folder = self.folder.get().strip()
        if not os.path.isdir(folder):
            messagebox.showwarning(APP_TITLE, "Please choose a folder that exists.")
            return
        try:
            samples, ignored = core.scan_folder(folder)
        except OSError as e:
            messagebox.showerror(APP_TITLE, f"Couldn't read the folder:\n{e}")
            return
        self.samples = {s.name: s for s in samples}
        self.selected = {s.name: True for s in samples}
        self.status = {}
        self.thumbs.clear()
        self.thumb_errors.clear()
        self.quick_cache.clear()
        self._refresh_table()
        self.log(f"Found {len(samples)} sample(s) in {folder}")
        if ignored:
            self.log(f"Ignored {len(ignored)} image(s) without a tile number, e.g. {ignored[0]}")
        if not samples:
            messagebox.showinfo(APP_TITLE, "No tiles found in this folder.\n\nTile files must end in "
                                           "\"_<number>\", e.g. Sample_01.tiff (see Help).")
            self._draw_previews()
        else:
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)

    def _set_field(self, field, value):
        """Rows / Columns / Order changed: apply to every selected sample."""
        names = self.tree.selection()
        for name in names:
            setattr(self.samples[name], field, value)
            self.status.pop(name, None)
        if names:
            self._schedule_redraw(table=True)

    def _set_all(self, value):
        for name in self.selected:
            self.selected[name] = value
        self._refresh_table()

    def _on_tree_click(self, event):
        if self.tree.identify_region(event.x, event.y) == "cell" and \
                self.tree.identify_column(event.x) == "#1":
            name = self.tree.identify_row(event.y)
            if name:
                self.selected[name] = not self.selected[name]
                self._refresh_table()

    def _on_engine(self):
        imagej = self.engine.get() == "imagej"
        if imagej:
            path = self.fiji.get() or "not found – click Change"
            if len(path) > 52:
                path = "…" + path[-51:]
            self.fiji_path_label.configure(text=path)
            self.fiji_label.grid()
            self.fiji_row.grid()
        else:
            self.fiji_label.grid_remove()
            self.fiji_row.grid_remove()
        self._update_after_note()

    def _on_overlap(self, value):
        v = int(round(value))
        if v != self.overlap.get():
            self.overlap.set(v)
            self.overlap_label.configure(text=f"{v}%")
            self._schedule_redraw()

    def toggle_list(self):
        self.list_expanded = not self.list_expanded
        if self.list_expanded:
            self.left.configure(width=540)
            self.options.grid()
            self.tree.configure(displaycolumns=("use", "sample", "tiles", "grid", "order", "status"))
            self.count_label.grid()
            self.all_none.grid()
            self.samples_title.grid()
            self.chevron.configure(text="‹")
        else:
            self.left.configure(width=170)
            self.options.grid_remove()
            self.tree.configure(displaycolumns=("use", "sample"))
            self.count_label.grid_remove()
            self.all_none.grid_remove()
            self.samples_title.grid_remove()
            self.chevron.configure(text="›")
        self._schedule_redraw(delay=80)

    def open_results(self):
        out = os.path.join(self.folder.get(), core.OUT_DIR_NAME)
        if not os.path.isdir(out):
            messagebox.showinfo(APP_TITLE, "Nothing has been stitched in this folder yet.")
            return
        if sys.platform.startswith("win"):
            os.startfile(out)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", out])

    # =================================================================== table
    def _done_cached(self, path):
        """core.output_ok, cached per file version (the table refreshes often)."""
        try:
            st = os.stat(path)
        except OSError:
            return False
        key = (path, st.st_mtime, st.st_size)
        if key not in self.done_cache:
            self.done_cache[key] = core.output_ok(path)
        return self.done_cache[key]

    def _row_status(self, s):
        """(text, tag) for the Status column."""
        if s.name in self.status:
            return self.status[s.name]
        if not self.selected.get(s.name):
            return "Skipped", "off"
        p = s.problem()
        if p:
            return "Fix layout", "bad"
        out = s.output_path(self.folder.get())
        if self.skip_done.get() and self._done_cached(out):
            return "Stitched", "done"
        if os.path.exists(out) and not self._done_cached(out):
            return "Redo", "redo"
        if s.warnings:
            return "Check tiles", "redo"
        return "Ready", "ready"

    def _todo(self):
        folder = self.folder.get()
        return [s for name, s in self.samples.items()
                if self.selected[name] and not s.problem()
                and not (self.skip_done.get() and self._done_cached(s.output_path(folder)))]

    def _refresh_table(self):
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for name, s in self.samples.items():
            text, tag = self._row_status(s)
            grid = "auto" if s.n == 2 else f"{s.rows} × {s.cols}"
            order = s.order if s.n > 2 else ("auto (pair)" if s.n == 2 else "—")
            self.tree.insert("", "end", iid=name, tags=(tag,), values=(
                "☑" if self.selected[name] else "☐", name, s.n, grid, order, "●  " + text))
        keep = [n for n in sel if self.tree.exists(n)]
        if keep:
            self.tree.selection_set(keep)
        n_sel = sum(self.selected.values())
        self.count_label.configure(text=f"{n_sel} / {len(self.samples)}" if self.samples else "")
        if not (self.worker and self.worker.is_alive()):
            n = len(self._todo())
            self.run_btn.configure(text=f"Stitch {n} sample{'s' if n != 1 else ''}" if self.samples else "Stitch")

    def _on_select(self):
        s = self._current_sample()
        if s is not None:
            self.cur_rows.set(s.rows)
            self.cur_cols.set(s.cols)
            self.cur_order.set(s.order)
            self.pan = [0.0, 0.0]
        self._draw_previews()

    def _current_sample(self):
        names = self.tree.selection()
        return self.samples.get(names[0]) if names else None

    # ======================================================== zoom, pan, redraw
    def zoom_step(self, d):
        i = min(range(len(ZOOMS)), key=lambda k: abs(ZOOMS[k] - self.zoom))
        i = max(0, min(len(ZOOMS) - 1, i + d))
        if ZOOMS[i] != self.zoom:
            self.pan = [p * ZOOMS[i] / self.zoom for p in self.pan]
            self.zoom = ZOOMS[i]
            self._draw_previews()

    def zoom_fit(self):
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self._draw_previews()

    def _drag_start(self, e):
        self._drag = (e.x, e.y)

    def _drag_move(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.pan[0] += dx
        self.pan[1] += dy
        for cv in (self.before, self.after_canvas):   # linked: move both
            cv.move("content", dx, dy)

    def _schedule_redraw(self, table=False, delay=250):
        """Redraw shortly after the last change so sliders and steppers stay smooth."""
        if self._redraw_job:
            self.after_cancel(self._redraw_job)

        def go():
            self._redraw_job = None
            if table:
                self._refresh_table()
            self._draw_previews()
        self._redraw_job = self.after(delay, go)

    def _draw_previews(self):
        s = self._current_sample()
        self.zoom_label.configure(text=f"{int(self.zoom * 100)}%")
        self.legend_overlap.configure(text=f"Overlap {self.overlap.get()}%")
        if s is None:
            self.cur_title.configure(text="")
            self.cur_summary.configure(text="")
            self.before_info.configure(text="")
        else:
            self.cur_title.configure(text=s.name)
            if s.n == 2:
                desc = "2 tiles · pair direction detected automatically"
            elif s.n == 1:
                desc = "1 tile · copied as it is"
            else:
                desc = f"{s.n} tiles · {s.rows}×{s.cols} · {s.order.lower()}"
            self.cur_summary.configure(text=desc)
            self.before_info.configure(text=f"{s.n} tile{'s' if s.n != 1 else ''} · overlap {self.overlap.get()}%")
        self._draw_before()
        self._draw_after()

    # --------------------------------------------------------------- images
    def _thumb(self, path, edge):
        if path not in self.thumbs:
            try:
                im = Image.fromarray(core.read_rgb(path))
                im.thumbnail((edge, edge))
                self.thumbs[path] = im
            except Exception as e:
                self.thumbs[path] = None
                self.thumb_errors[path] = str(e)
        return self.thumbs[path]

    def _load_then_draw(self, paths, edge=TILE_THUMB):
        """Decode thumbnails off the UI thread, then redraw."""
        paths = [p for p in paths if p not in self.loading]
        if not paths:
            return
        self.loading.update(paths)

        def work():
            for p in paths:
                self._thumb(p, edge)
                self.loading.discard(p)
            self.events.put(("redraw", None))
        threading.Thread(target=work, daemon=True).start()

    def _tiles_ready(self, s):
        missing = [p for p in s.paths if p not in self.thumbs]
        if missing:
            self._load_then_draw(missing)
        return not missing

    @staticmethod
    def _message(cv, text, colour=None):
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        cv.create_text(W / 2, H / 2, text=text, fill=colour or C["well_text"], width=W - 40,
                       font=("Figtree", 12), justify="center")

    def _place_image(self, cv, im, keep):
        """Draw `im` fitted to the canvas × zoom, centred + panned."""
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        fit = min((W - 24) / im.width, (H - 24) / im.height)
        scale = fit * self.zoom
        w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        ph = ImageTk.PhotoImage(im.resize((w, h), Image.LANCZOS if scale < 1 else Image.BILINEAR))
        keep.append(ph)
        cv.create_image(W / 2 + self.pan[0], H / 2 + self.pan[1], image=ph, tags="content")

    # ------------------------------------------------------------- before
    def _layout_for(self, s):
        """(rows, cols) to draw: pairs use the direction found by the quick preview."""
        if s.n == 2:
            q = self.quick_cache.get(self._quick_key(s))
            if isinstance(q, dict):
                return q["rows"], q["cols"]
            return 1, 2
        return s.rows, s.cols

    def _draw_before(self):
        cv = self.before
        cv.delete("all")
        self.before_images.clear()
        s = self._current_sample()
        if s is None:
            self._message(cv, "Choose a folder, then click a sample")
            return
        if not self._tiles_ready(s):
            self._message(cv, "Loading tiles…")
            return
        rows, cols = self._layout_for(s)
        if rows * cols < s.n:
            self._message(cv, (s.problem() or "") + "\n\nChange Rows / Columns under Options.", C["order"])
            return
        first = self.thumbs[s.paths[0]]
        A = (first.height / first.width) if first else 1.0       # tile height / width
        o = self.overlap.get() / 100.0
        span_w = cols - (cols - 1) * o                          # layout size in tile widths
        span_h = (rows - (rows - 1) * o) * A
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        unit = min((W - 40) / span_w, (H - 40) / span_h) * self.zoom   # px per tile width
        x0 = W / 2 + self.pan[0] - span_w * unit / 2
        y0 = H / 2 + self.pan[1] - span_h * unit / 2
        tw, th = unit, unit * A
        pos = core.tile_positions(s.n, rows, cols, s.order)

        # Compose tiles + translucent overlap strips into one picture (Tk can't do alpha).
        full_w, full_h = span_w * unit, span_h * unit
        iw, ih = max(1, int(round(full_w))), max(1, int(round(full_h)))
        tw_i, th_i = max(1, int(round(tw))), max(1, int(round(th)))
        layer = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
        centres = []
        for (r, c), path in zip(pos, s.paths):
            x, y = c * (1 - o) * tw, r * (1 - o) * th
            im = self.thumbs[path]
            if im is not None:
                layer.paste(im.convert("RGB").resize((tw_i, th_i),
                            Image.LANCZOS if tw_i < im.width else Image.BILINEAR), (int(x), int(y)))
            centres.append((x0 + x + tw / 2, y0 + y + th / 2))
        bands = []
        if o > 0:
            bands += [(j * (1 - o) * tw, 0, o * tw, full_h) for j in range(1, cols)]
            bands += [(0, i * (1 - o) * th, full_w, o * th) for i in range(1, rows)]
        if bands:
            tint = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(tint)
            for bx, by, bw, bh in bands:
                d.rectangle((bx, by, bx + bw, by + bh), fill=(174, 191, 146, 80))
            layer = Image.alpha_composite(layer, tint)
        ph = ImageTk.PhotoImage(layer)
        self.before_images.append(ph)
        cv.create_image(x0, y0, image=ph, anchor="nw", tags="content")
        for bx, by, bw, bh in bands:
            cv.create_rectangle(x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh, outline=C["band"],
                                dash=(4, 3), tags="content")
        if len(centres) > 1:
            cv.create_line(*[v for p in centres for v in p], fill=C["order"], width=2.5,
                           dash=(6, 5), arrow="last", arrowshape=(12, 14, 5), tags="content")
        rad = 15
        for (cx, cy), (index, _) in zip(centres, s.tiles):
            cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=C["order"], outline=C["well"],
                           width=3, tags="content")
            cv.create_text(cx, cy + 1, text=str(index), fill=C["well"],
                           font=(self.F.head_family, 11), tags="content")

    # -------------------------------------------------------------- after
    def _quick_key(self, s):
        return (s.name, tuple(s.paths), s.rows, s.cols, s.order, int(self.overlap.get()),
                bool(self.refine.get()))

    def _update_after_note(self):
        if self.after_mode.get() == "Saved":
            return
        note = " · ImageJ may differ slightly" if self.engine.get() == "imagej" else ""
        self.after_note.configure(text="Low-res preview" + note)

    def _draw_after(self):
        cv = self.after_canvas
        cv.delete("all")
        self.after_images.clear()
        s = self._current_sample()
        self._update_after_note()
        if s is None:
            return
        if self.after_mode.get() == "Saved":
            self._draw_saved(s)
            return
        if s.problem():
            self._message(cv, s.problem(), C["order"])
            return
        if not self._tiles_ready(s):
            self._message(cv, "Loading tiles…")
            return
        key = self._quick_key(s)
        result = self.quick_cache.get(key)
        if result is None:
            self._message(cv, "Working out the stitch…")
            self._start_quick(s, key)
            return
        if isinstance(result, str):
            self._message(cv, "Quick preview failed:\n" + result, C["order"])
            return
        self._place_image(cv, result["image"], self.after_images)

    def _start_quick(self, s, key):
        """Stitch shrunken copies of the tiles in the background (Python engine)."""
        if key in self.quick_pending:
            return
        self.quick_pending.add(key)
        thumbs = [self.thumbs[p] for p in s.paths]
        overlap, refine = key[5] / 100.0, key[6]
        rows, cols, order = s.rows, s.cols, s.order

        def work():
            import numpy as np
            import stitch_native
            try:
                if any(t is None for t in thumbs):
                    raise ValueError("a tile couldn't be opened")
                f = min(1.0, QUICK_EDGE / max(thumbs[0].size))
                size = (max(1, int(thumbs[0].width * f)), max(1, int(thumbs[0].height * f)))
                tiles = [np.asarray(t.convert("RGB").resize(size), dtype=np.float32) for t in thumbs]
                r, cl = rows, cols
                if len(tiles) == 1:
                    mosaic = tiles[0].astype(np.uint8)
                else:
                    if len(tiles) == 2:
                        r, cl = stitch_native.auto_orient_pair(tiles[0], tiles[1])
                    pos = core.tile_positions(len(tiles), r, cl, order)
                    mosaic = stitch_native.stitch_arrays(tiles, pos, overlap, refine=refine, verbose=False)
                result = {"image": Image.fromarray(mosaic), "rows": r, "cols": cl}
            except Exception as e:
                result = str(e)
            self.events.put(("quick", (key, result)))
        threading.Thread(target=work, daemon=True).start()

    def _draw_saved(self, s):
        cv = self.after_canvas
        out = s.output_path(self.folder.get())
        self.after_note.configure(text=os.path.basename(out))
        if not os.path.exists(out):
            self._message(cv, "Not stitched yet – press Stitch to make it")
            return
        if out not in self.thumbs:
            self._message(cv, "Loading…")
            self._load_then_draw([out], RESULT_THUMB)
            return
        im = self.thumbs[out]
        if im is None:
            self._message(cv, f"Couldn't open the result:\n{self.thumb_errors.get(out, '')}\n\n"
                              "Press Stitch to make it again.", C["order"])
            return
        self._place_image(cv, im, self.after_images)
        try:
            w, h = core.image_size(out)
            self.after_note.configure(text=f"{os.path.basename(out)} · {w} × {h} pixels")
        except Exception:
            pass

    # ================================================================ running
    def log(self, msg):
        self.logs.append(msg)
        if msg.strip():
            self.status_label.configure(text=msg.strip())

    def _save_settings(self):
        self.settings.update(folder=self.folder.get(), engine=self.engine.get(), fiji=self.fiji.get(),
                             overlap=int(self.overlap.get()), refine=bool(self.refine.get()),
                             skip_done=bool(self.skip_done.get()))
        core.save_settings(self.settings)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        folder = self.folder.get().strip()
        if not self.samples:
            messagebox.showinfo(APP_TITLE, "Choose a folder with tiles first.")
            return
        bad = [n for n, s in self.samples.items() if self.selected[n] and s.problem()]
        if bad:
            messagebox.showwarning(APP_TITLE, "Fix the layout of these samples first (marked \"Fix layout\"):\n\n"
                                   + "\n".join(bad[:15]))
            return
        todo = self._todo()
        if not todo:
            messagebox.showinfo(APP_TITLE, "Nothing to stitch: every selected sample is already done.\n"
                                           "Untick \"Skip already stitched\" to redo them.")
            return
        engine = self.engine.get()
        if engine == "imagej" and not core.resolve_fiji(self.fiji.get()):
            found = core.find_fiji()
            if found:
                self.fiji.set(found)
                self._on_engine()
            else:
                messagebox.showwarning(APP_TITLE, "Fiji wasn't found.\n\nClick \"Change\" next to Fiji "
                                       "under Options and choose it, or use the Python engine.")
                return
        self._save_settings()
        for s in todo:
            self.status[s.name] = ("Waiting", "off")
            out = s.output_path(folder)
            self.thumbs.pop(out, None)
            self.thumb_errors.pop(out, None)
        self.total, self.finished = len(todo), 0
        self._refresh_table()
        self.progress.set(0)
        self.progress.configure(progress_color=C["accent"])
        self.run_btn.configure(state="disabled", text=f"Stitching 0 / {self.total}")
        self.stop_btn.configure(state="normal")
        self.stop_flag.clear()
        self.log("")
        self.log(f"Stitching {len(todo)} sample(s) with {'ImageJ (Fiji)' if engine == 'imagej' else 'Python'}…")

        def on_start(name):
            self.events.put(("status", (name, ("Stitching…", "run"))))

        def on_done(name, ok):
            self.events.put(("done", (name, ok)))

        def work():
            try:
                kw = dict(overlap=self.overlap.get() / 100.0, refine=bool(self.refine.get()),
                          log=lambda m: self.events.put(("log", m)), on_start=on_start,
                          on_done=on_done, should_stop=self.stop_flag.is_set)
                if engine == "imagej":
                    core.run_imagej(todo, folder, self.fiji.get(), **kw)
                else:
                    core.run_native(todo, folder, **kw)
            except Exception as e:
                self.events.put(("log", f"Error: {e}"))
            self.events.put(("finished", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag.set()
        self.log("Stopping after the current step…")

    def _poll_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "log":
                    self.log(data)
                elif kind == "status":
                    name, st = data
                    if name in self.samples:
                        self.status[name] = st
                        self._refresh_table()
                elif kind == "done":
                    name, ok = data
                    self.status[name] = ("Done", "done") if ok else ("Failed", "bad")
                    self.finished += 1
                    self.progress.set(self.finished / max(1, self.total))
                    self.run_btn.configure(text=f"Stitching {self.finished} / {self.total}")
                    self._refresh_table()
                    s = self._current_sample()
                    if ok and s and s.name == name and self.after_mode.get() == "Saved":
                        self._draw_after()
                elif kind == "finished":
                    for name, (text, _) in list(self.status.items()):
                        if text in ("Waiting", "Stitching…"):
                            self.status[name] = ("Stopped", "off")
                    n_ok = sum(1 for t, _ in self.status.values() if t == "Done")
                    n_bad = sum(1 for t, _ in self.status.values() if t == "Failed")
                    self.stop_btn.configure(state="disabled")
                    self.run_btn.configure(state="normal")
                    self._refresh_table()
                    self.log(f"Finished: {n_ok} stitched, {n_bad} failed. Results are in the "
                             f"\"{core.OUT_DIR_NAME}\" folder.")
                elif kind == "redraw":
                    self._draw_previews()
                elif kind == "quick":
                    key, result = data
                    self.quick_pending.discard(key)
                    self.quick_cache[key] = result
                    s = self._current_sample()
                    if s is not None and self._quick_key(s) == key:
                        self._draw_previews()   # pairs: Before uses the detected direction
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(APP_TITLE, "Stitching is still running. Stop and quit?"):
                return
            self.stop_flag.set()
        self._save_settings()
        self.destroy()


def main():
    load_bundled_fonts()
    ctk.set_appearance_mode("light")
    Image.MAX_IMAGE_PIXELS = None
    StitcherApp().mainloop()


if __name__ == "__main__":
    main()
