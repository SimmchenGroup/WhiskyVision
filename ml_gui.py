"""
ml_gui.py — the "ML" tab of the Whisky Webs app (front end for ml_core.py).

Four pages, picked at the top of the tab:
  Explore          one droplet at a time: rim, ring map, normalisation steps, unwrap,
                   QC verdict, normalisation numbers and features; figures of several
  Sample average   the average droplet / unwrap / radial profiles of a whole whisky,
                   and profiles of several whiskies compared
  Train & evaluate load the dataset (same images as Identify/Gallery), QC table,
                   cross-validation, control (nuisance) run, feature-set comparison,
                   train & save a model
  Predict          load a model, add images, per-droplet results and a sample verdict

Everything heavier than drawing a view runs in one background thread (self._run);
results come back through the app's event queue as ("ml_…", data) events.
"""
import csv
import os
import tempfile
import threading
import tkinter as tk
import traceback
from collections import OrderedDict
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import identify_core as idc
import ml_core as mc

PAGES = ["Explore", "Sample average", "Train & evaluate", "Predict"]
VIEWS = ["Raw", "Rim", "Ring map", "Background", "Subtracted", "Normalised", "Unwrap", "Crop"]
FIG_ROWS = 6                 # per-droplet figures get one row per image
DROPLET_CACHE = 24           # analysed droplets kept in memory (Explore)
SETS = ["scalar", "radial", "scalar+radial"]
CNN_NOTE = ("EfficientNet-B0, used only to describe each droplet (not trained). "
            "First run re-reads every image, then it's cached.")
CNN_MISSING = "Needs PyTorch (about 1 GB, CPU version). Install it once with the link above."
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

HELP_TEXT = """\
ML: DROPLET FEATURES AND A TRAINED CLASSIFIER

Unlike Identify (which compares whole pictures), the ML tab finds the droplet
by its coffee ring, checks it (QC), unwraps it radially (so droplet size and
rotation stop mattering), measures 26 features and a radial profile, and
trains a classifier on them. It uses the same reference folder as Identify
and the Gallery.

EXPLORE
Add images (files, a folder, or the pictures ticked in the Gallery's Select
mode). Click one to analyse it (1–3 s) and switch between the views:
  Raw · Rim (green = passes QC, red = fails) · Ring map · Background
  (blue, beyond 1.25 R) · Subtracted · Normalised · Unwrap (angle left to
  right, centre at the bottom, rim at the dashed line) · Crop
"Normalised" next to the views switches Unwrap and Crop between raw and
normalised. Below: the QC verdict, the normalisation numbers and every
feature. Pipeline / Normalisation / Profiles figures use the selected images
(Ctrl/Shift-click for several, up to 6 rows).

SAMPLE AVERAGE
Pick a whisky and press Average: the average droplet (rotations are random,
so it shows the radial structure), the mean unwrap, mean ± SD profiles and
the droplets rejected by QC. Select several whiskies and press Compare to
overlay their profiles.

TRAIN & EVALUATE
"Load dataset" analyses every reference image. The FIRST time this takes a
long time (roughly 30–60 min for 2,400 images over the network); results are
saved on this computer, so later runs take seconds. Then:
  Run cross-validation   balanced accuracy, confusion matrix, per-whisky
                         recall, and accuracy when 1 / 3 / 5 droplets of a
                         sample are averaged
  Run control            the "nuisance" features (droplet size, ring score,
                         image size). If it scores close to the real
                         features, the model is reading imaging artefacts.
  Compare feature sets   all sets side by side
  Train & save model     trains on every good droplet and saves a .wwml.npz
  Training curves        inside one run: PCA, loss and accuracy per step
                         (one step = one pass over the data), and the
                         features the model leans on
Pretrained CNN sets (need PyTorch): a ready-made image network
(EfficientNet-B0, trained on ordinary photos) describes each droplet with
1,280 numbers; the same quick classifier is trained on those. The network
itself is never trained here. The first CNN run re-reads every image
(about as long as the first "Load dataset"), then it's cached. If PyTorch
isn't installed, use "Install PyTorch…" next to the CNN buttons: it installs
the CPU version (about 1 GB) into the Python this app runs on.
Scores are WITHIN-SLIP: all droplets of a whisky are on one cover slip, so
they mean "recognises another droplet from the same slip", not a guarantee
for a new slip.

PREDICT
Load a model (or use the one just trained), add images and press Predict.
Tick "These images are one sample" when they are droplets of the same
sample: they are judged together, which is far more reliable (and is the
only good check for "not like any trained whisky"). A model always names one
of the whiskies it was trained on; a close relative can be called as its
sibling. "Save results" writes a CSV."""


def fit_to(im, w, h):
    """`im` scaled to fit w × h (up or down)."""
    s = min(w / im.width, h / im.height)
    size = (max(1, int(im.width * s)), max(1, int(im.height * s)))
    return im.resize(size, Image.LANCZOS if s < 1 else Image.BICUBIC)


def images_in(folder):
    """Stitched images in `folder` (recursive), or every image if none is stitched."""
    found = idc.class_images(folder)
    stitched = [p for p in found if idc.is_stitched(p)]
    return stitched or found


def open_path(path):
    import subprocess
    import sys
    if sys.platform.startswith("win"):
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


# --------------------------------------------------------------------------- #
#  Small building blocks
# --------------------------------------------------------------------------- #
class FigureWindow(ctk.CTkToplevel):
    """A matplotlib figure (PIL image) fitted to a window, with Save / Open full size."""

    def __init__(self, app, C, title, image, default_name="figure.png"):
        super().__init__(app, fg_color=C["bg"])
        self.app, self.C, self.image, self.default_name = app, C, image, default_name
        self.title(title)
        w = min(1500, max(700, image.width // 2 + 60))
        h = min(950, max(420, int(image.height * (w - 60) / image.width) + 120))
        self.geometry(f"{w}x{h}")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        well = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=20)
        well.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))
        well.grid_rowconfigure(0, weight=1)
        well.grid_columnconfigure(0, weight=1)
        self.cv = tk.Canvas(well, background="#ffffff", highlightthickness=0)
        self.cv.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.cv.bind("<Configure>", lambda e: self._draw())
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
        ctk.CTkLabel(bar, text=f"{image.width} × {image.height} px", font=app.F.body(12),
                     text_color=C["muted"]).pack(side="left", padx=6)
        app._pill_button(bar, "Open full size", self.open_full, height=36, size=13).pack(side="right")
        app._pill_button(bar, "Save PNG…", self.save, height=36, size=13).pack(side="right", padx=8)
        self._ph = None
        self.after(50, self.lift)

    def _draw(self):
        W, H = max(self.cv.winfo_width(), 50), max(self.cv.winfo_height(), 50)
        self._ph = ImageTk.PhotoImage(fit_to(self.image, W, H))
        self.cv.delete("all")
        self.cv.create_image(W / 2, H / 2, image=self._ph)

    def save(self):
        f = filedialog.asksaveasfilename(parent=self, title="Save figure", defaultextension=".png",
                                         initialfile=self.default_name, filetypes=[("PNG", "*.png")])
        if f:
            try:
                self.image.save(f)
                self.app.logs.append(f"Saved figure to {f}")
            except OSError as e:
                messagebox.showerror(self.app.title(), f"Couldn't save:\n{e}", parent=self)

    def open_full(self):
        path = os.path.join(tempfile.gettempdir(), "whisky_ml_" + self.default_name)
        try:
            self.image.save(path)
            open_path(path)
        except OSError as e:
            messagebox.showerror(self.app.title(), f"Couldn't open:\n{e}", parent=self)


class ImageList:
    """Card with a list of image files: Add files / Add folder / From Gallery / Clear."""

    def __init__(self, tab, parent, title, columns, on_select=None, on_change=None):
        self.tab, self.on_change = tab, on_change
        C, F, app = tab.C, tab.F, tab.app
        self.paths = []
        self.card = card = ctk.CTkFrame(parent, fg_color=C["surface"], corner_radius=28)
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=(18, 12), pady=(14, 6))
        head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(head, text=title, font=F.semi(15), text_color=C["text"]).grid(row=0, column=0)
        self.count = ctk.CTkLabel(head, text="", font=F.body(13), text_color=C["muted"])
        self.count.grid(row=0, column=1, sticky="w", padx=8)
        links = ctk.CTkFrame(head, fg_color="transparent")
        links.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        for text, cmd in (("Add files", self.add_files), ("Add folder", self.add_folder),
                          ("From Gallery", self.from_gallery), ("Remove", self.remove_selected),
                          ("Clear", self.clear)):
            app._link(links, text, cmd).pack(side="left")

        holder = tk.Frame(card, bg=C["surface"], highlightthickness=0)
        holder.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 14))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(holder, columns=[c for c, *_ in columns], show="headings",
                                 selectmode="extended", style="WW.Treeview")
        for col, text, w, anchor, stretch in columns:
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=w, minwidth=40, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(holder, command=self.tree.yview, fg_color=C["surface"],
                              button_color=C["line"], button_hover_color=C["muted"])
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        for tag, colour in (("ok", C["sage_ink"]), ("bad", C["error"]), ("wait", C["muted"]),
                            ("warn", C["accent_deep"]), ("plain", C["text"])):
            self.tree.tag_configure(tag, foreground=colour)
        if on_select:
            self.tree.bind("<<TreeviewSelect>>", lambda e: on_select())

    def grid(self, **kw):
        self.card.grid(**kw)

    def selected(self):
        return [self.paths[int(i)] for i in self.tree.selection()]

    def add(self, paths):
        known = set(self.paths)
        new = [p for p in paths if p not in known]
        self.paths += new
        self.refresh()
        if new and not self.tree.selection():
            self.tree.selection_set(str(self.paths.index(new[0])))
        return len(new)

    def refresh(self):
        sel = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        for i, p in enumerate(self.paths):
            values, tag = self.tab.row_for(self, p)
            self.tree.insert("", "end", iid=str(i), values=values, tags=(tag,))
        keep = [i for i in sel if self.tree.exists(i)]
        if keep:
            self.tree.selection_set(keep)
        self.count.configure(text=f"{len(self.paths)} image{'s' if len(self.paths) != 1 else ''}"
                             if self.paths else "")
        if self.on_change:
            self.on_change()

    def update_row(self, path):
        if path in self.paths:
            i = str(self.paths.index(path))
            values, tag = self.tab.row_for(self, path)
            self.tree.item(i, values=values, tags=(tag,))

    def add_files(self):
        fs = filedialog.askopenfilenames(title="Choose stitched images",
                                         filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                                                    ("All files", "*.*")])
        if fs:
            self.add([os.path.normpath(f) for f in fs])

    def add_folder(self):
        d = filedialog.askdirectory(title="Choose a folder of stitched images")
        if not d:
            return
        try:
            found = images_in(d)
        except OSError as e:
            messagebox.showerror(self.tab.app.title(), f"Couldn't read the folder:\n{e}")
            return
        if not found:
            messagebox.showinfo(self.tab.app.title(), "No images in that folder.")
            return
        self.add(found)

    def from_gallery(self):
        g = self.tab.app.gallery
        sel = sorted(g.selected) if g is not None else []
        if not sel:
            messagebox.showinfo(self.tab.app.title(), "Nothing is ticked in the Gallery.\n\nOpen the Gallery, "
                                                      "press \"Select\", tick some pictures, then come back.")
            return
        n = self.add(sel)
        self.tab.log(f"Added {n} image(s) from the Gallery selection")

    def remove_selected(self):
        drop = set(self.selected())
        self.paths = [p for p in self.paths if p not in drop]
        self.refresh()

    def clear(self):
        self.paths = []
        self.refresh()


# --------------------------------------------------------------------------- #
#  Tab
# --------------------------------------------------------------------------- #
class MLTab:
    def __init__(self, app, C, identify):
        self.app, self.C, self.F, self.identify = app, C, app.F, identify
        s = app.settings
        self.page = tk.StringVar(value=s.get("ml_page", "Explore") if s.get("ml_page") in PAGES else "Explore")
        self.view = tk.StringVar(value="Rim")
        self.view_norm = tk.BooleanVar(value=True)
        fs = s.get("ml_feature_set", "scalar+radial")
        if fs not in SETS and not (fs in mc.CNN_SETS and not mc.cnn_available()):
            fs = "scalar+radial"
        self.feature_set = tk.StringVar(value=fs)
        self.normalised = tk.BooleanVar(value=bool(s.get("ml_normalised", False)))
        self.folds = tk.IntVar(value=5)
        self.min_per_class = tk.IntVar(value=5)
        self.include_failed = tk.BooleanVar(value=False)
        self.compare_norm = tk.BooleanVar(value=False)
        self.as_sample = tk.BooleanVar(value=True)
        self.model_path = s.get("ml_model", "")

        self.droplets = OrderedDict()   # path -> Droplet (Explore)
        self.failed = {}                # path -> error text
        self.analysing = None
        self.summaries = {}             # (label, include_failed) -> SampleSummary
        self.summary = None             # shown on the Sample page
        self.dataset = None
        self.cv = None                  # last CVResult
        self.cv_runs = []               # every CVResult this session (for Compare)
        self.cv_figure = None
        self.model = None
        self.preds = {}                 # path -> Prediction
        self.sample_pred = None
        self.worker = None
        self.stop_flag = threading.Event()
        self._photos = {}
        self._build()
        self._show_page()
        if self.model_path and os.path.isfile(self.model_path):
            try:
                self._set_model(mc.load_model(self.model_path), self.model_path, quiet=True)
            except Exception as e:  # noqa: BLE001
                self.log(f"Couldn't load the last model ({self.model_path}): {e}")

    # ================================================================== helpers
    def whisky(self, label):
        return self.identify.whisky_full(label) if label else ""

    def names(self, label):
        """For the figures: the whisky name from the key, or ''."""
        return self.identify.names.get(label, "")

    def folder(self):
        return self.identify.db_folder.get().strip()

    def log(self, msg):
        self.app.logs.append(msg)
        if msg.strip():
            self.status.configure(text=msg.strip())

    def busy(self):
        return bool(self.worker and self.worker.is_alive())

    def settings(self):
        return dict(ml_page=self.page.get(), ml_feature_set=self.feature_set.get(),
                    ml_normalised=bool(self.normalised.get()), ml_model=self.model_path)

    def _card(self, parent, title, badge=None):
        C, F = self.C, self.F
        card = ctk.CTkFrame(parent, fg_color=C["surface"], corner_radius=28)
        card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 6))
        head.grid_columnconfigure(2, weight=1)
        if badge:
            self.app._badge(head, badge).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkLabel(head, text=title, font=F.semi(15), text_color=C["text"]).grid(row=0, column=1)
        card.head = head
        return card

    def _text(self, parent, height=120, mono=True):
        C = self.C
        box = ctk.CTkTextbox(parent, fg_color=C["bg"], text_color=C["text"], corner_radius=16, height=height,
                             font=ctk.CTkFont(family="Consolas", size=12) if mono else self.F.body(13),
                             wrap="none" if mono else "word", border_width=0)
        box.configure(state="disabled")
        return box

    @staticmethod
    def _set_text(box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _button(self, parent, text, command, primary=False, width=None):
        C, F = self.C, self.F
        if primary:
            b = ctk.CTkButton(parent, text=text, command=command, height=40, corner_radius=20,
                              fg_color=C["accent"], hover_color=C["accent_hover"], text_color=C["bg"],
                              text_color_disabled=C["bg"], font=F.head(14))
        else:
            b = self.app._pill_button(parent, text, command, height=38, size=13,
                                      text_color_disabled=C["line"])
        if width:
            b.configure(width=width)
        return b

    def _well(self, parent, title):
        """Dark preview panel: (frame, canvas, title label, note label)."""
        C, F = self.C, self.F
        w = ctk.CTkFrame(parent, fg_color=C["well"], corner_radius=28)
        w.grid_rowconfigure(1, weight=1)
        w.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(w, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=(24, 16), pady=(14, 0))
        top.grid_columnconfigure(0, weight=1)
        lab = ctk.CTkLabel(top, text=title, font=F.head(17), text_color=C["bg"], anchor="w")
        lab.grid(row=0, column=0, sticky="w", pady=4)
        cv = tk.Canvas(w, background=C["well"], highlightthickness=0)
        cv.grid(row=1, column=0, sticky="nsew", padx=18, pady=8)
        note = ctk.CTkLabel(w, text="", font=F.body(12), text_color=C["well_faint"], anchor="w",
                            justify="left", wraplength=1050)
        note.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 14))
        w.top = top
        return w, cv, lab, note

    def _show_image(self, cv, im, key, message=None):
        """Draw a PIL image fitted to canvas `cv` (or a message)."""
        cv.delete("all")
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        if im is None:
            cv.create_text(W / 2, H / 2, text=message or "", fill=self.C["well_text"], width=W - 40,
                           font=(self.F.body_family, 12), justify="center")
            return
        ph = self._photos[key] = ImageTk.PhotoImage(fit_to(im, W - 8, H - 8))
        cv.create_image(W / 2, H / 2, image=ph)

    def _figure(self, title, make, name):
        """Make a matplotlib figure in the background and open it in a window."""
        if not mc.figures_available():
            messagebox.showinfo(self.app.title(), "Figures need matplotlib.\n\nRun:  pip install matplotlib")
            return
        self._run(f"Making the {title.lower()}…", make,
                  lambda im: FigureWindow(self.app, self.C, title, im, name))

    # ================================================================== layout
    def _build(self):
        C, F, app = self.C, self.F, self.app
        self.body = ctk.CTkFrame(app, fg_color=C["bg"], corner_radius=0)
        self.body.grid_rowconfigure(1, weight=1)
        self.body.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self.body, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        head.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(head, text="ML", font=F.head(30), text_color=C["accent"]).grid(row=0, column=0,
                                                                                  padx=(4, 16))
        ctk.CTkSegmentedButton(head, values=PAGES, variable=self.page, command=lambda v: self._show_page(),
                               font=F.semi(14), height=36, corner_radius=18, fg_color=C["surface"],
                               selected_color=C["accent"], selected_hover_color=C["accent_hover"],
                               unselected_color=C["surface"], unselected_hover_color=C["hover_tint"],
                               text_color=C["text"]).grid(row=0, column=1)
        self.head_note = ctk.CTkLabel(head, text="", font=F.body(13), text_color=C["muted"], anchor="e")
        self.head_note.grid(row=0, column=2, sticky="e", padx=(12, 4))

        self.pages = {}
        for name, build in (("Explore", self._build_explore), ("Sample average", self._build_sample),
                            ("Train & evaluate", self._build_train), ("Predict", self._build_predict)):
            f = ctk.CTkFrame(self.body, fg_color=C["bg"], corner_radius=0)
            f.grid(row=1, column=0, sticky="nsew")
            f.grid_rowconfigure(0, weight=1)
            f.grid_columnconfigure(1, weight=1)
            build(f)
            f.grid_remove()
            self.pages[name] = f

        # -- run bar
        self.runbar = bar = ctk.CTkFrame(app, fg_color=C["surface"], corner_radius=0)
        bar.grid_columnconfigure(1, weight=1)
        self.stop_btn = app._pill_button(bar, "Stop", self.stop, height=48, size=15, width=96,
                                         text_color_disabled=C["line"])
        self.stop_btn.grid(row=0, column=0, padx=(28, 12), pady=14)
        self.stop_btn.configure(state="disabled")
        mid = ctk.CTkFrame(bar, fg_color="transparent")
        mid.grid(row=0, column=1, sticky="ew", padx=(6, 28))
        mid.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(mid, height=8, corner_radius=4, fg_color=C["track"],
                                           progress_color=C["track"])
        self.progress.set(0)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.status = ctk.CTkLabel(mid, text="", anchor="w", font=F.body(13), text_color=C["soft"])
        self.status.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        app._link(mid, "Show log", app.show_log, 12).grid(row=1, column=1, pady=(4, 0))
        if not mc.figures_available():
            self.status.configure(text="Figures are off: matplotlib isn't installed (pip install matplotlib)")

    def show(self):
        self.body.grid(row=1, column=0, sticky="nsew", padx=28, pady=20)
        self.runbar.grid(row=2, column=0, sticky="ew")
        self._show_page()

    def hide(self):
        self.body.grid_remove()
        self.runbar.grid_remove()

    def _show_page(self):
        name = self.page.get()
        for n, f in self.pages.items():
            f.grid() if n == name else f.grid_remove()
        folder = self.folder()
        if name in ("Sample average", "Train & evaluate"):
            shown = folder if len(folder) <= 70 else "…" + folder[-69:]
            self.head_note.configure(text=f"Reference folder: {shown}" if folder else
                                     "Choose the reference folder on the Identify tab")
        else:
            self.head_note.configure(text="")
        if name == "Sample average":
            self._fill_whiskies()
        self.app.after(30, self._redraw)

    def _redraw(self):
        name = self.page.get()
        if name == "Explore":
            self._draw_explore()
        elif name == "Sample average":
            self._draw_sample()
        elif name == "Train & evaluate":
            self._draw_cv_figure()

    # ------------------------------------------------------------- Explore
    def _build_explore(self, page):
        C, F = self.C, self.F
        left = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0, width=460)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        left.grid_propagate(False)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.ex_list = ImageList(self, left, "Images", [
            ("image", "IMAGE", 150, "w", True), ("whisky", "WHISKY", 120, "w", True),
            ("qc", "QC", 130, "w", False)], on_select=self._explore_select)
        self.ex_list.grid(row=0, column=0, sticky="nsew")

        figs = self._card(left, "Figures of the selected images")
        figs.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        ctk.CTkLabel(figs, text=f"One row per image (up to {FIG_ROWS}); Ctrl/Shift-click for several.",
                     font=F.body(12), text_color=C["muted"], anchor="w").grid(
            row=1, column=0, sticky="ew", padx=20)
        row = ctk.CTkFrame(figs, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 16))
        self.fig_buttons = []
        for text, fn in (("Pipeline", self.fig_pipeline), ("Normalisation", self.fig_normalisation),
                         ("Profiles", self.fig_profiles)):
            b = self._button(row, text, fn)
            b.pack(side="left", padx=(0, 8))
            self.fig_buttons.append(b)
        if not mc.figures_available():
            for b in self.fig_buttons:
                b.configure(state="disabled")

        right = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.ex_well, self.ex_cv, self.ex_title, self.ex_note = self._well(right, "Droplet")
        self.ex_well.grid(row=0, column=0, sticky="nsew")
        self.ex_cv.bind("<Configure>", lambda e: self._draw_explore())
        sw = ctk.CTkFrame(self.ex_well.top, fg_color="transparent")
        sw.grid(row=0, column=1, sticky="e")
        ctk.CTkSegmentedButton(sw, values=VIEWS, variable=self.view, command=lambda v: self._draw_explore(),
                               font=F.semi(12), height=30, corner_radius=15, fg_color=C["seg"],
                               selected_color=C["accent"], selected_hover_color=C["accent_hover"],
                               unselected_color=C["seg"], unselected_hover_color=C["soft"],
                               text_color=C["bg"]).pack(side="left")
        ctk.CTkCheckBox(sw, text="Normalised", variable=self.view_norm, command=self._draw_explore,
                        font=F.body(12), text_color=C["bg"], fg_color=C["accent"], border_color=C["well_faint"],
                        hover_color=C["accent_hover"], checkbox_width=18, checkbox_height=18).pack(
            side="left", padx=(10, 0))
        self.ex_save = self.app._link(sw, "Save view", self.save_view, 12)
        self.ex_save.configure(text_color=C["bg"], hover_color=C["soft"])
        self.ex_save.pack(side="left", padx=(6, 0))

        info = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        info.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        info.grid_columnconfigure((0, 1), weight=1, uniform="info")
        self.ex_qc = ctk.CTkLabel(info, text="", font=F.semi(14), text_color=C["text"], anchor="w",
                                  justify="left", wraplength=900)
        self.ex_qc.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(14, 6))
        ctk.CTkLabel(info, text="Normalisation (R / G / B, 0–255)", font=F.semi(13), text_color=C["soft"],
                     anchor="w").grid(row=1, column=0, sticky="ew", padx=20)
        ctk.CTkLabel(info, text="Features (raw · normalised)", font=F.semi(13), text_color=C["soft"],
                     anchor="w").grid(row=1, column=1, sticky="ew", padx=(0, 20))
        self.ex_norm = self._text(info, 150)
        self.ex_norm.grid(row=2, column=0, sticky="ew", padx=(16, 8), pady=(4, 16))
        self.ex_feats = self._text(info, 150)
        self.ex_feats.grid(row=2, column=1, sticky="ew", padx=(8, 16), pady=(4, 16))

    def row_for(self, lst, path):
        """(values, tag) of one row of an ImageList."""
        name = os.path.splitext(os.path.basename(path))[0].replace("_stitched", "")
        label = mc.label_of(path)
        if lst is self.ex_list:
            d = self.droplets.get(path)
            if path in self.failed:
                return (name, self.whisky(label), "couldn't open"), "bad"
            if d is None:
                return (name, self.whisky(label), "…" if path == self.analysing else ""), "wait"
            return (name, self.whisky(label), "✓ passed" if d.qc_ok else "✗ " + d.qc_reason.split(" (")[0]), \
                ("ok" if d.qc_ok else "bad")
        p = self.preds.get(path)
        if p is None:
            return (name, "", "", ""), "wait"
        if not p.top:
            return (name, "—", "", p.qc_reason or "no droplet"), "bad"
        tag = "bad" if not p.qc_ok else ("warn" if p.unknown or p.margin < 0.1 else "ok")
        return (name, self.whisky(p.label), f"{p.probability:.2f}", p.confidence), tag

    def _explore_select(self):
        sel = self.ex_list.selected()
        if not sel:
            self._draw_explore()
            return
        path = sel[0]                       # the view shows the first selected image
        if path not in self.droplets and path not in self.failed:
            self._analyse(path)
        self._draw_explore()

    def _analyse(self, path):
        if self.busy():
            return
        problem = idc.opencv_problem()
        if problem:
            messagebox.showwarning(self.app.title(), problem)
            return
        self.analysing = path
        self.ex_list.update_row(path)

        def done(d):
            self.analysing = None
            if isinstance(d, str):
                self.failed[path] = d
            else:
                self.droplets[path] = d
                self.droplets.move_to_end(path)
                while len(self.droplets) > DROPLET_CACHE:
                    self.droplets.popitem(last=False)
            self.ex_list.update_row(path)
            self._draw_explore()

        def work():
            try:
                return mc.analyse(path)
            except InterruptedError:
                raise
            except Exception as e:  # noqa: BLE001
                return str(e)
        self._run(f"Analysing {os.path.basename(path)}…", work, done, quiet=True)

    def _current_droplet(self):
        sel = self.ex_list.selected()
        if not sel:
            return None, None
        path = sel[0]
        return path, self.droplets.get(path)

    def _view_image(self, d):
        v, norm = self.view.get(), bool(self.view_norm.get())
        if d.seg is None and v not in ("Raw", "Ring map"):
            return mc.view_raw(d), "No rim found – showing the raw image"
        fn = {"Raw": lambda: mc.view_raw(d), "Rim": lambda: mc.view_rim(d),
              "Ring map": lambda: mc.view_ring_map(d), "Background": lambda: mc.view_background(d),
              "Subtracted": lambda: mc.view_subtracted(d), "Normalised": lambda: mc.view_normalised(d),
              "Unwrap": lambda: mc.view_unwrap(d, normalised=norm),
              "Crop": lambda: mc.view_crop(d, normalised=norm)}[v]
        notes = {"Raw": "Stitched image at ¼ size, brightened ×2.5",
                 "Rim": "Fitted rim: green = passes QC, red = fails",
                 "Ring map": "Ring map (magma) with the rim (cyan) and centre",
                 "Background": "Normalisation step 1: background (blue) beyond 1.25 R (dashed)",
                 "Subtracted": "Step 2: background subtracted (same gain as Raw)",
                 "Normalised": "Step 3: divided by the droplet's own mean",
                 "Unwrap": f"Radial unwrap ({'normalised' if norm else 'raw'}): angle 0–360° left to right, "
                           "centre at the bottom, rim at the dashed line",
                 "Crop": f"Droplet-centred crop ({'normalised' if norm else 'raw'})"}
        return fn(), notes[v]

    def _draw_explore(self):
        if self.page.get() != "Explore":
            return
        cv = self.ex_cv
        path, d = self._current_droplet()
        if path is None:
            self.ex_title.configure(text="Droplet")
            self.ex_note.configure(text="")
            self.ex_qc.configure(text="")
            self._set_text(self.ex_norm, "")
            self._set_text(self.ex_feats, "")
            return self._show_image(cv, None, "ex", "Add images on the left (files, a folder or the "
                                                    "Gallery selection), then click one")
        name = os.path.splitext(os.path.basename(path))[0]
        self.ex_title.configure(text=f"{name}  ·  {self.whisky(mc.label_of(path))}")
        if path in self.failed:
            self.ex_qc.configure(text="")
            return self._show_image(cv, None, "ex", f"Couldn't analyse this image:\n{self.failed[path]}")
        if d is None:
            msg = "Analysing… (1–3 s, longer over the network)" if self.analysing == path else \
                "Busy with another job – it'll be analysed when you click it again" if self.busy() else ""
            return self._show_image(cv, None, "ex", msg)
        try:
            arr, note = self._view_image(d)
            self._show_image(cv, Image.fromarray(arr), "ex")
            self.ex_note.configure(text=note)
        except Exception as e:  # noqa: BLE001
            self._show_image(cv, None, "ex", f"Couldn't draw this view:\n{e}")
        if d.qc_ok:
            self.ex_qc.configure(text="✓  Passes QC", text_color=self.C["sage_ink"])
        else:
            self.ex_qc.configure(text=f"✗  Fails QC: {d.qc_reason}", text_color=self.C["error"])
        self._set_text(self.ex_norm, self._norm_text(d))
        self._set_text(self.ex_feats, self._feature_text(d))

    @staticmethod
    def _norm_text(d):
        if d.seg is None:
            return "No rim found, so nothing to normalise."
        n = mc.normalisation_numbers(d)
        rgb = lambda v: " / ".join(f"{x:6.1f}" for x in v)  # noqa: E731
        lines = [f"{'Background':<24}{rgb(n['background'])}",
                 f"{'Droplet mean (raw)':<24}{rgb(n['droplet_mean_raw'])}",
                 f"{'… after subtraction':<24}{rgb(n['droplet_mean_after_subtraction'])}",
                 f"{'Gain':<24}" + " / ".join(f"{x:6.2f}" for x in n["gain"]),
                 f"{'Background share':<24}{100 * n['background_share_of_image']:6.1f} % of the image"]
        seg = d.seg
        lines += ["", f"{'Ring score':<24}{seg['score']:.3f}   (QC needs ≥ {mc.QC_MIN_SCORE})",
                  f"{'Outside droplet':<24}{100 * seg['outside_frac']:.0f} %   (QC allows ≤ "
                  f"{100 * mc.QC_MAX_OUTSIDE:.0f} %)"]
        return "\n".join(lines)

    @staticmethod
    def _feature_text(d):
        if not d.scalars:
            return "No features (no rim found)."
        w = max(len(k) for k in mc.SCALAR_NAMES)
        return "\n".join(f"{k:<{w}}  {d.scalars[k]:>10.4g}  {d.scalars_norm[k]:>10.4g}" for k in mc.SCALAR_NAMES)

    def save_view(self):
        path, d = self._current_droplet()
        if d is None:
            return
        arr, _ = self._view_image(d)
        name = os.path.splitext(os.path.basename(path))[0] + f"_{self.view.get().lower().replace(' ', '_')}.png"
        f = filedialog.asksaveasfilename(title="Save view", defaultextension=".png", initialfile=name,
                                         filetypes=[("PNG", "*.png")])
        if f:
            Image.fromarray(arr).save(f)
            self.log(f"Saved {f}")

    def _fig_paths(self):
        sel = self.ex_list.selected()
        if not sel:
            messagebox.showinfo(self.app.title(), "Select one or more images in the list first.")
            return None
        if len(sel) > FIG_ROWS:
            self.log(f"Using the first {FIG_ROWS} of {len(sel)} selected images (one figure row each)")
        return sel[:FIG_ROWS]

    def _droplets_for(self, paths):
        """Droplets for `paths` (in the worker thread; analyses the ones not done yet)."""
        out = []
        for i, p in enumerate(paths):
            if self.stop_flag.is_set():
                raise InterruptedError("stopped")
            d = self.droplets.get(p) or mc.analyse(p)
            out.append(d)
            self.app.events.put(("ml_progress", (i + 1) / len(paths)))
        return out

    def _remember(self, ds):
        for d in ds:
            self.droplets[d.path] = d
            self.ex_list.update_row(d.path)

    def fig_pipeline(self):
        paths = self._fig_paths()
        if paths:
            def make():
                ds = self._droplets_for(paths)
                self.app.events.put(("ml_call", (self._remember, ds)))
                return mc.figure_pipeline(ds, self.names)
            self._figure("Pipeline figure", make, "pipeline.png")

    def fig_normalisation(self):
        paths = self._fig_paths()
        if paths:
            def make():
                ds = self._droplets_for(paths)
                self.app.events.put(("ml_call", (self._remember, ds)))
                return mc.figure_normalisation(ds, self.names)
            self._figure("Normalisation figure", make, "normalisation.png")

    def fig_profiles(self):
        sel = self.ex_list.selected()
        if not sel:
            messagebox.showinfo(self.app.title(), "Select one or more images in the list first.")
            return
        norm = bool(self.view_norm.get())
        self._figure("Profiles figure", lambda: mc.figure_profiles(
            [r for r in mc.features_many(sel, should_stop=self.stop_flag.is_set, log=self._bg_log)[0] if r],
            self.names, normalised=norm), "profiles.png")

    # ------------------------------------------------------- Sample average
    def _build_sample(self, page):
        C, F = self.C, self.F
        left = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0, width=460)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        left.grid_propagate(False)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        card = self._card(left, "Whiskies")
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        self.app._link(card.head, "Refresh", lambda: self._fill_whiskies(fresh=True)).grid(row=0, column=3)
        holder = tk.Frame(card, bg=C["surface"], highlightthickness=0)
        holder.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        self.wh_tree = ttk.Treeview(holder, columns=("whisky", "images"), show="headings",
                                    selectmode="extended", style="WW.Treeview")
        self.wh_tree.heading("whisky", text="WHISKY", anchor="w")
        self.wh_tree.heading("images", text="IMAGES", anchor="center")
        self.wh_tree.column("whisky", width=300, anchor="w", stretch=True)
        self.wh_tree.column("images", width=70, anchor="center", stretch=False)
        self.wh_tree.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(holder, command=self.wh_tree.yview, fg_color=C["surface"],
                              button_color=C["line"], button_hover_color=C["muted"])
        sb.grid(row=0, column=1, sticky="ns")
        self.wh_tree.configure(yscrollcommand=sb.set)
        self.wh_tree.bind("<Double-1>", lambda e: self.average())
        self.wh_paths = {}

        opts = ctk.CTkFrame(card, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 4))
        ctk.CTkCheckBox(opts, text="Include droplets that failed QC", variable=self.include_failed,
                        font=F.body(13), text_color=C["text"], fg_color=C["accent"],
                        hover_color=C["accent_hover"], border_color=C["line"], checkbox_width=18,
                        checkbox_height=18).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(opts, text="Compare normalised profiles", variable=self.compare_norm,
                        font=F.body(13), text_color=C["text"], fg_color=C["accent"],
                        hover_color=C["accent_hover"], border_color=C["line"], checkbox_width=18,
                        checkbox_height=18).pack(anchor="w", pady=2)
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.grid(row=3, column=0, sticky="ew", padx=16, pady=(6, 16))
        self._button(row, "Average", self.average, primary=True, width=130).pack(side="left")
        b = self._button(row, "Compare selected", self.compare)
        b.pack(side="left", padx=8)
        b2 = self._button(row, "Figure", self.fig_sample)
        b2.pack(side="left")
        if not mc.figures_available():
            b.configure(state="disabled")
            b2.configure(state="disabled")
        ctk.CTkLabel(card, text="The first average of a whisky analyses all its images (cached afterwards).",
                     font=F.body(12), text_color=C["muted"], anchor="w", wraplength=400, justify="left").grid(
            row=4, column=0, sticky="ew", padx=20, pady=(0, 14))

        right = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.sa_well, self.sa_cv, self.sa_title, self.sa_note = self._well(right, "Sample average")
        self.sa_well.grid(row=0, column=0, sticky="nsew")
        self.sa_cv.bind("<Configure>", lambda e: self._draw_sample())
        info = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        info.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        info.grid_columnconfigure(0, weight=2, uniform="sa")
        info.grid_columnconfigure(1, weight=3, uniform="sa")
        self.sa_head = ctk.CTkLabel(info, text="", font=F.semi(14), text_color=C["text"], anchor="w")
        self.sa_head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(14, 6))
        ctk.CTkLabel(info, text="Rejected by QC", font=F.semi(13), text_color=C["soft"], anchor="w").grid(
            row=1, column=0, sticky="ew", padx=20)
        ctk.CTkLabel(info, text="Features: mean ± SD", font=F.semi(13), text_color=C["soft"],
                     anchor="w").grid(row=1, column=1, sticky="ew", padx=(0, 20))
        self.sa_rej = self._text(info, 160)
        self.sa_rej.configure(wrap="word")
        self.sa_rej.grid(row=2, column=0, sticky="ew", padx=(16, 8), pady=(4, 16))
        self.sa_feats = self._text(info, 160)
        self.sa_feats.grid(row=2, column=1, sticky="ew", padx=(8, 16), pady=(4, 16))

    def _fill_whiskies(self, fresh=False):
        folder = self.folder()
        self.wh_tree.delete(*self.wh_tree.get_children())
        self.wh_paths = {}
        if not folder or not os.path.isdir(folder):
            return
        try:
            classes, _ = idc.scan_classes(folder, fresh=fresh)
        except OSError as e:
            self.log(f"Couldn't read the reference folder: {e}")
            return
        for label, paths in classes:
            self.wh_paths[label] = paths
            self.wh_tree.insert("", "end", iid=label, values=(self.whisky(label), len(paths)))
        if self.summary is not None and self.wh_tree.exists(self.summary.label):
            self.wh_tree.selection_set(self.summary.label)

    def _summaries(self, labels):
        """SampleSummary per label (worker thread)."""
        inc = bool(self.include_failed.get())
        out, todo = [], []
        for lab in labels:
            if (lab, inc) not in self.summaries:
                todo += self.wh_paths[lab]
        total, done = max(1, len(todo)), [0]

        def progress(d, t):
            self.app.events.put(("ml_progress", (done[0] + d) / total))
            if d % 25 == 0 or d == t:
                self.app.events.put(("ml_status", f"Analysing droplets: {done[0] + d} / {total} "
                                                  "(only slow the first time – they're remembered)"))
        for lab in labels:
            key = (lab, inc)
            if key not in self.summaries:
                paths = self.wh_paths[lab]
                recs, _ = mc.features_many(paths, [lab] * len(paths), log=self._bg_log, progress=progress,
                                           should_stop=self.stop_flag.is_set)
                done[0] += len(paths)
                self.summaries[key] = mc.summarise(recs, lab, include_failed=inc)
            out.append(self.summaries[key])
        return out

    def average(self):
        sel = self.wh_tree.selection()
        if not sel:
            messagebox.showinfo(self.app.title(), "Click a whisky first.")
            return

        def done(res):
            self.summary = res[0]
            self._draw_sample()
        self._run(f"Averaging {self.whisky(sel[0])}…", lambda: self._summaries([sel[0]]), done)

    def compare(self):
        sel = list(self.wh_tree.selection())
        if len(sel) < 2:
            messagebox.showinfo(self.app.title(), "Select two or more whiskies (Ctrl/Shift-click) to compare.")
            return
        norm = bool(self.compare_norm.get())
        self._figure("Profiles of the selected whiskies",
                     lambda: mc.figure_profiles(self._summaries(sel), self.names, normalised=norm),
                     "compare_profiles.png")

    def fig_sample(self):
        if self.summary is None:
            messagebox.showinfo(self.app.title(), "Press Average first.")
            return
        s = self.summary
        self._figure(f"Sample average – {s.label}", lambda: mc.figure_sample_average(s, self.names),
                     f"sample_average_{s.label}.png")

    def _sample_image(self, s):
        """Average droplet (raw), (normalised) and mean unwrap side by side."""
        def show(a, gain):
            return Image.fromarray((np.clip(a.astype(np.float32) / 255 * gain, 0, 1) * 255).astype(np.uint8))
        H = 320
        a = show(s.mean_droplet, 2.5).resize((H, H), Image.BICUBIC)
        b = show(s.mean_droplet_norm, 1.6).resize((H, H), Image.BICUBIC)
        u = show(s.mean_unwrap_norm[::-1], 1.6).resize((int(H * 720 / 256), H), Image.BICUBIC)
        gap = 24
        out = Image.new("RGB", (a.width + b.width + u.width + 2 * gap, H), self.C["well"])
        out.paste(a, (0, 0))
        out.paste(b, (a.width + gap, 0))
        out.paste(u, (a.width + b.width + 2 * gap, 0))
        return out

    def _draw_sample(self):
        if self.page.get() != "Sample average":
            return
        s = self.summary
        if s is None:
            self.sa_title.configure(text="Sample average")
            self.sa_note.configure(text="")
            self.sa_head.configure(text="")
            self._set_text(self.sa_rej, "")
            self._set_text(self.sa_feats, "")
            return self._show_image(self.sa_cv, None, "sa", "Pick a whisky on the left and press Average")
        self.sa_title.configure(text=self.whisky(s.label))
        self._show_image(self.sa_cv, self._sample_image(s), "sa")
        self.sa_note.configure(text="Left to right: average droplet (raw, ×2.5) · average droplet "
                                    "(normalised, ×1.6) · mean unwrap (normalised; angle left to right, "
                                    "centre at the bottom). Rotations are random, so the average shows "
                                    "the radial structure.")
        self.sa_head.configure(text=f"{s.n_used} of {s.n_images} droplets used"
                                    + (" (failed QC included)" if s.n_used > s.n_images - len(s.rejected) else ""))
        self._set_text(self.sa_rej, "\n".join(f"{n}: {r}" for n, r in s.rejected) or "None – every droplet passed.")
        w = max(len(k) for k in mc.SCALAR_NAMES)
        self._set_text(self.sa_feats, "\n".join(f"{k:<{w}}  {s.scalar_mean[k]:>10.4g} ± {s.scalar_sd[k]:<10.3g}"
                                                for k in mc.SCALAR_NAMES))

    # ------------------------------------------------------ Train & evaluate
    def _build_train(self, page):
        C, F = self.C, self.F
        left = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0, width=460)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        left.grid_propagate(False)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        data = self._card(left, "Dataset", 1)
        data.grid(row=0, column=0, sticky="nsew")
        data.grid_rowconfigure(2, weight=1)
        self.ds_btn = self._button(data.head, "Load dataset", self.load_dataset, primary=True, width=140)
        self.ds_btn.grid(row=0, column=3)
        self.ds_info = ctk.CTkLabel(data, text="Analyses every reference image. The first time takes a long "
                                               "time (roughly 30–60 min for 2,400 images over the network); "
                                               "after that it takes seconds.",
                                    font=F.body(12), text_color=C["muted"], anchor="w", justify="left",
                                    wraplength=400)
        self.ds_info.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 6))
        holder = tk.Frame(data, bg=C["surface"], highlightthickness=0)
        holder.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=(0, 14))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        self.qc_tree = ttk.Treeview(holder, columns=("whisky", "images", "passed", "recall"), show="headings",
                                    selectmode="none", style="WW.Treeview")
        for col, text, w, anchor, stretch in (("whisky", "WHISKY", 200, "w", True),
                                              ("images", "IMAGES", 62, "center", False),
                                              ("passed", "PASSED QC", 80, "center", False),
                                              ("recall", "RECALL", 64, "center", False)):
            self.qc_tree.heading(col, text=text, anchor=anchor)
            self.qc_tree.column(col, width=w, minwidth=40, anchor=anchor, stretch=stretch)
        self.qc_tree.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(holder, command=self.qc_tree.yview, fg_color=C["surface"],
                              button_color=C["line"], button_hover_color=C["muted"])
        sb.grid(row=0, column=1, sticky="ns")
        self.qc_tree.configure(yscrollcommand=sb.set)
        for tag, colour in (("ok", C["sage_ink"]), ("bad", C["error"]), ("warn", C["accent_deep"])):
            self.qc_tree.tag_configure(tag, foreground=colour)

        opts = self._card(left, "Model", 2)
        opts.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        grid = ctk.CTkFrame(opts, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="ew", padx=20)
        grid.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(grid, text="Features", font=F.body(13), text_color=C["muted"]).grid(row=0, column=0,
                                                                                         sticky="w")
        ctk.CTkSegmentedButton(grid, values=SETS, variable=self.feature_set, font=F.body(12), height=30,
                               corner_radius=15, fg_color=C["bg"], selected_color=C["accent"],
                               selected_hover_color=C["accent_hover"], unselected_color=C["bg"],
                               unselected_hover_color=C["hover_tint"], text_color=C["text"]).grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=3)
        ctk.CTkLabel(grid, text="Pretrained CNN", font=F.body(13), text_color=C["muted"]).grid(
            row=1, column=0, sticky="w")
        cnn_problem = mc.cnn_available()
        cnn_row = ctk.CTkFrame(grid, fg_color="transparent")
        cnn_row.grid(row=1, column=1, sticky="w", padx=(12, 0), pady=3)
        self.cnn_seg = ctk.CTkSegmentedButton(
            cnn_row, values=list(mc.CNN_SETS), variable=self.feature_set, font=F.body(12),
            height=30, corner_radius=15, fg_color=C["bg"], selected_color=C["accent"],
            selected_hover_color=C["accent_hover"], unselected_color=C["bg"],
            unselected_hover_color=C["hover_tint"], text_color=C["text"],
            state="disabled" if cnn_problem else "normal")
        self.cnn_seg.pack(side="left")
        self.cnn_install = None
        if cnn_problem:
            self.cnn_install = self.app._link(cnn_row, "Install PyTorch…", self.install_torch, 12)
            self.cnn_install.pack(side="left", padx=(10, 0))
        self.cnn_note = ctk.CTkLabel(
            grid, text=(CNN_MISSING if cnn_problem else CNN_NOTE), font=F.body(11), text_color=C["muted"],
            anchor="w", justify="left", wraplength=290)
        self.cnn_note.grid(row=2, column=1, sticky="w", padx=(12, 0))
        ctk.CTkCheckBox(grid, text="Normalised (ignores exposure and lamp colour; usually a few points lower)",
                        variable=self.normalised, font=F.body(12), text_color=C["text"], fg_color=C["accent"],
                        hover_color=C["accent_hover"], border_color=C["line"], checkbox_width=18,
                        checkbox_height=18).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        nums = ctk.CTkFrame(grid, fg_color="transparent")
        nums.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 4))
        for i, (text, var, vals) in enumerate((("Folds", self.folds, ["3", "5", "10"]),
                                               ("Min. good droplets per whisky", self.min_per_class,
                                                ["3", "5", "10", "20"]))):
            ctk.CTkLabel(nums, text=text, font=F.body(12), text_color=C["muted"]).grid(
                row=0, column=2 * i, sticky="w", padx=(0 if i == 0 else 16, 6))
            ctk.CTkOptionMenu(nums, values=vals, width=64, height=28, font=F.body(12),
                              fg_color=C["bg"], button_color=C["line"], button_hover_color=C["muted"],
                              text_color=C["text"], dropdown_font=F.body(12),
                              command=lambda v, var=var: var.set(int(v)),
                              variable=tk.StringVar(value=str(var.get()))).grid(row=0, column=2 * i + 1)
        b1 = ctk.CTkFrame(opts, fg_color="transparent")
        b1.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 4))
        self._button(b1, "Run cross-validation", self.run_cv, primary=True).pack(side="left")
        self._button(b1, "Run control", self.run_control).pack(side="left", padx=8)
        b2 = ctk.CTkFrame(opts, fg_color="transparent")
        b2.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 16))
        self._button(b2, "Compare feature sets", self.compare_sets).pack(side="left")
        self._button(b2, "Train & save model…", self.train_save).pack(side="left", padx=8)

        right = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        res = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        res.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        res.grid_columnconfigure(4, weight=1)
        self.cv_tiles = {}
        for i, (key, title) in enumerate((("bal", "Balanced accuracy"), ("s3", "3 droplets averaged"),
                                          ("s5", "5 droplets averaged"), ("chance", "Chance"))):
            t = ctk.CTkFrame(res, fg_color=C["bg"], corner_radius=20)
            t.grid(row=0, column=i, padx=(16 if i == 0 else 6, 6), pady=16, sticky="w")
            ctk.CTkLabel(t, text=title, font=F.body(12), text_color=C["muted"]).pack(anchor="w", padx=16,
                                                                                    pady=(10, 0))
            v = ctk.CTkLabel(t, text="–", font=F.head(28), text_color=C["accent"] if i == 0 else C["text"])
            v.pack(anchor="w", padx=16, pady=(0, 10))
            self.cv_tiles[key] = v
        self.cv_text = ctk.CTkLabel(res, text="Load the dataset, then run cross-validation.", font=F.body(13),
                                    text_color=C["soft"], anchor="w", justify="left", wraplength=520)
        self.cv_text.grid(row=0, column=4, sticky="ew", padx=(10, 20))
        ctk.CTkLabel(res, text="Within-slip scores: every droplet of a whisky is on one cover slip, so these "
                               "mean \"recognises another droplet from the same slip\". They are not a "
                               "guarantee for a new slip.",
                     font=F.body(12), text_color=C["accent_deep"], anchor="w", justify="left",
                     wraplength=1100).grid(row=1, column=0, columnspan=5, sticky="ew", padx=20, pady=(0, 14))

        self.cv_well, self.cv_cv, self.cv_title, self.cv_note = self._well(right, "Confusion matrix")
        self.cv_well.configure(fg_color="#ffffff")
        self.cv_cv.configure(background="#ffffff")
        self.cv_title.configure(text_color=C["text"])
        self.cv_well.grid(row=1, column=0, sticky="nsew")
        self.cv_cv.bind("<Configure>", lambda e: self._draw_cv_figure())
        self.app._link(self.cv_well.top, "Open figure", self.open_cv_figure, 13).grid(row=0, column=1)
        self.app._link(self.cv_well.top, "Compare runs", self.compare_runs, 13).grid(row=0, column=2)
        self.app._link(self.cv_well.top, "Training curves", self.training_curves, 13).grid(row=0, column=3)

    def _cb(self, label):
        ev = self.app.events

        def progress(d, t):
            ev.put(("ml_progress", d / max(1, t)))
            if d % 25 == 0 or d == t:
                ev.put(("ml_status", f"{label}: {d} / {t}"))
        return dict(log=self._bg_log, progress=progress, should_stop=self.stop_flag.is_set)

    def _bg_log(self, msg):
        self.app.events.put(("ml_log", msg))

    def load_dataset(self, then=None):
        folder = self.folder()
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo(self.app.title(), "Choose the reference folder on the Identify tab first.")
            return
        problem = idc.opencv_problem()
        if problem:
            messagebox.showwarning(self.app.title(), problem)
            return

        def done(ds):
            self.dataset = ds
            self._fill_qc()
            n_ok = int(ds.qc_ok.sum())
            text = f"{len(ds.records)} droplets from {len(ds.qc_table())} whiskies · {n_ok} passed QC"
            if ds.skipped:
                text += f" · {len(ds.skipped)} couldn't be read (see log)"
            self.ds_info.configure(text=text, text_color=self.C["soft"])
            self.log(text)
            if then:
                then()
        self._run("Loading dataset…", lambda: mc.build_dataset(folder, **self._cb("Analysing images")), done)

    def _fill_qc(self):
        self.qc_tree.delete(*self.qc_tree.get_children())
        if self.dataset is None:
            return
        recall = self.cv.recall if self.cv is not None else {}
        for label, n, ok in self.dataset.qc_table():
            r = recall.get(label)
            tag = "bad" if ok < self.min_per_class.get() else ("warn" if ok < 0.7 * n else "ok")
            self.qc_tree.insert("", "end", values=(self.whisky(label), n, ok,
                                                   "" if r is None else f"{r:.2f}"), tags=(tag,))

    def _need_dataset(self, then):
        if self.dataset is None:
            self.load_dataset(then)
            return True
        return False

    def _embeddings(self, feature_set):
        """Attach CNN embeddings to the dataset first if this feature set needs them."""
        if "cnn" in feature_set:
            n = mc.add_embeddings(self.dataset.records, **self._cb("CNN embeddings"))
            if n:
                self._bg_log(f"  {n} CNN embeddings computed and cached")

    def _cv_job(self, feature_set, normalised):
        ds, folds, mpc = self.dataset, int(self.folds.get()), int(self.min_per_class.get())
        self._embeddings(feature_set)
        res = mc.cross_validate(ds, feature_set=feature_set, normalised=normalised, folds=folds,
                                min_per_class=mpc, **self._cb("Cross-validation"))
        fig = None
        if mc.figures_available():
            fig = mc.figure_confusion(res, self.names)
        return res, fig

    def _cv_done(self, out):
        res, fig = out
        self.cv, self.cv_figure = res, fig
        self.cv_runs.append(res)
        self.log(res.summary())
        self._show_cv()

    def run_cv(self, feature_set=None):
        if self._need_dataset(lambda: self.run_cv(feature_set)):
            return
        fs = feature_set or self.feature_set.get()
        norm = bool(self.normalised.get())
        self._run(f"Cross-validating {fs}…", lambda: self._cv_job(fs, norm), self._cv_done)

    def run_control(self):
        self.run_cv("nuisance")

    def compare_sets(self):
        if self._need_dataset(self.compare_sets):
            return
        if not mc.figures_available():
            messagebox.showinfo(self.app.title(), "Figures need matplotlib.\n\nRun:  pip install matplotlib")
            return
        norm = bool(self.normalised.get())

        def work():
            runs = []
            sets = list(SETS)
            # CNN sets only once their embeddings exist (the first run is long; start it on purpose)
            if not mc.cnn_available() and any(r.cnn is not None for r in self.dataset.records):
                sets += list(mc.CNN_SETS)
            sets.append("nuisance")
            for i, fs in enumerate(sets):
                self.app.events.put(("ml_status", f"Cross-validating {fs} ({i + 1} / {len(sets)})…"))
                runs.append(self._cv_job(fs, norm))
            return runs, mc.figure_cv_comparison([r for r, _ in runs])

        def done(out):
            runs, fig = out
            for r in runs:
                self._cv_done(r)
            FigureWindow(self.app, self.C, "Feature sets compared", fig, "feature_sets.png")
        self._run("Comparing feature sets…", work, done)

    def compare_runs(self):
        if not self.cv_runs:
            messagebox.showinfo(self.app.title(), "Run cross-validation first.")
            return
        runs = list(self.cv_runs[-12:])
        self._figure("Cross-validation runs compared", lambda: mc.figure_cv_comparison(runs), "cv_runs.png")

    def install_torch(self):
        """pip-install the CPU build of PyTorch into the Python running this app (so it
        lands in the right place even when the PC has several Pythons)."""
        import subprocess
        import sys
        py = sys.executable
        if os.path.basename(py).lower() == "pythonw.exe":           # pip wants the console one
            cand = os.path.join(os.path.dirname(py), "python.exe")
            py = cand if os.path.exists(cand) else py
        if not messagebox.askyesno(
                self.app.title(),
                "Install PyTorch for the CNN feature sets?\n\n"
                f"Downloads about 1 GB (CPU version) and installs it into:\n{py}\n\n"
                "Takes a few minutes. The rest of the app keeps working meanwhile."):
            return
        cmd = [py, "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX]
        if not (sys.prefix != sys.base_prefix or os.path.exists(os.path.join(sys.prefix, "conda-meta"))):
            cmd.insert(4, "--user")                    # plain installs (e.g. Program Files) need --user

        def work():
            self._bg_log("$ " + " ".join(cmd))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in p.stdout:
                if self.stop_flag.is_set():
                    p.kill()
                    raise InterruptedError("stopped")
                line = line.rstrip()
                if line and "[notice]" not in line:
                    self._bg_log("  " + line)
                    if line.startswith(("Collecting", "Downloading", "Installing", "Successfully")):
                        self.app.events.put(("ml_status", line[:110]))
            if p.wait() != 0:
                raise RuntimeError("pip failed; see Show log for the details")
            import importlib
            importlib.invalidate_caches()
            return mc.cnn_available()

        def done(problem):
            if problem:                                 # installed, but this process can't import it yet
                messagebox.showinfo(self.app.title(), "PyTorch is installed. Close and reopen the app "
                                                      "to use the CNN feature sets.")
                return
            self.cnn_seg.configure(state="normal")
            self.cnn_note.configure(text=CNN_NOTE)
            if self.cnn_install is not None:
                self.cnn_install.destroy()
                self.cnn_install = None
            self.log("PyTorch installed: the CNN feature sets are ready.")
            messagebox.showinfo(self.app.title(), "PyTorch is installed. The CNN feature sets are ready.")
        self._run("Installing PyTorch (about 1 GB)…", work, done)

    def training_curves(self):
        """PCA, loss and accuracy per step, and the features used, for one training run."""
        if self._need_dataset(self.training_curves):
            return
        fs, norm, mpc = self.feature_set.get(), bool(self.normalised.get()), int(self.min_per_class.get())
        ds, names = self.dataset, self.names

        def make():
            self._embeddings(fs)
            rep = mc.training_report(ds, feature_set=fs, normalised=norm, min_per_class=mpc,
                                     should_stop=self.stop_flag.is_set, log=self._bg_log)
            return mc.figure_training(rep, names)
        self._figure("Training curves", make, f"training_{fs.replace('+', '_')}.png")

    def _show_cv(self):
        r = self.cv
        if r is None:
            return
        pct = lambda v: "–" if v != v else f"{v:.2f}"  # noqa: E731  (nan check)
        self.cv_tiles["bal"].configure(text=pct(r.balanced_accuracy))
        self.cv_tiles["s3"].configure(text=pct(r.sample_accuracy(3)))
        self.cv_tiles["s5"].configure(text=pct(r.sample_accuracy(5)))
        self.cv_tiles["chance"].configure(text=pct(r.chance))
        text = (f"{r.feature_set}{' (normalised)' if r.normalised else ''} · {len(r.y_true)} droplets, "
                f"{len(r.classes)} whiskies, {r.folds}-fold · accuracy {r.accuracy:.2f}, "
                f"macro-F1 {r.macro_f1:.2f}")
        if r.excluded:
            text += "\nLeft out: " + ", ".join(f"{n} {why}" for why, n in r.excluded.items())
        if r.feature_set == "nuisance":
            real = [x for x in self.cv_runs if x.feature_set != "nuisance"]
            if real:
                gap = real[-1].balanced_accuracy - r.balanced_accuracy
                text += (f"\nControl: {abs(gap):.2f} {'below' if gap >= 0 else 'ABOVE'} the last real run. "
                         + ("Close to the real features: the model may be reading imaging artefacts."
                            if gap < 0.15 else "Well below the real features: good."))
        self.cv_text.configure(text=text)
        self.cv_title.configure(text=f"Confusion matrix · {r.feature_set}{' (normalised)' if r.normalised else ''}")
        self._fill_qc()
        self._draw_cv_figure()

    def _draw_cv_figure(self):
        if self.page.get() != "Train & evaluate":
            return
        if self.cv_figure is None:
            msg = "The confusion matrix appears here after cross-validation." if mc.figures_available() else \
                "The confusion matrix needs matplotlib (pip install matplotlib)."
            self.cv_cv.delete("all")
            W, H = max(self.cv_cv.winfo_width(), 60), max(self.cv_cv.winfo_height(), 60)
            self.cv_cv.create_text(W / 2, H / 2, text=msg, fill=self.C["muted"], font=(self.F.body_family, 12))
            return
        self._show_image(self.cv_cv, self.cv_figure, "cv")

    def open_cv_figure(self):
        if self.cv_figure is not None:
            FigureWindow(self.app, self.C, "Confusion matrix", self.cv_figure,
                         f"confusion_{self.cv.feature_set}.png")

    def train_save(self):
        if self._need_dataset(self.train_save):
            return
        fs, norm = self.feature_set.get(), bool(self.normalised.get())
        start = os.path.dirname(self.model_path) if self.model_path else self.folder()
        f = filedialog.asksaveasfilename(title="Save model", defaultextension=".npz",
                                         initialdir=start or None,
                                         initialfile=f"whisky_{fs.replace('+', '_')}{'_norm' if norm else ''}.wwml.npz",
                                         filetypes=[("Whisky Webs model", "*.wwml.npz"), ("All files", "*.*")])
        if not f:
            return
        same = self.cv is not None and self.cv.feature_set == fs and self.cv.normalised == norm

        def work():
            self._embeddings(fs)
            cv = self.cv if same else None
            out = None
            if cv is None:              # the "unknown" check is calibrated from cross-validation
                out = self._cv_job(fs, norm)
                cv = out[0]
            m = mc.train_model(self.dataset, feature_set=fs, normalised=norm,
                               min_per_class=int(self.min_per_class.get()), cv_result=cv)
            mc.save_model(m, f)
            return m, out

        def done(res):
            m, out = res
            if out is not None:
                self._cv_done(out)
            self._set_model(m, f)
            self.log(f"Saved model to {f} ({len(m.classes)} whiskies)")
            messagebox.showinfo(self.app.title(), f"Model saved:\n{f}\n\nIt's loaded on the Predict page.")
        self._run("Training…", work, done)

    # --------------------------------------------------------------- Predict
    def _build_predict(self, page):
        C, F = self.C, self.F
        left = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0, width=460)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        left.grid_propagate(False)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        mcard = self._card(left, "Model", 1)
        mcard.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.app._link(mcard.head, "Load…", self.browse_model).grid(row=0, column=3)
        self.model_info = ctk.CTkLabel(mcard, text="No model loaded – load one, or train one on the "
                                                   "Train & evaluate page.", font=F.body(13),
                                       text_color=C["soft"], anchor="w", justify="left", wraplength=400)
        self.model_info.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 16))

        self.pr_list = ImageList(self, left, "Images", [
            ("image", "IMAGE", 120, "w", True), ("best", "BEST MATCH", 140, "w", True),
            ("p", "P", 46, "center", False), ("conf", "CONFIDENCE", 120, "w", False)])
        self.pr_list.grid(row=1, column=0, sticky="nsew")
        run = ctk.CTkFrame(left, fg_color=C["surface"], corner_radius=28)
        run.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        ctk.CTkCheckBox(run, text="These images are one sample (judge them together)", variable=self.as_sample,
                        font=F.body(13), text_color=C["text"], fg_color=C["accent"], hover_color=C["accent_hover"],
                        border_color=C["line"], checkbox_width=18, checkbox_height=18).pack(
            anchor="w", padx=20, pady=(14, 6))
        row = ctk.CTkFrame(run, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(4, 16))
        self._button(row, "Predict", self.predict, primary=True, width=140).pack(side="left")
        self._button(row, "Save results (CSV)", self.save_csv).pack(side="left", padx=8)

        right = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        v = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        v.grid(row=0, column=0, sticky="ew")
        v.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(v, text="Sample verdict", font=F.semi(15), text_color=C["text"], anchor="w").grid(
            row=0, column=0, sticky="ew", padx=22, pady=(16, 0))
        self.verdict = ctk.CTkLabel(v, text="–", font=F.head(34), text_color=C["accent"], anchor="w",
                                    justify="left", wraplength=1000)
        self.verdict.grid(row=1, column=0, sticky="ew", padx=22)
        self.verdict_sub = ctk.CTkLabel(v, text="Add images and press Predict.", font=F.body(13),
                                        text_color=C["soft"], anchor="w", justify="left", wraplength=1000)
        self.verdict_sub.grid(row=2, column=0, sticky="ew", padx=22, pady=(2, 16))

        tops = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        tops.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        tops.grid_columnconfigure(0, weight=1)
        tops.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(tops, text="Most likely whiskies", font=F.semi(15), text_color=C["text"], anchor="w").grid(
            row=0, column=0, sticky="ew", padx=22, pady=(16, 6))
        self.top_frame = ctk.CTkFrame(tops, fg_color="transparent")
        self.top_frame.grid(row=1, column=0, sticky="new", padx=22, pady=(0, 10))
        self.top_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(tops, text="A model always names one of the whiskies it was trained on; a close relative "
                                "(e.g. Glengoyne 15 vs a trained Glengoyne 10) can be called as its sibling. "
                                "The \"not like any trained whisky\" check is only reliable with 3 or more "
                                "droplets judged together.",
                     font=F.body(12), text_color=C["muted"], anchor="w", justify="left", wraplength=1000).grid(
            row=2, column=0, sticky="ew", padx=22, pady=(0, 16))

    def _set_model(self, m, path, quiet=False):
        self.model, self.model_path = m, path
        info = m.info or {}
        cv = info.get("cv") or {}
        lines = [f"{os.path.basename(path)}",
                 f"{len(m.classes)} whiskies · {m.feature_set}{' (normalised)' if m.normalised else ''} · "
                 f"trained {info.get('trained', '?').replace('T', ' ')}"]
        if cv:
            sa = cv.get("sample_accuracy", {})
            lines.append(f"Cross-validation: balanced accuracy {cv.get('balanced_accuracy', float('nan')):.2f}"
                         + (f" · 5 droplets {sa.get('5', sa.get(5, float('nan'))):.2f}" if sa else "")
                         + " (within-slip)")
        lines.append("\"Unknown whisky\" check: " + ("calibrated" if info.get("unknown_threshold") else
                                                     "off (trained without cross-validation)"))
        self.model_info.configure(text="\n".join(lines), text_color=self.C["soft"])
        if not quiet:
            self.app._save_settings()

    def browse_model(self):
        f = filedialog.askopenfilename(title="Load model", filetypes=[("Whisky Webs model", "*.wwml.npz *.npz"),
                                                                      ("All files", "*.*")],
                                       initialdir=os.path.dirname(self.model_path) or None)
        if not f:
            return
        try:
            self._set_model(mc.load_model(f), f)
            self.log(f"Loaded model {f}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(self.app.title(), f"Couldn't load the model:\n{e}")

    def predict(self):
        if self.model is None:
            messagebox.showinfo(self.app.title(), "Load a model first (or train one on Train & evaluate).")
            return
        paths = list(self.pr_list.paths)
        if not paths:
            messagebox.showinfo(self.app.title(), "Add the images to predict first.")
            return
        problem = idc.opencv_problem()
        if problem:
            messagebox.showwarning(self.app.title(), problem)
            return
        model, together = self.model, bool(self.as_sample.get()) and len(paths) > 1
        self.preds, self.sample_pred = {}, None
        self.pr_list.refresh()

        def work():
            cb = self._cb("Reading droplets")
            if together:
                return mc.predict_sample(model, paths, **cb)
            return mc.predict(model, paths, **cb)

        def done(res):
            if together:
                self.sample_pred = res
                singles = res.droplets
            else:
                singles = res
            self.preds = {p.path: p for p in singles}
            self.pr_list.refresh()
            self._show_verdict()
        self._run(f"Predicting {len(paths)} image{'s' if len(paths) != 1 else ''}…", work, done)

    def _show_verdict(self):
        C, F = self.C, self.F
        for w in self.top_frame.winfo_children():
            w.destroy()
        sp = self.sample_pred
        singles = list(self.preds.values())
        if sp is not None:
            self.verdict.configure(text=self._verdict_text(sp),
                                   text_color=C["error"] if sp.unknown or not sp.n_used else C["accent"])
            sub = (f"{sp.n_used} of {sp.n_images} droplets used (the others failed QC) · "
                   f"{100 * sp.agreement:.0f}% of single droplets agree · mean probability {sp.probability:.2f}")
            if sp.unknown_threshold is not None:
                sub += f" · unknown score {sp.unknown_score:.2f} (threshold {sp.unknown_threshold:.2f})"
            else:
                sub += " · unknown check off for this model"
            if sp.n_used and sp.n_used < 3:
                sub += "\nOnly 1–2 droplets: add more of this sample for a reliable verdict."
            self.verdict_sub.configure(text=sub)
            tops = sp.top
        elif len(singles) == 1 and singles[0].top:
            p = singles[0]
            self.verdict.configure(text=self.whisky(p.label), text_color=C["accent"])
            self.verdict_sub.configure(text=f"One droplet · probability {p.probability:.2f} · confidence: "
                                            f"{p.confidence}. One droplet is weak evidence – add more "
                                            "droplets of the sample and judge them together.")
            tops = p.top
        else:
            good = [p for p in singles if p.top]
            self.verdict.configure(text=f"{len(good)} image{'s' if len(good) != 1 else ''} predicted",
                                   text_color=C["accent"])
            self.verdict_sub.configure(text="Each image was judged on its own (see the list). Tick \"These "
                                            "images are one sample\" to judge them together.")
            tops = []
        for i, (label, prob) in enumerate(tops):
            ctk.CTkLabel(self.top_frame, text=self.whisky(label), font=F.semi(14) if i == 0 else F.body(14),
                         text_color=C["text"], anchor="w").grid(row=i, column=0, sticky="w", pady=3)
            bar = ctk.CTkProgressBar(self.top_frame, height=12, corner_radius=6, fg_color=C["track"],
                                     progress_color=C["accent"] if i == 0 else C["sage"])
            bar.set(max(0.0, min(1.0, prob)))
            bar.grid(row=i, column=1, sticky="ew", padx=14)
            ctk.CTkLabel(self.top_frame, text=f"{prob:.2f}", font=F.body(13), text_color=C["soft"],
                         width=44).grid(row=i, column=2)

    def _verdict_text(self, sp):
        if not sp.n_used:
            return "No usable droplets (all failed QC)"
        if sp.unknown:
            return f"Not like any trained whisky (closest: {self.whisky(sp.label)})"
        if sp.margin < 0.1 and len(sp.top) > 1:
            return f"{self.whisky(sp.label)}, but close to {self.whisky(sp.top[1][0])}"
        return self.whisky(sp.label)

    def save_csv(self):
        if not self.preds:
            messagebox.showinfo(self.app.title(), "Nothing predicted yet – press Predict first.")
            return
        first = self.pr_list.paths[0] if self.pr_list.paths else ""
        f = filedialog.asksaveasfilename(title="Save results", defaultextension=".csv",
                                         initialdir=os.path.dirname(first) or None,
                                         initialfile="ml_predictions.csv", filetypes=[("CSV", "*.csv")])
        if not f:
            return
        name = lambda l: self.identify.names.get(l, "")  # noqa: E731
        try:
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["image", "path", "predicted", "whisky name", "probability", "confidence",
                            "second", "second probability", "third", "third probability", "qc ok", "qc reason",
                            "possibly unknown"])
                for p in (self.preds[q] for q in self.pr_list.paths if q in self.preds):
                    top = list(p.top) + [("", "")] * 3
                    w.writerow([os.path.basename(p.path), p.path, p.label, name(p.label),
                                f"{p.probability:.4f}" if p.top else "", p.confidence if p.top else "",
                                top[1][0], f"{top[1][1]:.4f}" if top[1][0] else "",
                                top[2][0], f"{top[2][1]:.4f}" if top[2][0] else "",
                                p.qc_ok, p.qc_reason, p.unknown])
                sp = self.sample_pred
                if sp is not None:
                    w.writerow([])
                    w.writerow(["SAMPLE VERDICT", self._verdict_text(sp), sp.label, name(sp.label),
                                f"{sp.probability:.4f}", f"{sp.n_used} of {sp.n_images} droplets used",
                                f"agreement {sp.agreement:.2f}",
                                f"unknown score {sp.unknown_score:.3f}",
                                "" if sp.unknown_threshold is None else f"threshold {sp.unknown_threshold:.3f}"])
                w.writerow([])
                w.writerow(["model", self.model_path, f"{self.model.feature_set}"
                            f"{' (normalised)' if self.model.normalised else ''}", mc.VERSION])
        except OSError as e:
            messagebox.showerror(self.app.title(), f"Couldn't save:\n{e}")
            return
        self.log(f"Saved results to {f}")

    # ================================================================ running
    def _run(self, label, work, done, quiet=False):
        """work() in the background thread; done(result) back on the Tk thread."""
        if self.busy():
            messagebox.showinfo(self.app.title(), "Wait for the current job to finish (or press Stop).")
            return False
        self.app._save_settings()
        self.stop_flag.clear()
        self.progress.set(0)
        self.progress.configure(progress_color=self.C["accent"])
        self.stop_btn.configure(state="normal")
        if not quiet:
            self.log("")
        self.status.configure(text=label)
        ev = self.app.events

        def run():
            try:
                ev.put(("ml_call", (done, work())))
                ev.put(("ml_status", "Done."))
            except InterruptedError:
                ev.put(("ml_log", "Stopped."))
            except Exception as e:  # noqa: BLE001
                ev.put(("ml_log", traceback.format_exc().rstrip()))
                ev.put(("ml_error", (label, str(e))))
            ev.put(("ml_finished", None))
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()
        return True

    def stop(self):
        self.stop_flag.set()
        self.log("Stopping…")

    def handle(self, kind, data):
        """Events from the background thread (via the app's queue)."""
        if kind == "ml_log":
            self.log(data)
        elif kind == "ml_status":
            self.status.configure(text=data)
        elif kind == "ml_progress":
            self.progress.set(max(0.0, min(1.0, data)))
        elif kind == "ml_call":
            fn, result = data
            fn(result)
        elif kind == "ml_error":
            label, err = data
            self.status.configure(text=f"Error: {err}")
            if self.analysing:
                self.failed[self.analysing] = err
                self.ex_list.update_row(self.analysing)
                self.analysing = None
                self._draw_explore()
            else:
                messagebox.showerror(self.app.title(), f"{label.rstrip('…')} failed:\n{err}")
        elif kind == "ml_finished":
            self.stop_btn.configure(state="disabled")
            self.progress.configure(progress_color=self.C["track"])
            if self.analysing:                  # stopped while analysing
                self.analysing = None
            self.app.after(150, self._analyse_pending)

    def _analyse_pending(self):
        """A row clicked while another job was running: analyse it now."""
        if self.page.get() == "Explore" and not self.busy():
            path, d = self._current_droplet()
            if path and d is None and path not in self.failed:
                self._analyse(path)
