"""
ml_explain_gui.py: the "Explain" page of the ML tab (what a trained model uses).

Two modes:
  Selected image   one droplet: evidence heat map (red = for the whisky, blue = against),
                   the feature groups for / against in colour, ranked group bars
                   (scalar / radial models: ml_core.explain), or the CNN map
                   ("where the CNN looks", 8 × 8 cells: ml_core.cnn_map)
  Whole sample     the same averaged over the droplets of one sample / whisky
                   (ml_core.explain_average / cnn_map_average), plus the radial
                   evidence profile (evidence in each r/R band, mean ± SD)

The page uses the ML tab's model, worker thread and run bar (tab._run).
"""
import os
import threading
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import numpy as np
from PIL import Image

import collections_core as cc
import identify_core as idc
import ml_core as mc

MODES = ["Selected image", "Whole sample"]
AUTO_SINGLE = "Predicted whisky"
AUTO_SAMPLE = "This sample's whisky"
AVERAGE = "Average of the others"
RUNNER_UP = "Runner-up"
MAX_N = ["10", "20", "40", "All"]
RESULT_CACHE = 40
FIG_ROWS = 6
GROUPS_SHOWN = 10

VIEWS = {
    ("single", "features"): ["Evidence", "For", "Against", "Image"],
    ("single", "cnn"): ["CNN map", "Crop"],
    ("sample", "features"): ["Evidence", "For", "Against", "Spread", "Average droplet"],
    ("sample", "cnn"): ["Evidence", "Spread", "Average droplet"],
    ("single", "both"): ["Evidence", "For", "Against", "CNN map", "Image"],
    ("sample", "both"): ["Evidence", "For", "Against", "CNN map", "Spread", "Average droplet"],
}
SECONDS = {"features": 2, "cnn": 4, "both": 6}      # per droplet, for the time estimate
CNN_PART = "CNN part (all 1,280 numbers)"
NOTES = {
    "Evidence": "Red = pixels pushing the score towards {t}, blue = pushing it towards {v} "
                "(square-root scale, so faint wide rings still show).",
    "For": "The feature groups adding most to {t} (vs {v}), one colour each (see the bars below).",
    "Against": "The feature groups taking most away from {t} (vs {v}), one colour each.",
    "Image": "The droplet as the model sees it.",
    "CNN map": "Where the CNN evidence for {t} (vs {v}) sits: red = for, blue = against. The true "
               "resolution is 8 × 8 cells, so read it as \"where the CNN looks\", not \"what it sees\".",
    "Crop": "The 256 px droplet crop the CNN was given.",
    "Spread": "How much the evidence varies between droplets (SD over droplets): bright = "
              "inconsistent, dark = the same in every droplet.",
    "Average droplet": "Every droplet centred and scaled to the same rim, then averaged. Rotations "
                       "are random, so it shows the radial structure.",
}
MAGMA = np.array([(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
                  (245, 125, 21), (250, 193, 39), (252, 253, 191)], np.float32)


def _magma(x):
    """x in 0..1 → magma-like RGB (no matplotlib needed)."""
    t = np.clip(x, 0, 1) * (len(MAGMA) - 1)
    i = np.minimum(t.astype(int), len(MAGMA) - 2)
    f = (t - i)[..., None]
    return (MAGMA[i] * (1 - f) + MAGMA[i + 1] * f).astype(np.uint8)


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


class ExplainPage:
    def __init__(self, tab, page):
        self.tab, self.app, self.C, self.F = tab, tab.app, tab.C, tab.F
        s = self.app.settings
        self.mode = tk.StringVar(value=s.get("ml_explain_mode") if s.get("ml_explain_mode") in MODES
                                 else MODES[0])
        self.view = tk.StringVar(value="Evidence")
        self.target = tk.StringVar(value=AUTO_SINGLE)
        self.versus = tk.StringVar(value=AVERAGE)
        self.max_n = tk.StringVar(value=s.get("ml_explain_n") if s.get("ml_explain_n") in MAX_N else "20")
        self.collection = s.get("ml_explain_collection", "khalifa")
        self.coll_options = []                  # [(id, title, path)]
        self.sample_paths, self.sample_names = {}, {}
        self.label_of = {}                      # display name -> label (model classes)
        self.results = OrderedDict()            # (path, model id, target, versus) -> (kind, obj)
        self.averages = []                      # [(model id, AverageExplanation)] this session
        self.shown = None                       # ("single", kind, obj, path) / ("sample", kind, avg, label)
        self._img = None                        # PIL image on screen (for Save view)
        self._scan_gen = 0
        self._build(page)
        self._sync_mode()

    # ================================================================ layout
    def _build(self, page):
        C, F, app, tab = self.C, self.F, self.app, self.tab
        left = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0, width=460)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        left.grid_propagate(False)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        mcard = tab._card(left, "Model", 1)
        mcard.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        app._link(mcard.head, "Load…", self._load_model).grid(row=0, column=3)
        self.model_lab = ctk.CTkLabel(mcard, text="", font=F.body(13), text_color=C["soft"], anchor="w",
                                      justify="left", wraplength=400)
        self.model_lab.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))
        ctk.CTkSegmentedButton(mcard, values=MODES, variable=self.mode, command=lambda v: self._sync_mode(),
                               font=F.semi(13), height=32, corner_radius=16, fg_color=C["bg"],
                               selected_color=C["accent"], selected_hover_color=C["accent_hover"],
                               unselected_color=C["bg"], unselected_hover_color=C["hover_tint"],
                               text_color=C["text"]).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 16))

        holder = ctk.CTkFrame(left, fg_color=C["bg"], corner_radius=0)
        holder.grid(row=1, column=0, sticky="nsew")
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        # single image: a list of images
        from ml_gui import ImageList
        self.list = ImageList(tab, holder, "Images", [
            ("image", "IMAGE", 150, "w", True), ("whisky", "WHISKY", 130, "w", True),
            ("p", "EXPLAINED", 110, "w", False)], on_select=self._select_image)
        self.list.grid(row=0, column=0, sticky="nsew")
        # whole sample: a list of samples
        self.sample_card = card = tab._card(holder, "Samples")
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(2, weight=1)
        app._link(card.head, "Refresh", lambda: self._fill_samples(fresh=True)).grid(row=0, column=3)
        self.coll_menu = ctk.CTkOptionMenu(card, values=["Khalifa samples"], command=self._on_collection,
                                           font=F.body(13), dropdown_font=F.body(13), height=30,
                                           corner_radius=15, fg_color=C["bg"], button_color=C["bg"],
                                           button_hover_color=C["hover_tint"], text_color=C["text"],
                                           dropdown_fg_color=C["surface"], dropdown_text_color=C["text"],
                                           dropdown_hover_color=C["hover_tint"], width=260)
        self.coll_menu.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 6))
        th = tk.Frame(card, bg=C["surface"], highlightthickness=0)
        th.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        th.grid_rowconfigure(0, weight=1)
        th.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(th, columns=("sample", "images", "done"), show="headings",
                                 selectmode="browse", style="WW.Treeview")
        for col, text, w, anchor, stretch in (("sample", "SAMPLE", 260, "w", True),
                                              ("images", "IMAGES", 64, "center", False),
                                              ("done", "EXPLAINED", 84, "center", False)):
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=w, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(th, command=self.tree.yview, fg_color=C["surface"],
                              button_color=C["line"], button_hover_color=C["muted"])
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure("done", foreground=C["sage_ink"])
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._select_sample())
        self.tree.bind("<Double-1>", lambda e: self.run())
        nrow = ctk.CTkFrame(card, fg_color="transparent")
        nrow.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))
        ctk.CTkLabel(nrow, text="Droplets", font=F.body(13), text_color=C["text"]).pack(side="left")
        ctk.CTkSegmentedButton(nrow, values=MAX_N, variable=self.max_n, font=F.semi(12), height=28,
                               corner_radius=14, fg_color=C["bg"], selected_color=C["accent"],
                               selected_hover_color=C["accent_hover"], unselected_color=C["bg"],
                               unselected_hover_color=C["hover_tint"], text_color=C["text"]).pack(
            side="left", padx=10)
        self.n_note = ctk.CTkLabel(nrow, text="", font=F.body(12), text_color=C["muted"])
        self.n_note.pack(side="left")
        self.max_n.trace_add("write", lambda *a: self._time_note())

        run = ctk.CTkFrame(left, fg_color=C["surface"], corner_radius=28)
        run.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        run.grid_columnconfigure(1, weight=1)
        menu = dict(font=F.body(13), dropdown_font=F.body(13), height=30, corner_radius=15,
                    fg_color=C["bg"], button_color=C["bg"], button_hover_color=C["hover_tint"],
                    text_color=C["text"], dropdown_fg_color=C["surface"], dropdown_text_color=C["text"],
                    dropdown_hover_color=C["hover_tint"], dynamic_resizing=False, width=300)
        ctk.CTkLabel(run, text="Explain", font=F.semi(13), text_color=C["text"]).grid(
            row=0, column=0, sticky="w", padx=(20, 8), pady=(16, 4))
        self.target_menu = ctk.CTkOptionMenu(run, values=[AUTO_SINGLE], variable=self.target,
                                             command=lambda v: self._settings_changed(), **menu)
        self.target_menu.grid(row=0, column=1, sticky="w", pady=(16, 4))
        ctk.CTkLabel(run, text="versus", font=F.semi(13), text_color=C["text"]).grid(
            row=1, column=0, sticky="w", padx=(20, 8), pady=4)
        self.versus_menu = ctk.CTkOptionMenu(run, values=[AVERAGE, RUNNER_UP], variable=self.versus,
                                             command=lambda v: self._settings_changed(), **menu)
        self.versus_menu.grid(row=1, column=1, sticky="w", pady=4)
        row = ctk.CTkFrame(run, fg_color="transparent")
        row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(8, 16))
        self.go_btn = tab._button(row, "Explain", self.run, primary=True, width=130)
        self.go_btn.pack(side="left")
        self.fig_btn = tab._button(row, "Figure", self.figure)
        self.fig_btn.pack(side="left", padx=8)
        tab._button(row, "Save view", self.save_view).pack(side="left")
        if not mc.figures_available():
            self.fig_btn.configure(state="disabled")

        right = ctk.CTkFrame(page, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.well, self.cv, self.title, self.note = tab._well(right, "Explain")
        self.well.grid(row=0, column=0, sticky="nsew")
        self.cv.bind("<Configure>", lambda e: self.draw())
        self.well.bind("<Configure>", lambda e: self.note.configure(wraplength=max(300, e.width - 60)), add="+")
        self.view_seg = ctk.CTkSegmentedButton(self.well.top, values=VIEWS[("single", "features")],
                                               variable=self.view, command=lambda v: self.draw(),
                                               font=F.semi(12), height=30, corner_radius=15, fg_color=C["seg"],
                                               selected_color=C["accent"], selected_hover_color=C["accent_hover"],
                                               unselected_color=C["seg"], unselected_hover_color=C["soft"],
                                               text_color=C["bg"])
        self.view_seg.grid(row=0, column=1, sticky="e")

        info = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        info.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        info.grid_columnconfigure(0, weight=1, uniform="xp")
        info.grid_columnconfigure(1, weight=1, uniform="xp")
        self.head = ctk.CTkLabel(info, text="", font=F.semi(14), text_color=C["text"], anchor="w",
                                 justify="left", wraplength=1000)
        self.head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(14, 2))
        self.cap_l = ctk.CTkLabel(info, text="", font=F.semi(13), text_color=C["soft"], anchor="w")
        self.cap_l.grid(row=1, column=0, sticky="ew", padx=20)
        self.cap_r = ctk.CTkLabel(info, text="", font=F.semi(13), text_color=C["soft"], anchor="w")
        self.cap_r.grid(row=1, column=1, sticky="ew", padx=(0, 20))
        self.bars = tk.Canvas(info, height=210, background=C["surface"], highlightthickness=0)
        self.bars.grid(row=2, column=0, sticky="ew", padx=(16, 8), pady=(4, 16))
        self.bars.bind("<Configure>", lambda e: self._draw_info())
        self.side = tk.Frame(info, bg=C["surface"])
        self.side.grid(row=2, column=1, sticky="nsew", padx=(8, 16), pady=(4, 16))
        self.side.grid_rowconfigure(0, weight=1)
        self.side.grid_columnconfigure(0, weight=1)
        self.plot = tk.Canvas(self.side, height=210, background=C["surface"], highlightthickness=0)
        self.plot.grid(row=0, column=0, sticky="nsew")
        self.plot.bind("<Configure>", lambda e: self._draw_info())
        self.text = tab._text(self.side, 210)
        self.text.grid(row=0, column=0, sticky="nsew")

    # ================================================================ state
    def settings(self):
        return dict(ml_explain_mode=self.mode.get(), ml_explain_n=self.max_n.get(),
                    ml_explain_collection=self.collection)

    def single(self):
        return self.mode.get() == MODES[0]

    def model_kind(self, m=None):
        m = m or self.tab.model
        if m is None:
            return None
        if m.feature_set == "cnn":
            return "cnn"
        if "cnn" in m.feature_set:
            return "both"
        return None if m.feature_set == "nuisance" else "features"

    def model_changed(self):
        """Called by the tab when a model is loaded or trained."""
        m = self.tab.model
        if m is None:
            self.model_lab.configure(text="No model loaded: load one, or train one on Train & evaluate.")
        else:
            kind = self.model_kind(m)
            what = {"features": "Feature evidence maps (exact for this model)",
                    "cnn": "CNN maps (where the CNN looks; 8 × 8 cells)",
                    "both": "Feature evidence maps + CNN map, and how the score splits between them",
                    None: "The control (nuisance) model can't be explained."}[kind]
            self.model_lab.configure(text=f"{os.path.basename(self.tab.model_path) or 'Just trained'} · "
                                          f"{len(m.classes)} whiskies · {m.feature_set}"
                                          f"{' (normalised)' if m.normalised else ''}\n{what}")
        names = {}
        for lab in (m.classes if m else []):
            n = self.tab.whisky(lab) or lab
            names[n if n not in names else f"{n} ({lab})"] = lab
        self.label_of = names
        shown = sorted(names)
        auto = AUTO_SINGLE if self.single() else AUTO_SAMPLE
        self.target_menu.configure(values=[auto] + shown)
        self.versus_menu.configure(values=[AVERAGE, RUNNER_UP] + shown)
        if self.target.get() not in [auto] + shown:
            self.target.set(auto)
        if self.versus.get() not in [AVERAGE, RUNNER_UP] + shown:
            self.versus.set(AVERAGE)
        self.list.refresh()
        self._mark_samples()
        self._time_note()
        self.draw()

    def _load_model(self):
        self.tab.browse_model()

    def _sync_mode(self):
        single = self.single()
        if single:
            self.sample_card.grid_remove()
            self.list.card.grid()
        else:
            self.list.card.grid_remove()
            self.sample_card.grid()
            if not self.sample_paths:
                self._fill_samples()
        auto_old, auto_new = (AUTO_SAMPLE, AUTO_SINGLE) if single else (AUTO_SINGLE, AUTO_SAMPLE)
        vals = [auto_new] + [v for v in self.target_menu.cget("values") if v not in (AUTO_SINGLE, AUTO_SAMPLE)]
        self.target_menu.configure(values=vals)
        if self.target.get() == auto_old:
            self.target.set(auto_new)
        self.go_btn.configure(text="Explain" if single else "Explain sample")
        self._select_image() if single else self._select_sample()

    def _settings_changed(self):
        if self.single():
            self._select_image()
        else:
            self._select_sample()

    def _choice(self):
        """(target label or None, versus label or None, runner_up?) from the menus."""
        t = self.target.get()
        v = self.versus.get()
        tgt = self.label_of.get(t)
        return tgt, self.label_of.get(v), v == RUNNER_UP

    def _key(self, path):
        t, v, ru = self._choice()
        return path, id(self.tab.model), t, "runner-up" if ru else v

    # ================================================================ single image
    def row_for(self, path):
        name = os.path.splitext(os.path.basename(path))[0].replace("_stitched", "")
        label = mc.label_of(path)
        hits = [r for k, r in self.results.items() if k[0] == path and k[1] == id(self.tab.model)]
        if not hits:
            return (name, self.tab.whisky(label), ""), "wait"
        kind, obj = hits[-1]
        if isinstance(obj, str):
            return (name, self.tab.whisky(label), "couldn't"), "bad"
        return (name, self.tab.whisky(label), f"{obj.target} {obj.probability:.2f}"), "ok"

    def add_images(self, paths):
        """From the Predict page: put these images in the list and switch to single-image mode."""
        self.mode.set(MODES[0])
        self.list.add(list(paths))
        self._sync_mode()

    def _select_image(self):
        if not self.single():
            return
        sel = self.list.selected()
        if sel:
            hit = self.results.get(self._key(sel[0]))
            self.shown = ("single",) + hit + (sel[0],) if hit else None
        else:
            self.shown = None
        self.draw()

    def _need_model(self):
        if self.tab.model is None:
            messagebox.showinfo(self.app.title(), "Load a model first (or train one on Train & evaluate).")
            return False
        if self.model_kind() is None:
            messagebox.showinfo(self.app.title(), "The control (nuisance) model can't be explained: "
                                                  "load a scalar, radial or CNN model.")
            return False
        if self.model_kind() in ("cnn", "both") and mc.cnn_available():
            messagebox.showinfo(self.app.title(), "CNN maps need PyTorch.\n\n" + mc.cnn_available())
            return False
        problem = idc.opencv_problem()
        if problem:
            messagebox.showwarning(self.app.title(), problem)
            return False
        t, v, ru = self._choice()
        if t and v and t == v:
            messagebox.showinfo(self.app.title(), "Pick a different whisky to compare against.")
            return False
        return True

    def _explain_one(self, model, path, t, v, ru):
        """(kind, Explanation | CNNMap) for one image (worker thread)."""
        kind = self.model_kind(model)
        ds = (model.info or {}).get("downsample", mc.DOWNSAMPLE)
        if kind == "cnn":
            obj = mc.cnn_map(model, path, target=t, versus=None if ru else v, downsample=ds)
            if ru:
                obj = mc.cnn_map(model, path, target=obj.target, downsample=ds,
                                 versus=next(l for l, _ in obj.top if l != obj.target))
            return kind, obj
        d = self.tab.droplets.get(path) if ds == mc.DOWNSAMPLE else None
        if d is None:
            d = mc.analyse(path, downsample=ds)
        obj = mc.explain(model, d, target=t, versus=None if ru else v)
        if ru:
            obj = mc.explain(model, d, target=obj.target, versus=next(l for l, _ in obj.top if l != obj.target))
        if kind == "both":                     # the CNN half, same target and comparison
            obj._cnn = mc.cnn_map(model, path, target=obj.target, downsample=ds,
                                  versus=None if obj.versus == "average whisky" else obj.versus)
        return kind, obj

    def _remember(self, key, res):
        self.results[key] = res
        self.results.move_to_end(key)
        while len(self.results) > RESULT_CACHE:
            self.results.popitem(last=False)

    def _run_single(self):
        sel = self.list.selected()
        if not sel:
            messagebox.showinfo(self.app.title(), "Add images on the left and click one first.")
            return
        path, model = sel[0], self.tab.model
        t, v, ru = self._choice()
        key = self._key(path)

        def work():
            return self._explain_one(model, path, t, v, ru)

        def done(res):
            self._remember(key, res)
            self.list.update_row(path)
            self._select_image()
        self.tab._run(f"Explaining {os.path.basename(path)}…", work, done)

    # ================================================================ whole sample
    def _on_collection(self, title):
        cid = next((c for c, t, _ in self.coll_options if t == title), "khalifa")
        if cid != self.collection:
            self.collection = cid
            self._fill_samples()

    def _fill_samples(self, fresh=False):
        """Scan the chosen collection in a thread (SWRI scans take a few seconds)."""
        folder = self.tab.folder()
        self.tree.delete(*self.tree.get_children())
        self.sample_paths = {}
        if not folder or not os.path.isdir(folder):
            self.tree.insert("", "end", iid="_none", values=("Choose the reference folder on Identify", "", ""))
            return
        self._scan_gen += 1
        gen, cid = self._scan_gen, self.collection
        self.tree.insert("", "end", iid="_none", values=("Reading the folder…", "", ""))
        ev = self.app.events

        def scan():
            try:
                opts = cc.available_collections(folder)
            except Exception:  # noqa: BLE001
                opts = []
            if not any(c == cid for c, _, _ in opts):
                use = "khalifa"
            else:
                use = cid
            try:
                if use == "khalifa" and not any(c == "khalifa" for c, _, _ in opts):
                    classes, _ = idc.scan_classes(folder, fresh=fresh)      # folder isn't under "4. Whisky Webs"
                    names = {}
                else:
                    classes, _ = cc.scan_collection(use, folder, fresh=fresh)
                    names = cc.collection_names(use, folder) if use != "khalifa" else {}
                res = (opts, use, classes, names, None)
            except Exception as e:  # noqa: BLE001
                res = (opts, use, [], {}, str(e))
            ev.put(("ml_call", (lambda r: self._samples_ready(gen, r), res)))
        threading.Thread(target=scan, daemon=True).start()

    def _samples_ready(self, gen, res):
        if gen != self._scan_gen:
            return
        opts, cid, classes, names, err = res
        self.coll_options, self.collection = opts, cid
        if opts:
            self.coll_menu.configure(values=[t for _, t, _ in opts])
            self.coll_menu.set(next((t for c, t, _ in opts if c == cid), opts[0][1]))
        else:
            self.coll_menu.configure(values=["Reference folder"])
            self.coll_menu.set("Reference folder")
        self.tree.delete(*self.tree.get_children())
        self.sample_paths, self.sample_names = {}, {}
        if err:
            self.tree.insert("", "end", iid="_none", values=(f"Couldn't read it: {err}", "", ""))
            return
        for label, paths in classes:
            name = cc.label_name(label, names) if names else self.tab.whisky(label)
            self.sample_paths[label] = paths
            self.sample_names[label] = name
            self.tree.insert("", "end", iid=label, values=(name, len(paths), ""))
        self._mark_samples()
        if self.shown and self.shown[0] == "sample" and self.tree.exists(self.shown[3]):
            self.tree.selection_set(self.shown[3])
            self.tree.see(self.shown[3])

    def _avg_for(self, label):
        t, v, ru = self._choice()
        mid = id(self.tab.model)
        for m, key, a in reversed(self.averages):
            if m == mid and key == (label, t or label, "runner-up" if ru else v, self.max_n.get()):
                return a
        return None

    def _mark_samples(self):
        mid = id(self.tab.model)
        done = {k[0] for m, k, _ in self.averages if m == mid}
        for label in self.sample_paths:
            if self.tree.exists(label):
                self.tree.set(label, "done", "✓" if label in done else "")
                self.tree.item(label, tags=("done",) if label in done else ())

    def _select_sample(self):
        if self.single():
            return
        sel = [s for s in self.tree.selection() if s in self.sample_paths]
        if sel:
            a = self._avg_for(sel[0])
            kind = ("both" if hasattr(a, "_cnn") else a.kind) if a else None
            self.shown = ("sample", kind, a, sel[0]) if a else None
        else:
            self.shown = None
        self._time_note()
        self.draw()

    def _time_note(self):
        sel = [s for s in self.tree.selection() if s in self.sample_paths]
        n = len(self.sample_paths[sel[0]]) if sel else None
        cap = None if self.max_n.get() == "All" else int(self.max_n.get())
        k = n if cap is None else (cap if n is None else min(n, cap))
        if k is None:
            self.n_note.configure(text="")
            return
        per = SECONDS.get(self.model_kind(), 2)
        mins = k * per / 60
        self.n_note.configure(text=f"{k} droplets · about {max(1, round(mins))} min" if mins >= 1
                              else f"{k} droplets · under a minute")

    def _run_sample(self):
        sel = [s for s in self.tree.selection() if s in self.sample_paths]
        if not sel:
            messagebox.showinfo(self.app.title(), "Click a sample on the left first.")
            return
        label, model = sel[0], self.tab.model
        t, v, ru = self._choice()
        if t is None:
            if label not in model.classes:
                messagebox.showinfo(self.app.title(),
                                    f"The model wasn't trained on {self.sample_names.get(label, label)}.\n\n"
                                    "Choose which whisky to explain in the \"Explain\" menu (e.g. the one it "
                                    "gets called as).")
                return
            t = label
        if v == t and not ru:
            messagebox.showinfo(self.app.title(), "Pick a different whisky to compare against.")
            return
        paths = self.sample_paths[label]
        max_n = None if self.max_n.get() == "All" else int(self.max_n.get())
        key = (label, t, "runner-up" if ru else v, self.max_n.get())
        kind = self.model_kind(model)
        ds = (model.info or {}).get("downsample", mc.DOWNSAMPLE)
        cb = self.tab._cb("Explaining droplets")

        def work():
            vs = v
            if ru:
                use = mc._pick(list(paths), max_n)
                self.app.events.put(("ml_status", "Finding the runner-up (predicting the sample)…"))
                sp = mc.predict_sample(model, use, top=len(model.classes), log=cb["log"],
                                       should_stop=cb["should_stop"], downsample=ds)
                vs = next((l for l, _ in sp.top if l != t), None)
            fn = mc.cnn_map_average if kind == "cnn" else mc.explain_average
            a = fn(model, paths, t, versus=vs, max_n=max_n, downsample=ds, **cb)
            if kind == "both":                 # the CNN half over the same droplets
                self.app.events.put(("ml_status", "Now the CNN map of the same droplets…"))
                a._cnn = mc.cnn_map_average(model, paths, t, versus=vs, max_n=max_n, downsample=ds, **cb)
            return a

        def done(a):
            self.averages.append((id(model), key, a))
            self.averages = self.averages[-30:]
            self._mark_samples()
            self._select_sample()
            if a.skipped:
                self.tab.log(f"{len(a.skipped)} droplet(s) skipped: "
                             + "; ".join(f"{n} ({r})" for n, r in a.skipped[:6])
                             + (" …" if len(a.skipped) > 6 else ""))
        self.tab._run(f"Explaining {self.sample_names.get(label, label)}…", work, done)

    # ================================================================ actions
    def run(self):
        if not self._need_model():
            return
        self._run_single() if self.single() else self._run_sample()

    def figure(self):
        if not mc.figures_available():
            return
        if not self._need_model():
            return
        names = self.tab.names
        if self.single():
            if self.model_kind() == "cnn":
                messagebox.showinfo(self.app.title(), "There's no figure for single CNN maps yet: use "
                                                      "\"Save view\", or the Whole sample figure.")
                return
            sel = self.list.selected()[:FIG_ROWS]
            if not sel:
                messagebox.showinfo(self.app.title(), "Select one or more images (Ctrl/Shift-click, "
                                                      f"up to {FIG_ROWS}).")
                return
            model = self.tab.model
            t, v, ru = self._choice()
            keys = {p: self._key(p) for p in sel}
            have = {p: self.results.get(k) for p, k in keys.items()}

            def make():
                es = []
                for i, p in enumerate(sel, 1):
                    r = have[p]
                    if r is None or isinstance(r[1], str):
                        self.app.events.put(("ml_status", f"Explaining {i} / {len(sel)}…"))
                        r = self._explain_one(model, p, t, v, ru)
                        have[p] = r
                    es.append(r[1])
                return mc.figure_explain(es, names)

            def show(im):
                for p, r in have.items():
                    self._remember(keys[p], r)
                    self.list.update_row(p)
                from ml_gui import FigureWindow
                FigureWindow(self.app, self.C, "Explanations of the selected images", im, "explain.png")
                self._select_image()
            self.tab._run("Making the explanation figure…", make, show)
            return
        mid = id(self.tab.model)
        avgs = [b for m, _, a in self.averages if m == mid for b in (a, getattr(a, "_cnn", None)) if b]
        seen, rows = set(), []
        for a in reversed(avgs):                    # newest per sample
            if (a.target, a.kind) not in seen:
                seen.add((a.target, a.kind))
                rows.append(a)
        rows = rows[:FIG_ROWS][::-1]
        if not rows:
            messagebox.showinfo(self.app.title(), "Explain one or more samples first. The figure has one row "
                                                  f"per sample explained with this model (up to {FIG_ROWS}).")
            return
        self.tab._figure("Sample explanations", lambda: mc.figure_explain_average(rows, names),
                         "explain_samples.png")

    def save_view(self):
        if self._img is None:
            return
        f = filedialog.asksaveasfilename(title="Save view", defaultextension=".png",
                                         initialfile=f"explain_{self.view.get().lower().replace(' ', '_')}.png",
                                         filetypes=[("PNG", "*.png")])
        if f:
            try:
                self._img.save(f)
                self.tab.log(f"Saved view to {f}")
            except OSError as e:
                messagebox.showerror(self.app.title(), f"Couldn't save:\n{e}")

    # ================================================================ drawing
    def _name(self, label):
        if label in ("average whisky", None, ""):
            return "the average of the others"
        return self.sample_names.get(label) or self.tab.whisky(label) or label

    def draw(self):
        if self.tab.page.get() != "Explain":
            return
        sh = self.shown
        if sh is None:
            self._img = None
            self.view_seg.configure(values=["Evidence"] if not self.single() else VIEWS[("single", "features")])
            self.title.configure(text="Explain")
            self.note.configure(text="")
            self.head.configure(text="")
            self.cap_l.configure(text="")
            self.cap_r.configure(text="")
            self._draw_info()
            if self.tab.model is None:
                msg = "Load a model on the left (or train one on Train & evaluate)"
            elif self.single():
                msg = ("Add images on the left (files, a folder, the Gallery selection, or \"Explain…\" on "
                       "Predict), click one and press Explain")
            else:
                msg = ("Pick a sample on the left and press \"Explain sample\": the model's evidence is "
                       "averaged over its droplets")
            return self.tab._show_image(self.cv, None, "xp", msg)
        mode, kind, obj, ref = sh
        views = VIEWS[(mode, kind)]
        self.view_seg.configure(values=views)
        if self.view.get() not in views:
            self.view.set(views[0])
        v = self.view.get()
        if isinstance(obj, str):
            return self.tab._show_image(self.cv, None, "xp", obj)
        if mode == "single":
            img = self._single_image(kind, obj, v)
            title = os.path.splitext(os.path.basename(ref))[0].replace("_stitched", "")
        else:
            img = self._average_image(kind, obj, v)
            title = f"{self.sample_names.get(ref, ref)} · {obj.n} droplets"
        self._img = Image.fromarray(img)
        self.title.configure(text=title)
        note = NOTES[v].format(t=self._name(obj.target), v=self._name(obj.versus))
        if kind == "both" and v in ("Evidence", "For", "Against", "Spread"):
            note += " Named features only: the CNN half is under \"CNN map\"."
        self.note.configure(text=note)
        self.tab._show_image(self.cv, self._img, "xp")
        self._draw_info()

    def _single_image(self, kind, obj, v):
        if kind == "cnn":
            return mc.view_cnn_map(obj) if v == "CNN map" else obj.crop
        if v == "CNN map":
            return mc.view_cnn_map(obj._cnn)
        if v == "Evidence":
            return mc.view_explain_heat(obj)
        if v in ("For", "Against"):
            img, self._legend = mc.view_explain_groups(obj, k=6, sign=+1 if v == "For" else -1)
            return img
        return obj.image

    def _average_image(self, kind, a, v):
        if v == "CNN map":
            return mc.view_average_heat(a._cnn)
        if v == "Evidence":
            return mc.view_average_heat(a)
        if v in ("For", "Against"):
            img, self._legend = mc.view_average_groups(a, k=6, sign=+1 if v == "For" else -1)
            return img
        if v == "Spread":
            return _magma(a.heat_sd / (np.percentile(a.heat_sd, 99.5) + 1e-12))
        return a.mean_image

    def _draw_info(self):
        sh = self.shown
        for cv in (self.bars, self.plot):
            cv.delete("all")
        if sh is None or isinstance(sh[2], str):
            self.text.grid_remove()
            self.plot.grid()
            return
        mode, kind, obj, ref = sh
        T, V = self._name(obj.target), self._name(obj.versus)
        v = self.view.get()
        legend = {g: _hex(c) for g, _, c in getattr(self, "_legend", [])} if v in ("For", "Against") else {}
        if mode == "single":
            tops = " · ".join(f"{self.tab.whisky(l) or l} {p:.2f}" for l, p in obj.top)
            split = (f"\nScore split: named features {obj.total - obj.cnn_total:+.2f} · CNN part "
                     f"{obj.cnn_total:+.2f} (logit)" if kind == "both" else "")
            self.head.configure(text=f"Why {T} rather than {V}?   p({obj.target}) = {obj.probability:.2f} · "
                                     f"top matches: {tops}{split}")
            if kind in ("features", "both"):
                self.cap_l.configure(text=f"Feature groups (logit; total {obj.total:+.2f})")
                self.cap_r.configure(text="Single features, largest first")
                cnn = [(CNN_PART, obj.cnn_total, None, "#8a8a8a")] if kind == "both" else []
                self._bar_chart(cnn + [(g, c, None, legend.get(g))
                                       for g, c in obj.groups[:GROUPS_SHOWN - len(cnn)]])
                if kind == "both" and v == "CNN map":
                    self.cap_r.configure(text="CNN evidence in each r/R band of the crop")
                    self.text.grid_remove()
                    self.plot.grid()
                    self._band_plot(*self._cnn_bands(obj._cnn.cells), None)
                    return
                w = max([len(n) for n, _ in obj.features[:40]] + [8])
                self.plot.grid_remove()
                self.text.grid()
                self.tab._set_text(self.text, "\n".join(f"{n:<{w}}  {c:+8.3f}" for n, c in obj.features[:40]))
            else:
                self.cap_l.configure(text="Top matches (probability)")
                self.cap_r.configure(text="CNN evidence in each r/R band of the crop")
                self._bar_chart([(self.tab.whisky(l) or l, p, None, None) for l, p in obj.top], signed=False)
                self.text.grid_remove()
                self.plot.grid()
                r, prof = self._cnn_bands(obj.cells)
                self._band_plot(r, prof, None)
            return
        a = obj
        sk = f" · {len(a.skipped)} skipped (see the log)" if a.skipped else ""
        split = (f"\nScore split (mean ± SD): named features {a.features_part:+.2f} ± {a.features_part_sd:.2f}"
                 f" · CNN part {a.cnn_part:+.2f} ± {a.cnn_part_sd:.2f} (logit)" if kind == "both" else "")
        self.head.configure(text=f"{T} vs {V}: mean p({a.target}) = {a.probability:.2f} · "
                                 f"{a.accuracy:.0%} of {a.n} droplets predicted as {T}{sk}{split}")
        self.text.grid_remove()
        self.plot.grid()
        self.cap_r.configure(text="Evidence in each r/R band (sums to the score; mean ± SD)")
        if a.kind == "features":
            self.cap_l.configure(text="Feature groups: mean contribution ± SD (logit)")
            cnn = [(CNN_PART, a.cnn_part, a.cnn_part_sd, "#8a8a8a")] if kind == "both" else []
            self._bar_chart(cnn + [(g, c, s, legend.get(g)) for g, c, s in a.groups[:GROUPS_SHOWN - len(cnn)]])
        else:
            self.cap_l.configure(text="")
            self.bars.create_text(10, 20, anchor="nw", width=max(200, self.bars.winfo_width() - 20),
                                  fill=self.C["soft"], font=(self.F.body_family, 11),
                                  text=f"CNN evidence for {T} vs {V}, averaged over {a.n} droplets (8 × 8 "
                                       "cells per droplet, exact). The CNN's features have no names, so "
                                       "there are no groups: use the band plot to see where it looks, and a "
                                       "scalar + radial model for the named view.")
        b = a._cnn if kind == "both" and v == "CNN map" else a
        if b is not a:
            self.cap_r.configure(text="CNN evidence in each r/R band (mean ± SD)")
        self._band_plot(b.radial_r, b.radial, b.radial_sd)

    @staticmethod
    def _cnn_bands(cells):
        """Radial evidence profile of one CNN map (cells → crop frame, rim at 1/1.2 of half-width)."""
        import cv2
        n = 64                                   # upsample like view_cnn_map, keeping the total
        up = cv2.resize(np.asarray(cells, np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        up *= cells.size / float(n * n)
        c = (np.arange(n) + 0.5) / n * 2 - 1
        rr = np.hypot(*np.meshgrid(c, c)) * 1.2
        edges = np.linspace(0, 1.5, 16)
        idx = np.clip(np.digitize(rr, edges) - 1, 0, len(edges) - 2)
        s = np.bincount(idx.ravel(), up.ravel(), len(edges) - 1)
        return (edges[:-1] + edges[1:]) / 2, s

    def _bar_chart(self, items, signed=True):
        """Horizontal bars: [(label, value, sd or None, colour or None)]."""
        cv, C = self.bars, self.C
        cv.delete("all")
        W = max(cv.winfo_width(), 200)
        if not items:
            return
        font = (self.F.body_family, 10)
        lab_w = min(220, int(W * 0.48))
        x0, x1 = lab_w + 12, W - 48
        vmax = max(abs(v) + (s or 0) for _, v, s, _ in items) or 1
        mid = (x0 + x1) / 2 if signed else x0
        span = (x1 - x0) / 2 if signed else (x1 - x0)
        rowh = min(20, 200 / len(items))
        for i, (label, val, sd, col) in enumerate(items):
            y = 6 + i * rowh + rowh / 2
            text = label if len(label) <= 34 else label[:33] + "…"
            cv.create_text(lab_w, y, text=text, anchor="e", fill=C["text"], font=font)
            colour = col or (C["accent"] if not signed else ("#d9534f" if val > 0 else "#3d7fd9"))
            x = mid + val / vmax * span
            cv.create_rectangle(min(mid, x), y - rowh * 0.32, max(mid, x), y + rowh * 0.32,
                                fill=colour, outline="")
            if sd:
                cv.create_line(mid + (val - sd) / vmax * span, y, mid + (val + sd) / vmax * span, y,
                               fill=C["muted"])
            cv.create_text(x1 + 44, y, text=f"{val:+.2f}" if signed else f"{val:.2f}", anchor="e",
                           fill=C["soft"], font=font)
        cv.create_line(mid, 2, mid, 8 + len(items) * rowh, fill=C["muted"])

    def _band_plot(self, r, y, sd):
        cv, C = self.plot, self.C
        cv.delete("all")
        W, H = max(cv.winfo_width(), 200), max(cv.winfo_height(), 120)
        L, R, T, B = 52, 10, 10, 34
        y = np.asarray(y, float)
        lo = y - (sd if sd is not None else 0)
        hi = y + (sd if sd is not None else 0)
        ymax = max(float(np.abs(np.concatenate([lo, hi])).max()), 1e-9)
        xmax = float(max(r.max(), 1.2))
        X = lambda v: L + v / xmax * (W - L - R)  # noqa: E731
        Y = lambda v: T + (1 - (v / ymax + 1) / 2) * (H - T - B)  # noqa: E731
        font = (self.F.body_family, 9)
        if sd is not None:
            pts = [c for xv, h in zip(r, hi) for c in (X(xv), Y(h))] + \
                  [c for xv, l in zip(r[::-1], lo[::-1]) for c in (X(xv), Y(l))]
            cv.create_polygon(pts, fill=C["line"], outline="")
        for i in range(len(r) - 1):                      # red / blue fill to zero
            col = "#e8a09d" if (y[i] + y[i + 1]) > 0 else "#9dbbe8"
            cv.create_polygon(X(r[i]), Y(0), X(r[i]), Y(y[i]), X(r[i + 1]), Y(y[i + 1]), X(r[i + 1]), Y(0),
                              fill=col, outline="")
        cv.create_line(*[c for xv, v in zip(r, y) for c in (X(xv), Y(v))], fill=C["text"], width=2)
        cv.create_line(L, Y(0), W - R, Y(0), fill=C["muted"])
        cv.create_line(X(1), T, X(1), H - B, fill=C["muted"], dash=(3, 3))
        for xv, t in ((0, "0 centre"), (0.5, "0.5"), (1.0, "1 rim")):
            cv.create_text(X(xv), H - B + 4, text=t, anchor="n", fill=C["soft"], font=font)
        cv.create_text(W - R, H - 2, text="r / R", anchor="se", fill=C["soft"], font=font)
        cv.create_text(L - 6, Y(ymax), text=f"+{ymax:.2g}", anchor="e", fill=C["soft"], font=font)
        cv.create_text(L - 6, Y(-ymax), text=f"−{ymax:.2g}", anchor="e", fill=C["soft"], font=font)
        cv.create_text(L - 6, Y(ymax * 0.5), text="for", anchor="e", fill="#c0392b", font=font)
        cv.create_text(L - 6, Y(-ymax * 0.5), text="against", anchor="e", fill="#2f6fc0", font=font)
