"""
identify_gui.py — the "Identify" tab of the Whisky Webs app.

Matches whisky-web images against a folder of reference images (one sub-folder
per whisky) by HOG features + cosine similarity, see identify_core.py:
  1. choose the reference folder (and, optionally, the whisky key spreadsheet)
  2. add the images to test (files or folders, e.g. the Stitched folder)
  3. press "Identify"
Click an image to see it next to its closest reference and the best-scoring whiskies.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from PIL import Image, ImageTk

import identify_core as idc
import stitcher_core as core

THUMB = 1000
TOP_N = 5

HELP_TEXT = """\
IDENTIFY: WHICH WHISKY MADE THIS WEB?

1. Reference images: click "Browse…" and choose the folder that holds one
   sub-folder per whisky, each with that whisky's images:

       Learning_Database/
           A/   image1.tif  image2.tif …
           B/   …

   A whisky key spreadsheet (column 1 = folder name, column 2 = whisky
   name, first row = headings) is found automatically if its file name
   contains "key" and it sits in or next to that folder. Without one, the
   folder names are shown, so you can simply name the folders after the
   whiskies. "Check database" matches every reference image against the
   others, to show how well each whisky can be told apart.

2. Test images: "Add folder" or "Add files" – for example the Stitched
   folder made on the Stitch tab.

3. Press "Identify". Each image gets the whisky of its most similar
   reference image, with the cosine similarity (1.000 = identical).
   Click an image to see it next to that reference, and the best score
   for each whisky.

RIGHT OR WRONG

If a test image's name starts with a class code and "_" (O_4.tif = class O)
or it is in a folder named after a class, the expected answer is known and
the RESULT column shows ✓ or ✗. "Save results" writes everything to a CSV.

HOW IT WORKS

Each image is turned grey, shrunk to 128 × 128 pixels and described by its
HOG features (the directions of edges in small patches). The test image is
compared with every reference image and the closest one wins. Method from
Khalifa Mohamed's "Whisky Web Identification" script."""


class IdentifyTab:
    def __init__(self, app, C, badge_image):
        self.app, self.C, self.F = app, C, app.F
        s = app.settings
        self.db_folder = tk.StringVar(value=s.get("id_database", ""))
        self.key_path = tk.StringVar(value=s.get("id_key", ""))
        self.sources = [p for p in s.get("id_tests", []) if os.path.exists(p)]
        self.images = []            # [(path, display name)]
        self.results = {}           # path -> idc.Match
        self.db_items = []          # [(label, path)] from the last scan
        self.names = {}             # class code -> whisky name
        self.key_note = ""
        self.thumbs, self.loading = {}, set()
        self.photos = []
        self.worker = None
        self.stop_flag = threading.Event()
        self._redraw_job = None
        self._build(badge_image)
        self._scan_database(quiet=True)
        self._refresh_images()

    # ================================================================== layout
    def _build(self, badge_image):
        C, F, app = self.C, self.F, self.app
        self.body = ctk.CTkFrame(app, fg_color=C["bg"], corner_radius=0)
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(self.body, fg_color=C["bg"], corner_radius=0, width=540)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        left.grid_propagate(False)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # -- 1 Reference images
        ref = ctk.CTkFrame(left, fg_color=C["surface"], corner_radius=28)
        ref.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ref.grid_columnconfigure(1, weight=1)
        head = ctk.CTkFrame(ref, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(16, 8))
        head.grid_columnconfigure(2, weight=1)
        app._badge(head, 1).grid(row=0, column=0)
        ctk.CTkLabel(head, text="Reference images", font=F.semi(15), text_color=C["text"]).grid(
            row=0, column=1, padx=10)
        app._link(head, "Check database", self.check_database).grid(row=0, column=3)

        pill = ctk.CTkFrame(ref, fg_color=C["bg"], corner_radius=19, height=38)
        pill.grid(row=1, column=0, columnspan=2, sticky="ew", padx=18)
        pill.grid_columnconfigure(0, weight=1)
        entry = ctk.CTkEntry(pill, textvariable=self.db_folder, border_width=0, fg_color=C["bg"],
                             text_color=C["soft"], font=F.body(13), height=30,
                             placeholder_text="Folder with one sub-folder per whisky")
        entry.grid(row=0, column=0, sticky="ew", padx=(14, 4))
        entry.bind("<Return>", lambda e: self._scan_database())
        ctk.CTkButton(pill, text="Browse…", command=self.browse_database, height=30, width=90,
                      corner_radius=15, fg_color=C["text"], hover_color=C["soft"], text_color=C["bg"],
                      font=F.head(13)).grid(row=0, column=1, padx=4, pady=4)

        def label(row, text):
            ctk.CTkLabel(ref, text=text, font=F.body(13), text_color=C["muted"]).grid(
                row=row, column=0, sticky="nw", padx=(20, 10), pady=(8, 0))
        label(2, "Contents")
        self.db_info = ctk.CTkLabel(ref, text="", font=F.body(13), text_color=C["soft"], anchor="w",
                                    justify="left", wraplength=380)
        self.db_info.grid(row=2, column=1, sticky="ew", padx=(0, 18), pady=(8, 0))
        label(3, "Whisky key")
        krow = ctk.CTkFrame(ref, fg_color="transparent")
        krow.grid(row=3, column=1, sticky="ew", padx=(0, 18), pady=(4, 16))
        krow.grid_columnconfigure(0, weight=1)
        self.key_info = ctk.CTkLabel(krow, text="", font=F.body(13), text_color=C["soft"], anchor="w",
                                     justify="left", wraplength=280)
        self.key_info.grid(row=0, column=0, sticky="ew")
        app._link(krow, "Change", self.browse_key).grid(row=0, column=1)
        app._link(krow, "None", self.clear_key).grid(row=0, column=2)

        # -- 2 Test images
        card = ctk.CTkFrame(left, fg_color=C["surface"], corner_radius=28)
        card.grid(row=1, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=(16, 12), pady=(14, 6))
        head.grid_columnconfigure(3, weight=1)
        app._badge(head, 2).grid(row=0, column=0)
        ctk.CTkLabel(head, text="Test images", font=F.semi(15), text_color=C["text"]).grid(
            row=0, column=1, padx=(10, 8))
        self.count_label = ctk.CTkLabel(head, text="", font=F.body(13), text_color=C["muted"])
        self.count_label.grid(row=0, column=2)
        links = ctk.CTkFrame(head, fg_color="transparent")
        links.grid(row=0, column=4)
        app._link(links, "Add folder", self.add_folder).pack(side="left")
        app._link(links, "Add files", self.add_files).pack(side="left")
        app._link(links, "Clear", self.clear_images).pack(side="left")

        holder = tk.Frame(card, bg=C["surface"], highlightthickness=0)
        holder.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 14))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        cols = ("image", "match", "score", "result")
        self.tree = ttk.Treeview(holder, columns=cols, show="headings", selectmode="browse",
                                 style="WW.Treeview")
        for col, text, w, anchor, stretch in (
                ("image", "IMAGE", 150, "w", True), ("match", "CLOSEST WHISKY", 150, "w", True),
                ("score", "SIMILARITY", 90, "center", False), ("result", "RESULT", 70, "center", False)):
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=w, minwidth=60, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(holder, command=self.tree.yview, fg_color=C["surface"],
                              button_color=C["line"], button_hover_color=C["muted"])
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        for tag, colour in (("ok", C["sage_ink"]), ("bad", C["error"]), ("wait", C["muted"]),
                            ("plain", C["text"])):
            self.tree.tag_configure(tag, foreground=colour)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._draw())

        # -- previews
        right = ctk.CTkFrame(self.body, fg_color=C["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure((0, 1), weight=1, uniform="wells")
        right.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(right, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.cur_title = ctk.CTkLabel(head, text="", font=F.head(30), text_color=C["accent"])
        self.cur_title.grid(row=0, column=0, padx=(4, 14))
        self.cur_summary = ctk.CTkLabel(head, text="", font=F.body(15), text_color=C["soft"])
        self.cur_summary.grid(row=0, column=1)

        self.wells = []
        for col, title in ((0, "Test image"), (1, "Closest reference")):
            w = ctk.CTkFrame(right, fg_color=C["well"], corner_radius=28)
            w.grid(row=1, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0))
            w.grid_rowconfigure(1, weight=1)
            w.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(w, text=title, font=F.head(17), text_color=C["bg"]).grid(
                row=0, column=0, sticky="w", padx=24, pady=(18, 0))
            cv = tk.Canvas(w, background=C["well"], highlightthickness=0)
            cv.grid(row=1, column=0, sticky="nsew", padx=18, pady=8)
            cv.bind("<Configure>", lambda e: self._schedule_draw())
            note = ctk.CTkLabel(w, text="", font=F.body(12), text_color=C["well_faint"], anchor="w")
            note.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 16))
            self.wells.append((cv, note))

        rank = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=28)
        rank.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        rank.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(rank, text="Best score for each whisky", font=F.semi(15), text_color=C["text"]).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(14, 4))
        self.rank_rows = []
        for i in range(TOP_N):
            name = ctk.CTkLabel(rank, text="", font=F.body(13), text_color=C["text"], anchor="w", width=220)
            name.grid(row=i + 1, column=0, sticky="w", padx=(20, 10), pady=2)
            bar = ctk.CTkProgressBar(rank, height=8, corner_radius=4, fg_color=C["track"],
                                     progress_color=C["accent"])
            bar.grid(row=i + 1, column=1, sticky="ew")
            score = ctk.CTkLabel(rank, text="", font=F.semi(13), text_color=C["text"], width=60)
            score.grid(row=i + 1, column=2, padx=(10, 20))
            self.rank_rows.append((name, bar, score))
        ctk.CTkFrame(rank, fg_color="transparent", height=10).grid(row=TOP_N + 1, column=0)

        # -- run bar
        self.runbar = bar = ctk.CTkFrame(app, fg_color=C["surface"], corner_radius=0)
        bar.grid_columnconfigure(2, weight=1)
        self.run_btn = ctk.CTkButton(bar, text="Identify", image=badge_image(3), compound="left",
                                     command=self.start, height=56, corner_radius=28,
                                     fg_color=C["accent"], hover_color=C["accent_hover"],
                                     text_color=C["bg"], text_color_disabled=C["bg"],
                                     font=F.head(19), width=300)
        self.run_btn.grid(row=0, column=0, padx=(28, 12), pady=16)
        self.stop_btn = app._pill_button(bar, "Stop", self.stop, height=56, size=15, width=96,
                                         text_color_disabled=C["line"])
        self.stop_btn.grid(row=0, column=1)
        self.stop_btn.configure(state="disabled")
        mid = ctk.CTkFrame(bar, fg_color="transparent")
        mid.grid(row=0, column=2, sticky="ew", padx=18)
        mid.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(mid, height=8, corner_radius=4, fg_color=C["track"],
                                           progress_color=C["track"])
        self.progress.set(0)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.status_label = ctk.CTkLabel(mid, text="Choose the reference images to begin", anchor="w",
                                         font=F.body(13), text_color=C["soft"])
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        app._link(mid, "Show log", app.show_log, 12).grid(row=1, column=1, pady=(4, 0))
        app._pill_button(bar, "Save results (CSV)", self.save_csv, height=44, size=14).grid(
            row=0, column=3, padx=(0, 28))

    def show(self):
        self.body.grid(row=1, column=0, sticky="nsew", padx=28, pady=20)
        self.runbar.grid(row=2, column=0, sticky="ew")
        self._schedule_draw()

    def hide(self):
        self.body.grid_remove()
        self.runbar.grid_remove()

    def settings(self):
        return dict(id_database=self.db_folder.get(), id_key=self.key_path.get(), id_tests=self.sources)

    def busy(self):
        return bool(self.worker and self.worker.is_alive())

    def log(self, msg):
        self.app.logs.append(msg)
        if msg.strip():
            self.status_label.configure(text=msg.strip())

    # =========================================================== reference set
    def browse_database(self):
        d = filedialog.askdirectory(title="Choose the folder with one sub-folder per whisky",
                                    initialdir=self.db_folder.get() or os.path.expanduser("~"))
        if d:
            self.db_folder.set(os.path.normpath(d))
            self.key_path.set("")           # look for this folder's own key
            self._scan_database()

    def _scan_database(self, quiet=False):
        folder = self.db_folder.get().strip()
        self.db_items = []
        if not folder:
            self.db_info.configure(text="Not chosen yet")
        elif not os.path.isdir(folder):
            self.db_info.configure(text="This folder doesn't exist")
        else:
            try:
                self.db_items = idc.scan_database(folder)
            except OSError as e:
                self.db_info.configure(text=f"Couldn't read the folder: {e}")
            else:
                classes = sorted({lab for lab, _ in self.db_items})
                if not classes:
                    self.db_info.configure(text="No images in sub-folders – see Help for the layout")
                else:
                    self.db_info.configure(
                        text=f"{len(classes)} whisk{'y' if len(classes) == 1 else 'ies'} · "
                             f"{len(self.db_items)} images ({', '.join(classes[:8])}"
                             f"{', …' if len(classes) > 8 else ''})")
                if not self.key_path.get():
                    found = idc.find_key(folder)
                    if found:
                        self.key_path.set(found)
        self._load_key()
        if not quiet and self.db_items:
            self.log(f"Reference folder: {folder} ({len(self.db_items)} images)")

    def _classes(self):
        return sorted({lab for lab, _ in self.db_items})

    def _load_key(self):
        path = self.key_path.get()
        self.names = {}
        if not path or path == "-":
            self.key_info.configure(text="None – folder names are shown")
        elif not os.path.isfile(path):
            self.key_info.configure(text=f"{os.path.basename(path)} not found – folder names are shown")
        else:
            try:
                self.names = idc.load_key(path, self._classes())
            except Exception as e:
                self.key_info.configure(text=f"Couldn't read {os.path.basename(path)}: {e}")
            else:
                classes = self._classes()
                missing = [c for c in classes if c not in self.names]
                text = f"{os.path.basename(path)} · {len(self.names)} names"
                if classes and missing:
                    text += f" · no name for {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}"
                self.key_info.configure(text=text)
        self._refresh_table()
        self._draw()

    def browse_key(self):
        cur = self.key_path.get()
        f = filedialog.askopenfilename(
            title="Choose the whisky key (column 1 = folder name, column 2 = whisky name)",
            initialdir=os.path.dirname(cur) if cur and cur != "-" else (self.db_folder.get() or None),
            filetypes=[("Spreadsheets", "*.xlsx *.xlsm *.csv"), ("All files", "*.*")])
        if f:
            self.key_path.set(os.path.normpath(f))
            self._load_key()

    def clear_key(self):
        self.key_path.set("-")              # "-" = deliberately none (don't auto-find again)
        self._load_key()

    def whisky(self, label):
        return self.names.get(label, label)

    # ============================================================= test images
    def add_folder(self):
        d = filedialog.askdirectory(title="Choose a folder of images to identify",
                                    initialdir=self.app.folder.get() or os.path.expanduser("~"))
        if d:
            self._add([os.path.normpath(d)])

    def add_files(self):
        exts = " ".join("*" + e for e in core.IMG_EXTS)
        fs = filedialog.askopenfilenames(title="Choose images to identify",
                                         filetypes=[("Images", exts), ("All files", "*.*")])
        if fs:
            self._add([os.path.normpath(f) for f in fs])

    def _add(self, paths):
        self.sources += [p for p in paths if p not in self.sources]
        self._refresh_images()
        if not self.images:
            messagebox.showinfo(self.app.title(), "No images found there.")

    def clear_images(self):
        if self.busy():
            return
        self.sources, self.results = [], {}
        self._refresh_images()

    def _refresh_images(self):
        self.images, seen = [], set()
        for src in self.sources:
            for p in idc.list_test_images([src]):
                if p not in seen:
                    seen.add(p)
                    shown = os.path.relpath(p, src) if os.path.isdir(src) else os.path.basename(p)
                    self.images.append((p, shown))
        self._refresh_table()
        kids = self.tree.get_children()
        if kids and not self.tree.selection():
            self.tree.selection_set(kids[0])
        self._draw()

    def _refresh_table(self):
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for i, (path, shown) in enumerate(self.images):
            m = self.results.get(path)
            if m is None:
                vals, tag = (shown, "", "", ""), "wait"
            elif m.error:
                vals, tag = (shown, "Couldn't open", "", "!"), "bad"
            else:
                result = {True: "✓", False: "✗", None: "—"}[m.correct]
                tag = {True: "ok", False: "bad", None: "plain"}[m.correct]
                vals = (shown, self.whisky(m.label), f"{m.score:.3f}", result)
            self.tree.insert("", "end", iid=str(i), values=vals, tags=(tag,))
        keep = [n for n in sel if self.tree.exists(n)]
        if keep:
            self.tree.selection_set(keep)
        n = len(self.images)
        text = f"{n}" if n else ""
        scored = [m for p, _ in self.images for m in [self.results.get(p)] if m and m.correct is not None]
        if scored:
            right = sum(m.correct for m in scored)
            text += f" · {100 * right / len(scored):.0f}% right"
        self.count_label.configure(text=text)
        if not self.busy():
            self.run_btn.configure(text=f"Identify {n} image{'s' if n != 1 else ''}" if n else "Identify")

    def _current(self):
        sel = self.tree.selection()
        if not sel:
            return None
        i = int(sel[0])
        return self.images[i][0] if i < len(self.images) else None

    # ================================================================ previews
    def _schedule_draw(self, delay=60):
        if self._redraw_job:
            self.app.after_cancel(self._redraw_job)

        def go():
            self._redraw_job = None
            self._draw()
        self._redraw_job = self.app.after(delay, go)

    def _thumb(self, path):
        """Thumbnail, or False while it loads in the background (None if unreadable)."""
        if path in self.thumbs:
            return self.thumbs[path]
        if path not in self.loading:
            self.loading.add(path)

            def work():
                try:
                    im = Image.fromarray(core.read_rgb(path))
                    im.thumbnail((THUMB, THUMB))
                except Exception:
                    im = None
                self.app.events.put(("id_thumb", (path, im)))
            threading.Thread(target=work, daemon=True).start()
        return False

    def _show_image(self, cv, path):
        if not path:
            return
        im = self._thumb(path)
        if im is False:
            self.app._message(cv, "Loading…")
            return
        if im is None:
            self.app._message(cv, "Couldn't open this image", self.C["order"])
            return
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        scale = min((W - 24) / im.width, (H - 24) / im.height)
        w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        ph = ImageTk.PhotoImage(im.resize((w, h), Image.LANCZOS if scale < 1 else Image.BILINEAR))
        self.photos.append(ph)
        cv.create_image(W / 2, H / 2, image=ph)

    def _draw(self):
        if not self.body.winfo_ismapped():
            return
        (tcv, tnote), (rcv, rnote) = self.wells
        for cv in (tcv, rcv):
            cv.delete("all")
        self.photos.clear()
        path = self._current()
        m = self.results.get(path) if path else None
        ranking = m.ranking[:TOP_N] if m and not m.error else []
        for i, (name, bar, score) in enumerate(self.rank_rows):
            if i < len(ranking):
                lab, sc, _ = ranking[i]
                extra = "   (expected)" if m.expected == lab and lab != m.label else ""
                name.configure(text=self.whisky(lab) + extra,
                               text_color=self.C["accent_deep"] if i == 0 else self.C["text"])
                bar.set(max(0.0, min(1.0, sc)))
                bar.configure(progress_color=self.C["accent"] if i == 0 else self.C["muted"])
                score.configure(text=f"{sc:.3f}")
                for w in (name, bar, score):
                    w.grid()
            else:
                for w in (name, bar, score):
                    w.grid_remove()

        if path is None:
            self.cur_title.configure(text="")
            self.cur_summary.configure(text="")
            tnote.configure(text="")
            rnote.configure(text="")
            self.app._message(tcv, "Add the images you want to identify")
            return
        tnote.configure(text=os.path.basename(path))
        self._show_image(tcv, path)
        if m is None:
            self.cur_title.configure(text="")
            self.cur_summary.configure(text="Not identified yet")
            rnote.configure(text="")
            self.app._message(rcv, "Press Identify")
            return
        if m.error:
            self.cur_title.configure(text="")
            self.cur_summary.configure(text="")
            rnote.configure(text="")
            self.app._message(rcv, "Couldn't open this image:\n" + m.error, self.C["order"])
            return
        self.cur_title.configure(text=self.whisky(m.label))
        summary = f"similarity {m.score:.3f}"
        if len(m.ranking) > 1:
            summary += f" · next {self.whisky(m.ranking[1][0])} {m.ranking[1][1]:.3f}"
        if m.correct is True:
            summary += " · ✓ matches the file name"
        elif m.correct is False:
            summary += f" · ✗ expected {self.whisky(m.expected)}"
        self.cur_summary.configure(text=summary)
        rnote.configure(text=os.path.relpath(m.ref_path, self.db_folder.get())
                        if self.db_folder.get() else os.path.basename(m.ref_path))
        self._show_image(rcv, m.ref_path)

    # ================================================================= running
    def _check_ready(self):
        if self.busy():
            return False
        self._scan_database(quiet=True)
        if not self.db_items:
            messagebox.showinfo(self.app.title(), "Choose a reference folder with one sub-folder of "
                                                  "images per whisky first (see Help).")
            return False
        try:
            import cv2  # noqa: F401
        except ImportError:
            messagebox.showwarning(self.app.title(), "The image-matching part (OpenCV) isn't installed.\n\n"
                                   "Run:  python -m pip install opencv-python-headless openpyxl")
            return False
        return True

    def _run(self, label, job):
        """Build the database, then job(db) in the background."""
        folder = self.db_folder.get().strip()
        self.app._save_settings()
        self.stop_flag.clear()
        self.progress.set(0)
        self.progress.configure(progress_color=self.C["accent"])
        self.run_btn.configure(state="disabled", text=label)
        self.stop_btn.configure(state="normal")
        self.log("")
        ev = self.app.events

        def work():
            try:
                ev.put(("id_log", f"Reading {len(self.db_items)} reference images…"))
                db = idc.build_database(folder, log=lambda msg: ev.put(("id_log", msg)),
                                        progress=lambda d, t: ev.put(("id_progress", 0.5 * d / t)),
                                        should_stop=self.stop_flag.is_set)
                job(db)
            except InterruptedError:
                ev.put(("id_log", "Stopped."))
            except Exception as e:
                ev.put(("id_log", f"Error: {e}"))
            ev.put(("id_finished", None))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def start(self):
        if not self.images:
            messagebox.showinfo(self.app.title(), "Add the images to identify first (Add folder / Add files).")
            return
        if not self._check_ready():
            return
        todo = [p for p, _ in self.images]
        for p in todo:
            self.results.pop(p, None)
        self._refresh_table()
        self._draw()
        ev = self.app.events

        def job(db):
            ev.put(("id_log", f"Identifying {len(todo)} image(s) against {len(db.classes)} whiskies…"))
            for i, p in enumerate(todo):
                if self.stop_flag.is_set():
                    raise InterruptedError
                ev.put(("id_result", idc.identify(p, db)))
                ev.put(("id_progress", 0.5 + 0.5 * (i + 1) / len(todo)))
            ev.put(("id_done", None))
        self._run(f"Identifying {len(todo)} image{'s' if len(todo) != 1 else ''}…", job)

    def check_database(self):
        if not self._check_ready():
            return

        def job(db):
            ev = self.app.events
            if len(db.paths) < 2:
                raise ValueError("at least two reference images are needed")
            ev.put(("id_loo", (db, *idc.leave_one_out(db))))
        self._run("Checking database…", job)

    def stop(self):
        self.stop_flag.set()
        self.log("Stopping…")

    def handle(self, kind, data):
        """Events from the background threads (via the app's queue)."""
        if kind == "id_log":
            self.log(data)
        elif kind == "id_progress":
            self.progress.set(data)
        elif kind == "id_thumb":
            path, im = data
            self.loading.discard(path)
            self.thumbs[path] = im
            self._schedule_draw()
        elif kind == "id_result":
            m = data
            self.results[m.path] = m
            if m.error:
                self.log(f"  {os.path.basename(m.path)}: couldn't open ({m.error})")
            else:
                verdict = {True: " ✓", False: f" ✗ expected {self.whisky(m.expected)}", None: ""}[m.correct]
                self.log(f"  {os.path.basename(m.path)}: {self.whisky(m.label)} "
                         f"(similarity {m.score:.3f}){verdict}")
            self._refresh_table()
            if m.path == self._current():
                self._draw()
        elif kind == "id_done":
            ms = [self.results[p] for p, _ in self.images if p in self.results]
            scored = [m for m in ms if m.correct is not None]
            msg = f"Identified {sum(not m.error for m in ms)} image(s)"
            if scored:
                right = sum(m.correct for m in scored)
                msg += f" · {right} of {len(scored)} match their file name ({100 * right / len(scored):.0f}%)"
            self.log(msg)
        elif kind == "id_loo":
            self._show_loo(*data)
        elif kind == "id_finished":
            self.stop_btn.configure(state="disabled")
            self.run_btn.configure(state="normal")
            self._refresh_table()

    def _show_loo(self, db, per_class, misses):
        ok = sum(c for c, _ in per_class.values())
        total = sum(t for _, t in per_class.values())
        lines = ["DATABASE CHECK (leave one out)", "",
                 "Each reference image is matched against all the other reference",
                 "images. A whisky that is often mistaken for another needs more,",
                 "or more distinct, reference images.", "",
                 f"Overall: {ok} of {total} right ({100 * ok / max(1, total):.0f}%)", ""]
        width = max(len(self.whisky(c)) for c in per_class) + 2
        for c in sorted(per_class):
            r, t = per_class[c]
            lines.append(f"  {self.whisky(c):<{width}} {r:>3} / {t:<3} {100 * r / t:>4.0f}%"
                         + ("   (only one image – nothing to compare)" if t == 1 else ""))
        if misses:
            lines += ["", "MISTAKES", ""]
            for path, true, got, score in misses:
                lines.append(f"  {os.path.relpath(path, db.folder)}: matched {self.whisky(got)} "
                             f"({score:.3f}), is {self.whisky(true)}")
        if db.skipped:
            lines += ["", "COULDN'T OPEN", ""] + [f"  {os.path.relpath(p, db.folder)}: {e}" for p, e in db.skipped]
        self.log(f"Database check: {ok} of {total} reference images matched their own whisky")
        self.app._text_window("Database check", "\n".join(lines), 760, 640)

    def save_csv(self):
        ms = [self.results[p] for p, _ in self.images if p in self.results]
        if not ms:
            messagebox.showinfo(self.app.title(), "Nothing identified yet – press Identify first.")
            return
        first = self.sources[0] if self.sources else ""
        start = first if os.path.isdir(first) else os.path.dirname(first)
        f = filedialog.asksaveasfilename(title="Save results", defaultextension=".csv",
                                         initialdir=start or None, initialfile="identification_results.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not f:
            return
        try:
            idc.write_csv(ms, f, self.names)
        except OSError as e:
            messagebox.showerror(self.app.title(), f"Couldn't save:\n{e}")
            return
        self.log(f"Saved results to {f}")
