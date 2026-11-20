"""
restitch_dialog.py — "Restitch…" window, opened from the Gallery's right-click menu.

Finds the tiles a stitched image was made from (gallery_gui.stitch_source), shows
the Stitch tab's options for that one sample (rows, columns, order, overlap,
refine) with the tile layout, the current image and a low-res preview of the new
stitch side by side, then stitches it again at full size with the Python engine
and replaces the image (the old one is kept in Stitched/_replaced by default).
"""
import datetime
import os
import queue
import shutil
import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

import stitcher_core as core

REPLACED_DIR = "_replaced"


class RestitchDialog:
    def __init__(self, app, C, image_path, on_done=None):
        self.app, self.C, self.F = app, C, app.F
        self.path, self.on_done = image_path, on_done
        self.q = queue.Queue()
        self.sample, self.folder = None, None
        self.tiles = {}                  # tile path -> PIL thumbnail (None if unreadable)
        self.current = None              # PIL thumbnail of the image being replaced
        self.preview = None              # {"image", "rows", "cols"} or error text
        self.gen = 0                     # bumped on each option change; stale previews are dropped
        self.busy = False
        self._job = None
        self._photos = {}
        s = app.settings if hasattr(app, "settings") else {}
        self.rows, self.cols = tk.IntVar(value=1), tk.IntVar(value=1)
        self.order = tk.StringVar(value=core.DEFAULT_ORDER)
        self.overlap = tk.IntVar(value=int(app.overlap.get()) if hasattr(app, "overlap") else s.get("overlap", 20))
        self.refine = tk.BooleanVar(value=bool(app.refine.get()) if hasattr(app, "refine") else True)
        self.keep_old = tk.BooleanVar(value=True)
        self.stitch_folder = app.folder.get() if hasattr(app, "folder") else ""
        self._build()
        self.win.after(100, self._poll)
        threading.Thread(target=self._find, daemon=True).start()

    # ------------------------------------------------------------------ UI
    def _build(self):
        C, F, app = self.C, self.F, self.app
        self.win = win = ctk.CTkToplevel(app, fg_color=C["bg"])
        win.title("Restitch")
        win.geometry("1320x760")
        win.minsize(1000, 600)
        win.transient(app)
        win.protocol("WM_DELETE_WINDOW", self.close)
        win.grid_columnconfigure(1, weight=1)
        win.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(win, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(18, 10))
        self.title = ctk.CTkLabel(head, text="Restitch", font=F.head(24), text_color=C["text"], anchor="w")
        self.title.pack(anchor="w")
        self.sub = ctk.CTkLabel(head, text="Finding the tiles this image was stitched from…", font=F.body(13),
                                text_color=C["muted"], anchor="w", justify="left", wraplength=1200)
        self.sub.pack(anchor="w")

        # -- options
        opts = ctk.CTkFrame(win, fg_color=C["surface"], corner_radius=24, width=290)
        opts.grid(row=1, column=0, sticky="nsw", padx=(24, 14), pady=(0, 10))
        opts.grid_propagate(False)
        opts.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(opts, text="Options", font=F.semi(15), text_color=C["text"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 10))

        def label(row, text):
            ctk.CTkLabel(opts, text=text, font=F.body(13), text_color=C["muted"]).grid(
                row=row, column=0, sticky="w", padx=(18, 8), pady=5)
        label(1, "Rows")
        self._stepper(opts, self.rows).grid(row=1, column=1, sticky="w")
        label(2, "Columns")
        self._stepper(opts, self.cols).grid(row=2, column=1, sticky="w")
        label(3, "Order")
        ctk.CTkOptionMenu(opts, values=list(core.ORDERS), variable=self.order, width=170, height=30,
                          corner_radius=15, font=F.body(12), dropdown_font=F.body(12), fg_color=C["bg"],
                          button_color=C["line"], button_hover_color=C["muted"], text_color=C["text"],
                          command=lambda v: self._changed()).grid(row=3, column=1, sticky="w")
        label(4, "Overlap")
        ov = ctk.CTkFrame(opts, fg_color="transparent")
        ov.grid(row=4, column=1, sticky="ew", padx=(0, 14))
        ov.grid_columnconfigure(0, weight=1)
        self.ov_label = ctk.CTkLabel(ov, text=f"{self.overlap.get()}%", width=40, font=F.body(12),
                                     text_color=C["text"])
        self.ov_label.grid(row=0, column=1)
        sl = ctk.CTkSlider(ov, from_=1, to=60, number_of_steps=59, height=16, command=self._on_overlap,
                           button_color=C["accent"], button_hover_color=C["accent_hover"],
                           progress_color=C["accent"], fg_color=C["line"])
        sl.set(self.overlap.get())
        sl.grid(row=0, column=0, sticky="ew")
        for row, (text, var) in enumerate((("Refine positions (match the overlaps)", self.refine),
                                           ("Keep the old image in Stitched/_replaced", self.keep_old)), 5):
            ctk.CTkCheckBox(opts, text=text, variable=var, font=F.body(12), text_color=C["text"],
                            fg_color=C["accent"], hover_color=C["accent_hover"], border_color=C["line"],
                            checkbox_width=18, checkbox_height=18,
                            command=self._changed if var is self.refine else None).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=18, pady=(10 if row == 5 else 4, 0))
        self.problem = ctk.CTkLabel(opts, text="", font=F.body(12), text_color=C["error"], anchor="w",
                                    justify="left", wraplength=250)
        self.problem.grid(row=7, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 0))
        ctk.CTkLabel(opts, text="Engine: Python (full-size tiles). The preview on the right is a quick "
                                "low-res stitch with these options.",
                     font=F.body(11), text_color=C["muted"], anchor="w", justify="left", wraplength=250).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 0))

        # -- three views
        views = ctk.CTkFrame(win, fg_color="transparent")
        views.grid(row=1, column=1, sticky="nsew", padx=(0, 24), pady=(0, 10))
        views.grid_rowconfigure(0, weight=1)
        self.canvases = {}
        for i, (key, title) in enumerate((("layout", "Tiles"), ("current", "Current image"),
                                          ("new", "New (preview)"))):
            views.grid_columnconfigure(i, weight=1, uniform="v")
            card = ctk.CTkFrame(views, fg_color=C["well"], corner_radius=20)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            card.grid_rowconfigure(1, weight=1)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text=title, font=F.semi(14), text_color=C["well_text"]).grid(
                row=0, column=0, sticky="w", padx=16, pady=(12, 4))
            cv = tk.Canvas(card, background=C["well"], highlightthickness=0)
            cv.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 12))
            cv.bind("<Configure>", lambda e: self._schedule_draw())
            self.canvases[key] = cv

        # -- bottom bar
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=24, pady=(0, 18))
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text="", font=F.body(13), text_color=C["soft"], anchor="w")
        self.status.grid(row=0, column=0, sticky="w")
        self.cancel_btn = app._link(bar, "Close", self.close, 14)
        self.cancel_btn.grid(row=0, column=1, padx=(0, 10))
        self.go_btn = ctk.CTkButton(bar, text="Stitch & replace", command=self.stitch, height=40, width=180,
                                    corner_radius=20, fg_color=C["accent"], hover_color=C["accent_hover"],
                                    text_color="#ffffff", font=F.head(15), state="disabled")
        self.go_btn.grid(row=0, column=2)
        win.after(60, win.lift)
        win.after(90, win.focus_force)

    def _stepper(self, parent, var):
        C, F = self.C, self.F
        f = ctk.CTkFrame(parent, fg_color="transparent")

        def step(d):
            if not self.busy:
                var.set(max(1, min(20, var.get() + d)))
                self._changed()
        for col, (txt, d) in enumerate((("−", -1), (None, 0), ("+", 1))):
            if txt is None:
                ctk.CTkLabel(f, textvariable=var, width=34, font=F.semi(14), text_color=C["text"]).grid(
                    row=0, column=col)
            else:
                ctk.CTkButton(f, text=txt, width=30, height=30, corner_radius=15, fg_color=C["bg"],
                              hover_color=C["hover_tint"], text_color=C["text"], font=F.semi(15),
                              command=lambda d=d: step(d)).grid(row=0, column=col)
        return f

    def _on_overlap(self, v):
        v = int(round(v))
        if v != self.overlap.get():
            self.overlap.set(v)
            self.ov_label.configure(text=f"{v}%")
            self._changed()

    def _say(self, text, colour=None):
        self.status.configure(text=text, text_color=colour or self.C["soft"])

    # ------------------------------------------------------------- loading
    def _find(self):
        """Background: locate the tiles, then load thumbnails of them and of the image."""
        from gallery_gui import cached_thumb, stitch_source
        from stitch_gui import TILE_THUMB, cached_thumbnail
        try:
            src = stitch_source(self.path, [self.stitch_folder])
            if not src:
                self.q.put(("error", "The tiles this image was stitched from weren't found.\n\nThe Gallery "
                                     "looks in the folder above its \"Stitched\" folder and in the folder "
                                     "open on the Stitch tab. Open the tile folder on the Stitch tab and "
                                     "try again."))
                return
            folder, name = src
            sample = next(s for s in core.scan_folder(folder)[0] if s.name == name)
            self.q.put(("found", (folder, sample)))
            try:
                self.q.put(("current", cached_thumb(self.path)))
            except Exception:  # noqa: BLE001
                self.q.put(("current", None))
            for p in sample.paths:
                try:
                    im = cached_thumbnail(p, TILE_THUMB)
                except Exception:  # noqa: BLE001
                    im = None
                self.q.put(("tile", (p, im)))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"Couldn't read the tile folder:\n{e}"))

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                kind, data = self.q.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        self.win.after(100, self._poll)

    def _handle(self, kind, data):
        if kind == "error":
            self.sub.configure(text=data.split("\n")[0])
            self._say("")
            messagebox.showinfo("Restitch", data, parent=self.win)
            self.close()
        elif kind == "found":
            self.folder, s = data
            app = self.app
            # use the Stitch tab's settings for this sample if that folder is open there
            if os.path.normcase(os.path.normpath(app.folder.get() or ".")) == os.path.normcase(self.folder) \
                    and s.name in getattr(app, "samples", {}):
                live = app.samples[s.name]
                s.rows, s.cols, s.order = live.rows, live.cols, live.order
            self.sample = s
            self.rows.set(s.rows)
            self.cols.set(s.cols)
            self.order.set(s.order)
            warn = f"   ·   {'; '.join(s.warnings)}" if s.warnings else ""
            self.title.configure(text=f"Restitch {s.name}")
            self.sub.configure(text=f"{s.n} tile{'s' if s.n != 1 else ''} in {self.folder}{warn}")
            self._say("Loading the tiles…")
            self._draw()
        elif kind == "current":
            self.current = data
            self._draw()
        elif kind == "tile":
            p, im = data
            self.tiles[p] = im
            if self.sample and len(self.tiles) == self.sample.n:
                self._say("")
                self._changed()
            else:
                self._draw()
        elif kind == "preview":
            gen, result = data
            if gen == self.gen:
                self.preview = result
                self._draw()
        elif kind == "log":
            self._say(data.strip())
        elif kind == "stitched":
            self._finished(*data)

    def _ready(self):
        return self.sample is not None and len(self.tiles) == self.sample.n

    # ------------------------------------------------------------- changes
    def _apply(self):
        s = self.sample
        s.rows, s.cols, s.order = self.rows.get(), self.cols.get(), self.order.get()
        return s.problem() if s.n > 2 else None

    def _changed(self):
        if self.sample is None:
            return
        problem = self._apply()
        self.problem.configure(text=problem or "")
        self.gen += 1
        self.preview = None
        self.go_btn.configure(state="disabled" if (problem or self.busy or not self._ready()) else "normal")
        if self._job:
            self.win.after_cancel(self._job)
        self._job = self.win.after(250, self._start_preview)
        self._draw()

    def _start_preview(self):
        """Quick stitch of shrunken tiles in the background (as on the Stitch tab)."""
        self._job = None
        if not self._ready() or self._apply():
            return
        gen, s = self.gen, self.sample
        thumbs = [self.tiles[p] for p in s.paths]
        rows, cols, order = s.rows, s.cols, s.order
        overlap, refine = self.overlap.get() / 100.0, bool(self.refine.get())

        def work():
            import numpy as np
            import stitch_native
            from stitch_gui import QUICK_EDGE
            try:
                if any(t is None for t in thumbs):
                    raise ValueError("a tile couldn't be opened")
                f = min(1.0, QUICK_EDGE / max(thumbs[0].size))
                size = (max(1, int(thumbs[0].width * f)), max(1, int(thumbs[0].height * f)))
                arrs = [np.asarray(t.convert("RGB").resize(size), dtype=np.float32) for t in thumbs]
                r, c = rows, cols
                if len(arrs) == 1:
                    mosaic = arrs[0].astype(np.uint8)
                else:
                    if len(arrs) == 2:
                        r, c = stitch_native.auto_orient_pair(arrs[0], arrs[1])
                    pos = core.tile_positions(len(arrs), r, c, order)
                    mosaic = stitch_native.stitch_arrays(arrs, pos, overlap, refine=refine, verbose=False)
                result = {"image": Image.fromarray(mosaic), "rows": r, "cols": c}
            except Exception as e:  # noqa: BLE001
                result = str(e)
            self.q.put(("preview", (gen, result)))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------- drawing
    def _schedule_draw(self):
        if getattr(self, "_draw_job", None):
            self.win.after_cancel(self._draw_job)
        self._draw_job = self.win.after(60, self._draw)

    def _message(self, cv, text, colour=None):
        cv.delete("all")
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        cv.create_text(W / 2, H / 2, text=text, fill=colour or self.C["well_text"], width=W - 30,
                       font=(self.F.body_family, 12), justify="center")

    def _fit(self, key, im):
        cv = self.canvases[key]
        cv.delete("all")
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        f = min((W - 10) / im.width, (H - 10) / im.height)
        im = im.convert("RGB").resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.LANCZOS)
        ph = self._photos[key] = ImageTk.PhotoImage(im)
        cv.create_image(W / 2, H / 2, image=ph)
        return W, H, f

    def _draw(self):
        self._draw_job = None
        if not self.win.winfo_exists():
            return
        C = self.C
        # current image
        cvc = self.canvases["current"]
        if self.current is None:
            self._message(cvc, "Loading…")
        else:
            self._fit("current", self.current)
        s = self.sample
        if s is None:
            self._message(self.canvases["layout"], "Finding the tiles…")
            self._message(self.canvases["new"], "")
            return
        if not self._ready():
            text = f"Loading tiles… {len(self.tiles)} / {s.n}"
            self._message(self.canvases["layout"], text)
            self._message(self.canvases["new"], text)
            return
        problem = self._apply()
        if problem:
            self._message(self.canvases["layout"], problem, C["order"])
            self._message(self.canvases["new"], "Fix the layout first.", C["order"])
            return
        self._draw_layout()
        if self.preview is None:
            self._message(self.canvases["new"], "Working out the stitch…")
        elif isinstance(self.preview, str):
            self._message(self.canvases["new"], "Preview failed:\n" + self.preview, C["order"])
        else:
            self._fit("new", self.preview["image"])

    def _draw_layout(self):
        """Tiles in their grid positions with the overlaps shaded and the visiting order."""
        C, s, cv = self.C, self.sample, self.canvases["layout"]
        rows, cols = s.rows, s.cols
        if s.n == 2:
            rows, cols = (self.preview["rows"], self.preview["cols"]) if isinstance(self.preview, dict) else (1, 2)
        first = next((self.tiles[p] for p in s.paths if self.tiles[p] is not None), None)
        A = (first.height / first.width) if first else 1.0
        o = self.overlap.get() / 100.0
        span_w, span_h = cols - (cols - 1) * o, (rows - (rows - 1) * o) * A
        W, H = max(cv.winfo_width(), 60), max(cv.winfo_height(), 60)
        unit = min((W - 20) / span_w, (H - 20) / span_h)
        tw, th = unit, unit * A
        full_w, full_h = span_w * unit, span_h * unit
        x0, y0 = (W - full_w) / 2, (H - full_h) / 2
        layer = Image.new("RGBA", (max(1, int(full_w)), max(1, int(full_h))), (0, 0, 0, 0))
        pos = core.tile_positions(s.n, rows, cols, s.order)
        centres = []
        for (r, c), p in zip(pos, s.paths):
            x, y = c * (1 - o) * tw, r * (1 - o) * th
            im = self.tiles[p]
            if im is not None:
                layer.paste(im.convert("RGB").resize((max(1, int(tw)), max(1, int(th)))), (int(x), int(y)))
            centres.append((x0 + x + tw / 2, y0 + y + th / 2))
        if o > 0:
            tint = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(tint)
            for j in range(1, cols):
                bx = j * (1 - o) * tw
                d.rectangle((bx, 0, bx + o * tw, full_h), fill=(174, 191, 146, 80))
            for i in range(1, rows):
                by = i * (1 - o) * th
                d.rectangle((0, by, full_w, by + o * th), fill=(174, 191, 146, 80))
            layer = Image.alpha_composite(layer, tint)
        cv.delete("all")
        ph = self._photos["layout"] = ImageTk.PhotoImage(layer)
        cv.create_image(x0, y0, image=ph, anchor="nw")
        if len(centres) > 1:
            cv.create_line(*[v for p in centres for v in p], fill=C["order"], width=2.5, dash=(6, 5),
                           arrow="last", arrowshape=(12, 14, 5))
        rad = 13
        for (cx, cy), (index, _) in zip(centres, s.tiles):
            cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=C["order"], outline=C["well"], width=3)
            cv.create_text(cx, cy + 1, text=str(index), fill=C["well"], font=(self.F.head_family, 10))

    # ------------------------------------------------------------- stitching
    def stitch(self):
        s = self.sample
        if not self._ready() or self._apply() or self.busy:
            return
        w = getattr(self.app, "worker", None)
        if w is not None and w.is_alive():
            messagebox.showinfo("Restitch", "The Stitch tab is stitching right now. Wait for it to finish, "
                                            "then try again.", parent=self.win)
            return
        self.busy = True
        self.go_btn.configure(state="disabled", text="Stitching…")
        self.cancel_btn.configure(state="disabled")
        self._say("Reading the full-size tiles and stitching… (this can take a minute on the network drive)")
        folder, overlap, refine, keep = self.folder, self.overlap.get() / 100.0, bool(self.refine.get()), \
            bool(self.keep_old.get())
        old = self.path
        out = s.output_path(folder)

        def work():
            backup, ok, err = None, False, ""
            try:
                if keep and os.path.exists(old):
                    d = os.path.join(os.path.dirname(old), REPLACED_DIR)
                    os.makedirs(d, exist_ok=True)
                    stem, ext = os.path.splitext(os.path.basename(old))
                    backup = os.path.join(d, f"{stem} ({datetime.datetime.now():%Y-%m-%d %H%M}){ext}")
                    shutil.copy2(old, backup)
                result = {}
                core.run_native([s], folder, overlap=overlap, refine=refine,
                                log=lambda m: self.q.put(("log", m)),
                                on_done=lambda name, good: result.setdefault("ok", good))
                ok = bool(result.get("ok"))
                if ok and os.path.normcase(old) != os.path.normcase(out) and os.path.exists(old):
                    os.remove(old)          # the new image has the standard name; drop the old copy
                if not ok:
                    err = "Stitching failed – see the log (Show log on the Stitch tab)."
            except Exception as e:  # noqa: BLE001
                err = str(e)
            self.q.put(("stitched", (ok, out, backup, err)))
        threading.Thread(target=work, daemon=True).start()

    def _finished(self, ok, out, backup, err):
        self.busy = False
        self.cancel_btn.configure(state="normal")
        self.go_btn.configure(text="Stitch & replace")
        if not ok:
            self._say(err or "Stitching failed.", self.C["error"])
            self._changed()
            return
        self.app.log(f"Restitched {self.sample.name} from the Gallery -> {out}"
                     + (f" (old image kept as {backup})" if backup else ""))
        self._after_restitch(out)
        if self.on_done:
            self.on_done(self.path, out)
        kept = f" The old one is in Stitched/{REPLACED_DIR}." if backup else ""
        self._say(f"Done: {os.path.basename(out)} replaced.{kept}", self.C["sage_ink"])
        try:
            from gallery_gui import cached_thumb
            self.current = cached_thumb(out)
        except Exception:  # noqa: BLE001
            pass
        self.path = out
        self._changed()

    def _after_restitch(self, out):
        """Update file-date caches (so thumbnails refresh) and clear the sample's Redo mark."""
        import identify_core as idc
        try:
            st = os.stat(out)
            idc._file_stats[out] = core.file_stats[out] = (st.st_mtime, st.st_size)
        except OSError:
            pass
        app, name = self.app, self.sample.name
        same = os.path.normcase(os.path.normpath(app.folder.get() or ".")) == os.path.normcase(self.folder)
        if same and hasattr(app, "redo"):
            app.redo.discard(name)
            app.status.pop(name, None)
            app.thumbs.pop(out, None)
            app._save_redo()
            app._restat_output(out)
            app._refresh_table()
        else:                                  # edit that folder's saved Redo list directly
            path = os.path.join(self.folder, core.OUT_DIR_NAME, getattr(app, "REDO_FILE", "_redo.txt"))
            try:
                with open(path, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                keep = [line for line in lines if line.strip() != name]
                if len(keep) != len(lines):
                    if any(line.strip() and not line.startswith("#") for line in keep):
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("\n".join(keep) + "\n")
                    else:
                        os.remove(path)
            except OSError:
                pass

    def close(self):
        if self.busy:
            messagebox.showinfo("Restitch", "Stitching is still running; the window can close when it's done.",
                                parent=self.win)
            return
        self.gen += 1
        self.win.destroy()
