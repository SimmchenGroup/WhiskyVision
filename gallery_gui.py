"""
gallery_gui.py — the "Gallery" tab of the Whisky Webs app.

Shows every stitched sample in the Identify tab's reference folder as a
thumbnail, grouped by whisky (one sub-folder per whisky). Click a thumbnail to
open the full image in the computer's own image viewer.

Shows the same images the Identify tab uses as references (identify_core.scan_classes):
in each whisky's folder, files named "*_stitched.*" or kept in a "Stitched"
folder. Raw tiles are never shown; folders with only tiles are listed as
"not stitched yet".

A navigation pane jumps to each whisky. The ✕ on a thumbnail (or right-click)
moves the image to <reference folder>/_removed and, when its tiles can be
found, marks that sample "Redo" on the Stitch tab.

Thumbnails are cached in ~/.whisky_stitcher_thumbs, so big stitched TIFFs are
only decoded once.
"""
import bisect
import hashlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

import collections_core as cc
import identify_core as idc
import stitcher_core as core

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".whisky_stitcher_thumbs")
CACHE_EDGE = 320                               # stored thumbnail size (px)
SIZES = {"Small": 130, "Medium": 190, "Large": 280}
GAP, LABEL_H, HEAD_H, MARGIN = 14, 26, 46, 18
X_R = 12                                       # radius of the remove button
REMOVED_DIR = "_removed"

HELP_TEXT = """\
GALLERY

Shows every stitched image in the Identify tab's reference folder (one
sub-folder per whisky), grouped by whisky. "Change folder" changes it for
both tabs.

Only stitched images are shown (the same ones the Identify tab
compares against): files named "…_stitched" or kept in a "Stitched"
folder. Raw tiles are never shown, and a whisky folder with no stitched
images yet is listed as "not stitched yet" at the top.

Click a whisky in the list on the left to jump to it.

COLLECTIONS

The menu next to "Gallery" switches between image collections:
  Khalifa samples            the reference folder, one folder per whisky
  SWRI · Normal (DFK)        grouped by product
  SWRI · Different years     grouped by product (all years together)
  SWRI · Counterfeit         grouped by sample number (there's no key)
  SWRI · all genuine         Normal + Different years, by product
SWRI captions show the sample number and droplet volume (and the year where
it matters). Use the volume / year buttons above the list to narrow it down.
Point at a picture to see its product, category, year and source folder at
the bottom; "file says …" means a typo in the file name was corrected. Only
DFK images are used (Thor and Old are left out).

Click a picture to open it full size in your usual image viewer. Type in
the filter box to find a whisky or file name, and use Small / Medium /
Large to change the picture size.

RESTITCHING AN IMAGE

Right-click a picture and choose "Restitch…". The window finds the tiles it
was stitched from (the folder above its "Stitched" folder, or the folder open
on the Stitch tab) and shows the tile layout, the current image and a quick
preview of the new stitch side by side. Change rows, columns, order, overlap
or refine until the preview looks right, then press "Stitch & replace": it's
stitched again at full size with the Python engine. The old image is kept in
Stitched/_replaced unless you untick that, and the sample's Redo mark (if
any) is cleared.

REMOVING A BAD IMAGE

Point at a picture and click its ✕ (or right-click it) to remove it from
the dataset. It's moved to a "_removed" folder inside the reference
folder, not deleted, so you can put it back. If the tiles it was stitched
from can be found (the folder above its "Stitched" folder, or the folder
open on the Stitch tab), that sample is marked Redo on the Stitch tab so
you can stitch it again.

REMOVING MANY AT ONCE

Click "Select" (top right). Click pictures to tick them, Shift-click to
tick a range, or press "Select all" to tick everything shown (type in the
filter box first to select all of one whisky). Press "Remove selected" to
move them all to "_removed" in one go. Press "Done" to leave Select mode."""


# --------------------------------------------------------------------------- #
#  Finding images
# --------------------------------------------------------------------------- #
def collect(folder, fresh=False):
    """([(whisky code, [image paths])], [codes with only unstitched tiles]).
    The same images the Identify tab uses as references (and the same cached scan)."""
    return idc.scan_classes(folder, fresh=fresh)


def cached_thumb(path):
    """A small PIL copy of the image, from the disk cache when possible."""
    if path in idc._file_stats:              # noted while scanning (fast on network drives)
        mtime, size = idc._file_stats[path]
    else:
        st = os.stat(path)
        mtime, size = st.st_mtime, st.st_size
    key = hashlib.sha1(f"{path}|{mtime}|{size}|{CACHE_EDGE}".encode("utf-8")).hexdigest()
    cache = os.path.join(CACHE_DIR, key + ".jpg")
    if os.path.exists(cache):
        try:
            with Image.open(cache) as im:
                return im.convert("RGB")
        except OSError:
            pass
    im = Image.fromarray(core.read_rgb(path))
    im.thumbnail((CACHE_EDGE, CACHE_EDGE), Image.LANCZOS)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        im.save(cache, quality=88)
    except OSError:
        pass
    return im


def fit_to(im, px):
    """A copy of `im` scaled so its longer side is `px`."""
    t = im.copy()
    if max(t.size) > px:
        t.thumbnail((px, px), Image.LANCZOS)
    else:
        scale = px / max(t.size)
        t = t.resize((max(1, int(t.width * scale)), max(1, int(t.height * scale))), Image.BILINEAR)
    return t


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])




def show_in_folder(path):
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])


def stitch_source(path, extra_folders=(), cache=None):
    """(tile folder, sample name) of the tiles a "<sample>_stitched" image was made from, or None.
    Looks next to the image's "Stitched" folder, beside the image, then in `extra_folders`.
    `cache` (a dict) keeps each folder's sample names between calls."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if not stem.lower().endswith("_stitched"):
        return None
    sample = stem[:-len("_stitched")]
    parent = os.path.dirname(path)
    folders = [os.path.dirname(parent)] if os.path.basename(parent) == core.OUT_DIR_NAME else []
    folders += [parent] + [f for f in extra_folders if f]
    for folder in folders:
        if cache is not None and folder in cache:
            names = cache[folder]
        else:
            try:
                names = {s.name for s in core.scan_folder(folder)[0]}
            except OSError:
                names = set()
            if cache is not None:
                cache[folder] = names
        if sample in names:
            return folder, sample
    return None


def move_to_removed(path, root):
    """Move `path` to <root>/_removed/<same relative path>; returns the new path.
    Folders starting with "_" are ignored by the Gallery and Identify tabs."""
    dest = os.path.join(root, REMOVED_DIR, os.path.relpath(path, root))
    base, ext = os.path.splitext(dest)
    n = 2
    while os.path.exists(dest):
        dest = f"{base} ({n}){ext}"
        n += 1
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(path, dest)
    return dest


# --------------------------------------------------------------------------- #
#  Tab
# --------------------------------------------------------------------------- #
class GalleryTab:
    def __init__(self, app, C, identify):
        self.app, self.C, self.F, self.identify = app, C, app.F, identify
        self.groups = []             # [(code, [paths])]
        self.unstitched = []         # codes whose folders only hold raw tiles
        self.thumbs = {}             # path -> PIL image, or None if unreadable
        self.photos = {}             # path -> PhotoImage at the current size
        self.items = {}              # path -> (canvas image item, tag)
        self.tiles = []              # [(x0, y0, x1, y1, path, tag)] for hit-testing
        self.group_y = []            # [(y, code)] where each whisky's section starts
        self.content_h = 1
        self.hover = None            # tag of the tile under the mouse
        self.nav_buttons = {}        # code -> CTkButton ("" = all)
        self.nav_sig = None
        self.nav_active = None
        self._jumped = 0.0
        self.jobs = queue.Queue()
        self.queued = set()          # paths waiting for the loader
        self.generation = 0          # bumped on refresh, so stale jobs are dropped
        self.shown_folder = None
        self.size = tk.StringVar(value=app.settings.get("gallery_size", "Medium"))
        self._layout_job = None
        self._last_width = 0
        self.tile_by_tag, self.tile_y, self._tile_bottoms = {}, {}, []
        self.drawn = set()           # tags of tiles drawn on the canvas so far
        self.px = SIZES["Medium"]
        self.view = (0, 0)           # canvas y range being drawn (read by the loader)
        self._render_job = None
        self.select_mode = False
        self.selected = set()        # paths ticked in Select mode
        self._anchor = None          # last ticked path, for Shift-click ranges
        # collections (collections_core): Khalifa = the Identify tab's folder; SWRI by product
        self.collection = app.settings.get("gallery_collection", "khalifa")
        self.coll_options = []       # [(id, title, path)]
        self.names = {}              # SWRI label -> display name
        self.info = {}               # SWRI path -> cc.image_info
        self.captions = {}           # SWRI path -> caption under the thumbnail
        self.loading = False
        self.volume = tk.StringVar(value="All")
        self.year = tk.StringVar(value="All")
        self._note_text = ""
        self._build()
        threading.Thread(target=self._loader, daemon=True).start()

    def _build(self):
        C, F, app = self.C, self.F, self.app
        self.body = ctk.CTkFrame(app, fg_color=C["bg"], corner_radius=0)
        self.body.grid_rowconfigure(1, weight=1)
        self.body.grid_columnconfigure(1, weight=1)

        head = ctk.CTkFrame(self.body, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        head.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(head, text="Gallery", font=F.head(30), text_color=C["accent"]).grid(
            row=0, column=0, padx=(4, 14))
        left = ctk.CTkFrame(head, fg_color="transparent")
        left.grid(row=0, column=1, columnspan=2, sticky="ew")
        left.grid_columnconfigure(1, weight=1)
        self.coll_menu = ctk.CTkOptionMenu(
            left, values=["Khalifa samples"], command=self._on_collection, width=250, height=34,
            corner_radius=17, font=F.semi(13), dropdown_font=F.body(13), fg_color=C["surface"],
            button_color=C["surface"], button_hover_color=C["hover_tint"], text_color=C["text"],
            dropdown_fg_color=C["bg"], dropdown_text_color=C["text"], dropdown_hover_color=C["hover_tint"])
        self.coll_menu.grid(row=0, column=0, padx=(0, 14))
        self.summary = ctk.CTkLabel(left, text="", font=F.body(15), text_color=C["soft"], anchor="w")
        self.summary.grid(row=0, column=1, sticky="w")
        self.search = search = ctk.CTkEntry(head, width=220, height=34, corner_radius=17,
                                            border_width=1, border_color=C["line"], fg_color=C["bg"],
                                            text_color=C["text"], font=F.body(13),
                                            placeholder_text="Filter by name…")
        search.grid(row=0, column=3, padx=(12, 12))
        search.bind("<KeyRelease>", lambda e: self._schedule_layout(150))
        ctk.CTkSegmentedButton(head, values=list(SIZES), variable=self.size,
                               command=lambda v: self._schedule_layout(0), font=F.body(13), height=34,
                               corner_radius=17, fg_color=C["surface"], selected_color=C["accent"],
                               selected_hover_color=C["accent_hover"], unselected_color=C["surface"],
                               unselected_hover_color=C["hover_tint"], text_color=C["text"]).grid(
            row=0, column=4, padx=(0, 6))
        self.select_link = app._link(head, "Select", self.toggle_select, 14)
        self.select_link.grid(row=0, column=5)
        app._link(head, "Change folder", self.change_folder, 14).grid(row=0, column=6)
        app._link(head, "Refresh", lambda: self.refresh(force=True), 14).grid(row=0, column=7)

        # -- selection bar, shown while Select is on
        self.sel_bar = bar = ctk.CTkFrame(head, fg_color=C["accent_tint"], corner_radius=20)
        bar.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.sel_label = ctk.CTkLabel(bar, text="", font=F.semi(14), text_color=C["accent_ink"], anchor="w")
        self.sel_label.grid(row=0, column=0, sticky="w", padx=18, pady=6)
        app._link(bar, "Select all", self.select_all, 14).grid(row=0, column=1)
        app._link(bar, "Clear", self.select_none, 14).grid(row=0, column=2)
        self.sel_remove = ctk.CTkButton(bar, text="Remove selected", command=self.remove_selected,
                                        height=30, corner_radius=15, fg_color=C["error"],
                                        hover_color="#8f1e18", text_color="#ffffff", font=F.semi(14))
        self.sel_remove.grid(row=0, column=3, padx=(8, 6), pady=6)
        app._link(bar, "Done", self.toggle_select, 14).grid(row=0, column=4, padx=(0, 10))
        bar.grid_remove()

        # -- navigation pane
        nav = ctk.CTkFrame(self.body, fg_color=C["surface"], corner_radius=28, width=300)
        nav.grid(row=1, column=0, sticky="nsw", padx=(0, 16))
        nav.grid_propagate(False)
        nav.grid_rowconfigure(2, weight=1)
        nav.grid_columnconfigure(0, weight=1)
        self.nav_title = ctk.CTkLabel(nav, text="Whiskies", font=F.semi(15), text_color=C["text"], anchor="w")
        self.nav_title.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 6))
        # volume / year filters, SWRI collections only
        self.filters = filt = ctk.CTkFrame(nav, fg_color="transparent")
        filt.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        seg = dict(font=F.body(12), height=28, corner_radius=14, fg_color=C["bg"], selected_color=C["accent"],
                   selected_hover_color=C["accent_hover"], unselected_color=C["bg"],
                   unselected_hover_color=C["hover_tint"], text_color=C["text"],
                   command=lambda v: self._schedule_layout(0))
        self.vol_seg = ctk.CTkSegmentedButton(filt, values=["All", "0.5 µL", "1 µL"], variable=self.volume, **seg)
        self.vol_seg.grid(row=0, column=0, sticky="w")
        self.year_seg = ctk.CTkSegmentedButton(filt, values=["All"], variable=self.year, **seg)
        self.year_seg.grid(row=1, column=0, sticky="w", pady=(6, 0))
        filt.grid_remove()
        self.nav_list = ctk.CTkScrollableFrame(nav, fg_color=C["surface"], corner_radius=0,
                                               scrollbar_button_color=C["line"],
                                               scrollbar_button_hover_color=C["muted"])
        self.nav_list.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=(0, 16))
        self.nav_list.grid_columnconfigure(0, weight=1)

        # -- thumbnails
        card = ctk.CTkFrame(self.body, fg_color=C["surface"], corner_radius=28)
        card.grid(row=1, column=1, sticky="nsew")
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)
        self.canvas = cv = tk.Canvas(card, background=C["surface"], highlightthickness=0)
        cv.grid(row=0, column=0, sticky="nsew", padx=(14, 0), pady=14)
        self.scrollbar = ctk.CTkScrollbar(card, command=cv.yview, fg_color=C["surface"],
                                          button_color=C["line"], button_hover_color=C["muted"])
        self.scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=20)
        cv.configure(yscrollcommand=self._on_scroll)
        cv.bind("<Configure>", self._on_resize)
        cv.bind("<MouseWheel>", self._on_wheel)
        cv.bind("<Button-4>", lambda e: cv.yview_scroll(-3, "units"))   # Linux
        cv.bind("<Button-5>", lambda e: cv.yview_scroll(3, "units"))
        cv.bind("<Motion>", self._on_motion)
        cv.bind("<Leave>", lambda e: self._set_hover(None))
        cv.bind("<Button-1>", self._on_click)
        cv.bind("<Button-3>", self._on_menu)
        if sys.platform == "darwin":          # Mac right-click / Control-click
            cv.bind("<Button-2>", self._on_menu)
            cv.bind("<Control-Button-1>", self._on_menu)

        self.menu = tk.Menu(app, tearoff=0)
        self.menu_path = None
        self.menu.add_command(label="Open", command=lambda: self._open(self.menu_path))
        self.menu.add_command(label="Show in folder", command=lambda: show_in_folder(self.menu_path))
        self.menu.add_command(label="Restitch…", command=lambda: self.restitch(self.menu_path))
        self.menu.add_separator()
        self.menu.add_command(label="Remove from dataset…", command=lambda: self.remove(self.menu_path))

        self.note = ctk.CTkLabel(self.body, text="", font=F.body(12), text_color=C["muted"], anchor="w")
        self.note.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))

    def show(self):
        self.body.grid(row=1, column=0, sticky="nsew", padx=28, pady=20)
        self.refresh()

    def hide(self):
        self.body.grid_remove()

    def settings(self):
        return dict(gallery_size=self.size.get(), gallery_collection=self.collection)

    def log(self, msg):
        self.app.logs.append(msg)

    # ------------------------------------------------------------- folder
    def folder(self):
        return self.identify.db_folder.get().strip()

    def change_folder(self):
        before = self.folder()
        self.identify.browse_database()        # same folder as the Identify tab
        if self.folder() != before:
            self.refresh(force=True)

    def swri(self):
        return self.collection != "khalifa"

    def collection_root(self):
        """The folder the current collection lives in (also where "_removed" goes)."""
        if not self.swri():
            return self.folder()
        return next((p for c, _, p in self.coll_options if c == self.collection),
                    cc.collection_path(self.collection, self.folder()))

    def _on_collection(self, title):
        cid = next((c for c, t, _ in self.coll_options if t == title), "khalifa")
        if cid == self.collection:
            return
        self.collection = cid
        self.volume.set("All")
        self.year.set("All")
        if self.search.get():
            self.search.delete(0, "end")
            if hasattr(self.search, "_activate_placeholder"):   # CTk drops the hint after delete()
                self.search._activate_placeholder()
        if self.select_mode:
            self.toggle_select()
        self.app._save_settings()
        self.refresh()

    def refresh(self, force=False):
        """Scan the current collection in the background (SWRI scans take seconds on J:)."""
        folder = self.folder()
        key = (self.collection, folder)
        if not force and key == self.shown_folder:
            self._schedule_layout(0)
            return
        self.shown_folder = key
        self.generation += 1
        gen, cid = self.generation, self.collection
        self.queued.clear()
        if force:
            self.thumbs.clear()
            self.photos.clear()
        self.groups, self.unstitched = [], []
        self.loading = bool(folder and os.path.isdir(folder))
        self._schedule_layout(0)
        if not self.loading:
            return

        def work():
            try:
                opts = cc.available_collections(folder) if os.path.isfile(cc.SWRI_KEY) else []
                if cid == "khalifa" or cid not in [c for c, _, _ in opts]:
                    groups, unstitched = collect(folder, fresh=force)
                    names, info, caps, cid_used = {}, {}, {}, "khalifa"
                else:
                    groups, unstitched = cc.scan_collection(cid, folder, fresh=force)
                    names = cc.collection_names(cid, folder)
                    info = {p: cc.image_info(p) for _, ps in groups for p in ps}
                    caps = {p: cc.image_caption(p) for p in info}
                    cid_used = cid
                self.app.events.put(("gal_groups", (gen, cid_used, opts, groups, unstitched, names, info, caps,
                                                    None)))
            except Exception as e:  # noqa: BLE001
                self.app.events.put(("gal_groups", (gen, cid, [], [], [], {}, {}, {}, str(e))))
        threading.Thread(target=work, daemon=True).start()

    def _set_options(self, opts):
        """Fill the collection menu. The Khalifa entry is always the Identify tab's folder."""
        folder = self.folder()
        base = os.path.basename(os.path.normpath(folder)) if folder else ""
        k_title = "Khalifa samples" if base.lower() == "khalifa samples" or not base else f"Reference: {base}"
        if not os.path.isfile(cc.SWRI_KEY):     # SWRI_Key.csv isn't in the repo: no SWRI collections
            opts = []
        self.coll_options = [("khalifa", k_title, folder)] + [o for o in opts if o[0] != "khalifa"]
        self.coll_menu.configure(values=[t for _, t, _ in self.coll_options])
        self.coll_menu.set(next(t for c, t, _ in self.coll_options if c == self.collection))

    # ------------------------------------------------------------- loading
    def _loader(self):
        """Background thread: make thumbnails one at a time, in display order, already
        resized for display. Pictures scrolled far out of view are skipped (they're
        queued again when they come back into view)."""
        while True:
            gen, path, px = self.jobs.get()
            if gen != self.generation or path in self.thumbs:
                self.queued.discard(path)
                continue
            y, (lo, hi) = self.tile_y.get(path), self.view
            if y is None or not lo - 600 <= y <= hi + 600:
                self.queued.discard(path)
                continue
            try:
                im = cached_thumb(path)
                disp = fit_to(im, px)
            except Exception:
                im = disp = None
            self.app.events.put(("gal_thumb", (gen, path, im, disp, px)))

    def handle(self, kind, data):
        if kind == "gal_groups":
            gen, cid, opts, groups, unstitched, names, info, caps, err = data
            if gen != self.generation:
                return
            self.loading = False
            if err:
                self.summary.configure(text=f"Couldn't read the folder: {err}")
                self._schedule_layout(0)
                return
            self.collection = cid
            self._set_options(opts)
            self.groups, self.unstitched = groups, unstitched
            self.names, self.info, self.captions = names, info, caps
            noun = {"swri_counterfeit": "Samples"}.get(cid, "Products" if self.swri() else "Whiskies")
            self.nav_title.configure(text=noun)
            if self.swri():
                years = sorted({i.get("year", "") for i in info.values()} - {""})
                self.year_seg.configure(values=["All"] + years)
                if self.year.get() not in ["All"] + years:
                    self.year.set("All")
                if len(years) > 1:
                    self.year_seg.grid()
                else:
                    self.year_seg.grid_remove()
                self.filters.grid()
            else:
                self.filters.grid_remove()
            self._schedule_layout(0)
            return
        if kind == "gal_thumb":
            gen, path, im, disp, px = data
            self.thumbs[path] = im
            self.queued.discard(path)
            if gen == self.generation:
                if disp is not None and px == self.px and path not in self.photos:
                    self.photos[path] = ImageTk.PhotoImage(disp)
                self._place_thumb(path)

    # ---------------------------------------------------------- navigation
    def _build_nav(self, groups):
        """One button per whisky; rebuilt only when the list changes."""
        sig = tuple((code, name, len(paths)) for code, name, paths in groups)
        if sig == self.nav_sig:
            return
        self.nav_sig = sig
        for w in self.nav_list.winfo_children():
            w.destroy()
        self.nav_buttons, self.nav_active = {}, None
        total = sum(n for _, _, n in sig)
        everything = "All" if self.swri() else "All whiskies"
        for row, (code, name, n) in enumerate([("", everything, total)] + list(sig)):
            label = _ellipsize(self._nav_label(code) if code else name, 236, 7.4)
            b = ctk.CTkButton(self.nav_list, text=label, anchor="w", height=34, corner_radius=17,
                              fg_color="transparent", hover_color=self.C["hover_tint"],
                              text_color=self.C["text"], font=self.F.body(13),
                              command=lambda c=code: self.jump_to(c))
            b.grid(row=row, column=0, sticky="ew", pady=1)
            ctk.CTkLabel(self.nav_list, text=str(n), font=self.F.body(12), text_color=self.C["muted"],
                         width=34).grid(row=row, column=1, padx=(4, 6))
            self.nav_buttons[code] = b

    def _set_nav_active(self, code):
        if code == self.nav_active:
            return
        for c, b in self.nav_buttons.items():
            on = c == code
            b.configure(fg_color=self.C["accent_tint"] if on else "transparent",
                        text_color=self.C["accent_ink"] if on else self.C["text"])
        self.nav_active = code

    def jump_to(self, code):
        """Scroll so that whisky's section is at the top ("" = back to the top)."""
        y = next((gy for gy, c in self.group_y if c == code), 0) if code else 0
        self._jumped = time.monotonic()     # keep the clicked whisky highlighted
        self.canvas.yview_moveto(y / max(1, self.content_h))
        self._set_nav_active(code)

    def _on_scroll(self, first, last):
        """Scrollbar update; highlight the whisky whose section is at the top."""
        self.scrollbar.set(first, last)
        self._schedule_render()
        if not self.group_y or time.monotonic() - self._jumped < 0.4:
            return
        top = float(first) * self.content_h
        if top <= 0 and len(self.group_y) > 1:
            current = ""
        else:
            current = self.group_y[0][1]
            for gy, code in self.group_y:
                if gy <= top + HEAD_H:
                    current = code
        self._set_nav_active(current)

    # ------------------------------------------------------------- layout
    def _on_resize(self, e):
        if abs(e.width - self._last_width) > 4:
            self._schedule_layout(80)

    def _on_wheel(self, e):
        step = -int(e.delta / 120) if abs(e.delta) >= 120 else (-1 if e.delta > 0 else 1)
        self.canvas.yview_scroll(step * 2, "units")

    def _schedule_layout(self, delay=100):
        if self._layout_job:
            self.app.after_cancel(self._layout_job)

        def go():
            self._layout_job = None
            self._layout()
        self._layout_job = self.app.after(delay, go)

    # ------------------------------------------------------------- names
    def _name(self, code):
        """Section heading: whisky name (Khalifa) or product / counterfeit sample (SWRI)."""
        if not self.swri():
            return self.identify.whisky(code)
        return cc.label_name(code, self.names)

    def _nav_label(self, code):
        return self._name(code) if self.swri() else self.identify.whisky_full(code)

    def _section_note(self, code, paths):
        n = f"{len(paths)} image{'s' if len(paths) != 1 else ''}"
        if not self.swri():
            return f"{self.identify.code(code)} · {n}"
        inf = [self.info.get(p, {}) for p in paths]
        bits = [inf[0].get("category", "")] if inf and self.collection != "swri_counterfeit" else []
        nums = {i.get("sample_number") for i in inf} - {None}
        if self.collection != "swri_counterfeit" and nums:
            bits.append(f"{len(nums)} sample{'s' if len(nums) != 1 else ''}")
        return " · ".join(b for b in bits + [n] if b)

    def caption(self, path):
        if self.swri():
            return self.captions.get(path) or os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    def _passes_filters(self, path):
        if not self.swri():
            return True
        i = self.info.get(path, {})
        v, y = self.volume.get(), self.year.get()
        return (v == "All" or i.get("volume") == v) and (y == "All" or i.get("year") == y)

    def _visible_groups(self):
        q = self.search.get().strip().lower()
        out = []
        for code, paths in self.groups:
            paths = [p for p in paths if self._passes_filters(p)]
            if not paths:
                continue
            name = self._name(code)
            if not q or q in code.lower() or q in name.lower():
                out.append((code, name, paths))
            else:
                hits = [p for p in paths if q in os.path.basename(p).lower() or q in self.caption(p).lower()]
                if hits:
                    out.append((code, name, hits))
        return out

    def _layout(self):
        C, cv = self.C, self.canvas
        cv.delete("all")
        self.items.clear()
        self.tiles, self.group_y, self.hover = [], [], None
        W = max(cv.winfo_width(), 200)
        self._last_width = W
        px = SIZES.get(self.size.get(), 190)
        if any(p.width() != px and p.height() != px for p in self.photos.values()):
            self.photos.clear()
        folder = self.folder()

        def message(text):
            self._build_nav([])
            cv.create_text(W / 2, 120, text=text, fill=C["muted"], width=W - 80,
                           font=(self.F.body_family, 13), justify="center")
            self.content_h = 240
            cv.configure(scrollregion=(0, 0, W, 240))
            self._update_sel_bar()

        if not folder:
            self.summary.configure(text="")
            self.note.configure(text="")
            return message("No folder chosen yet.\n\nClick \"Change folder\" and choose the folder with "
                           "one sub-folder per whisky (the same one the Identify tab uses).")
        if not os.path.isdir(folder):
            self.summary.configure(text="")
            return message(f"This folder doesn't exist:\n{folder}")
        if self.loading:
            self.summary.configure(text="Loading…")
            return message("Finding the stitched images…" + ("\n\nThe first SWRI scan takes a few seconds."
                                                            if self.swri() else ""))
        groups = self._visible_groups()
        n = sum(len(p) for _, _, p in groups)
        total = sum(len(p) for _, p in self.groups)
        if self.collection == "swri_counterfeit":
            noun = ("sample", "samples")
        elif self.swri():
            noun = ("product", "products")
        else:
            noun = ("whisky", "whiskies")
        text = (f"{n} of {total} images" if n != total else f"{total} images") \
            + f" · {len(groups)} {noun[0] if len(groups) == 1 else noun[1]}"
        if self.unstitched:
            shown_u = self.unstitched[:6]
            text += f" · not stitched yet: {', '.join(shown_u)}" + (" …" if len(self.unstitched) > 6 else "")
        self.summary.configure(text=text)
        root = self.collection_root()
        shown = root if len(root) <= 90 else "…" + root[-89:]
        if self.select_mode:
            self._note_text = (f"{shown}   ·   Click pictures to select them (Shift-click for a "
                               f"range), then press Remove selected.")
        else:
            self._note_text = (f"{shown}   ·   Click a picture to open it full size, ✕ or right-click "
                               f"to remove it from the dataset."
                               + ("  Point at one for its details." if self.swri() else ""))
        self.note.configure(text=self._note_text)
        if not groups:
            return message("No images here." if not self.groups else "Nothing matches the filter.")
        self._build_nav(groups)

        cell_w = px + GAP
        cols = max(1, int((W - MARGIN) // cell_w))
        y = 6
        for code, name, paths in groups:
            self.group_y.append((y, code))
            cv.create_text(MARGIN / 2, y + 18, anchor="w", text=name, fill=C["text"],
                           font=(self.F.head_family, 16))
            sub = self._section_note(code, paths)
            cv.create_text(MARGIN / 2 + tkfont_width(self.F.head_family, 16, name, cv) + 14, y + 19,
                           anchor="w", text=sub, fill=C["muted"], font=(self.F.body_family, 11))
            y += HEAD_H
            for i, path in enumerate(paths):
                r, c = divmod(i, cols)
                x0 = MARGIN / 2 + c * cell_w
                y0 = y + r * (px + LABEL_H + GAP)
                tag = f"p{len(self.tiles)}"
                bx, by = x0 + px - X_R - 6, y0 + X_R + 6      # remove button (shown on hover)
                self.tiles.append((x0, y0, x0 + px, y0 + px + LABEL_H, path, tag, bx, by))
            rows = -(-len(paths) // cols)
            y += rows * (px + LABEL_H + GAP) + 8
        # Only the pictures near the visible part are drawn (see _render_visible):
        # thousands of canvas items make every scroll and redraw slow.
        self.tile_by_tag = {t[5]: t for t in self.tiles}
        self.tile_y = {t[4]: t[1] for t in self.tiles}
        self._tile_bottoms = [t[3] for t in self.tiles]
        self.drawn = set()
        self.px = px
        self.content_h = y + 10
        cv.configure(scrollregion=(0, 0, W, self.content_h))
        self._update_sel_bar()
        self._render_visible()

    def _schedule_render(self):
        if not self._render_job:
            def go():
                self._render_job = None
                self._render_visible()
            self._render_job = self.app.after_idle(go)

    def _render_visible(self):
        """Draw the pictures in (and a screen either side of) the visible area,
        and queue their thumbnails, top first."""
        if not self.tiles:
            return
        cv = self.canvas
        H = max(cv.winfo_height(), 300)
        top = cv.canvasy(0)
        lo, hi = top - H, top + 2 * H
        self.view = (lo, hi)
        gen, want = self.generation, []
        i = bisect.bisect_left(self._tile_bottoms, lo)
        while i < len(self.tiles) and self.tiles[i][1] <= hi:
            t = self.tiles[i]
            if t[5] not in self.drawn:
                self._draw_tile(t)
            if t[4] not in self.thumbs and t[4] not in self.queued:
                want.append(t[4])
            i += 1
        for path in want:
            self.queued.add(path)
            self.jobs.put((gen, path, self.px))

    def _draw_tile(self, t):
        C, cv, px = self.C, self.canvas, self.px
        x0, y0, _, _, path, tag, _, _ = t
        cv.create_rectangle(x0, y0, x0 + px, y0 + px, fill=C["well"], outline=C["well"],
                            width=2, tags=(tag, tag + "box"))
        self.items[path] = (cv.create_image(x0 + px / 2, y0 + px / 2, tags=(tag,)), tag)
        label = self.caption(path)
        cv.create_text(x0 + px / 2, y0 + px + 13, text=_ellipsize(label, px, 7),
                       fill=C["soft"], font=(self.F.body_family, 10), tags=(tag,))
        if self.select_mode:                  # selection tick
            cx, cy = x0 + X_R + 6, y0 + X_R + 6
            cv.create_oval(cx - X_R, cy - X_R, cx + X_R, cy + X_R, fill=C["bg"], outline=C["bg"],
                           width=1.5, tags=(tag + "ck", tag + "ckdot"))
            cv.create_text(cx, cy, text="✓", fill="#ffffff", font=(self.F.body_family, 11, "bold"),
                           state="hidden", tags=(tag + "ckmark",))
        self.drawn.add(tag)
        self._place_thumb(path)
        self._paint_tag(tag, path in self.selected)

    def _place_thumb(self, path):
        entry = self.items.get(path)
        if entry is None:
            return
        item, tag = entry
        im = self.thumbs.get(path, False)
        cv = self.canvas
        if im is False:
            cv.itemconfigure(item, image="")
            return
        if im is None:
            x, y = cv.coords(item)
            cv.create_text(x, y, text="Couldn't open", fill=self.C["order"],
                           font=(self.F.body_family, 10), tags=(tag,))
            return
        ph = self.photos.get(path)
        if ph is None:
            ph = self.photos[path] = ImageTk.PhotoImage(fit_to(im, self.px))
        cv.itemconfigure(item, image=ph)

    # ----------------------------------------------------------- mouse
    def _tile_at(self, e):
        x, y = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        for t in self.tiles:
            if t[0] <= x <= t[2] and t[1] <= y <= t[3]:
                return t, (x - t[6]) ** 2 + (y - t[7]) ** 2 <= (X_R + 3) ** 2
        return None, False

    def _set_hover(self, tag):
        if tag == self.hover:
            return
        cv, C = self.canvas, self.C
        old, self.hover = self.hover, tag
        cv.delete("hoverx")
        if old:
            self._paint_tag(old)
        if tag:
            cv.itemconfigure(tag + "box", outline=C["accent"])
            t = self.tile_by_tag.get(tag)
            if t and not self.select_mode:    # remove button, only on the picture under the mouse
                bx, by = t[6], t[7]
                cv.create_oval(bx - X_R, by - X_R, bx + X_R, by + X_R, fill=C["text"], outline=C["bg"],
                               width=1.5, tags=("hoverx",))
                cv.create_text(bx, by, text="✕", fill=C["bg"], font=(self.F.body_family, 10),
                               tags=("hoverx",))
        cv.configure(cursor="hand2" if tag else "")
        if self.swri():                       # details of the picture under the mouse
            t = self.tile_by_tag.get(tag) if tag else None
            self.note.configure(text=self._details(t[4]) if t else self._note_text,
                                text_color=C["text"] if t else C["muted"])

    def _details(self, path):
        i = self.info.get(path) or cc.image_info(path)
        num = i.get("sample_number", "")
        if i.get("written_as") and i.get("written_as") != num:
            num += f" (file says {i['written_as']})"
        bits = [num, i.get("product", ""), i.get("category", ""), i.get("volume", ""), i.get("year", ""),
                f"folder {i.get('source_folder', '')}"]
        note = i.get("note", "")
        if note and "file says" not in note:
            bits.append(note)
        return "   ·   ".join(b for b in bits if b) + f"   ·   {os.path.basename(path)}"

    def _on_motion(self, e):
        t, _ = self._tile_at(e)
        self._set_hover(t[5] if t else None)

    def _on_click(self, e):
        t, on_x = self._tile_at(e)
        if not t:
            return
        if self.select_mode:
            self._toggle(t[4], shift=bool(e.state & 0x0001))
        elif on_x:
            self.remove(t[4])
        else:
            self._open(t[4])

    # ----------------------------------------------------------- select
    def toggle_select(self):
        """Turn Select mode on or off (off clears the selection)."""
        self.select_mode = not self.select_mode
        self.selected.clear()
        self._anchor = None
        self.select_link.configure(text="Done" if self.select_mode else "Select")
        if self.select_mode:
            self.sel_bar.grid()
        else:
            self.sel_bar.grid_remove()
        self._schedule_layout(0)

    def _visible_paths(self):
        return [t[4] for t in self.tiles]

    def _toggle(self, path, shift=False):
        paths = self._visible_paths()
        if shift and self._anchor in paths:
            a, b = sorted((paths.index(self._anchor), paths.index(path)))
            for p in paths[a:b + 1]:
                self.selected.add(p)
                self._paint_selected(p)
        else:
            if path in self.selected:
                self.selected.discard(path)
            else:
                self.selected.add(path)
            self._paint_selected(path)
        self._anchor = path
        self._update_sel_bar()

    def select_all(self):
        """Tick every picture currently shown (respects the filter)."""
        for p in self._visible_paths():
            self.selected.add(p)
            self._paint_selected(p)
        self._update_sel_bar()

    def select_none(self):
        paths = list(self.selected)
        self.selected.clear()
        for p in paths:
            self._paint_selected(p)
        self._update_sel_bar()

    def _paint_selected(self, path):
        entry = self.items.get(path)
        if entry:
            self._paint_tag(entry[1], path in self.selected)

    def _paint_tag(self, tag, on=None):
        """Draw a tile's outline and tick for its selected state."""
        cv, C = self.canvas, self.C
        if tag not in self.drawn:
            return
        if on is None:
            t = self.tile_by_tag.get(tag)
            on = t is not None and t[4] in self.selected
        hovered = tag == self.hover
        cv.itemconfigure(tag + "box", outline=C["accent"] if (on or hovered) else C["well"],
                         width=4 if on else 2)
        if self.select_mode:
            cv.itemconfigure(tag + "ckdot", state="normal", fill=C["accent"] if on else C["bg"],
                             outline="#ffffff" if on else C["muted"])
            cv.itemconfigure(tag + "ckmark", state="normal" if on else "hidden")
        else:
            cv.itemconfigure(tag + "ckdot", state="hidden")
            cv.itemconfigure(tag + "ckmark", state="hidden")

    def _update_sel_bar(self):
        if not self.select_mode:
            return
        shown = set(self._visible_paths())
        self.selected &= set(p for _, paths in self.groups for p in paths)   # drop vanished files
        n = len(self.selected)
        hidden = len(self.selected - shown)
        text = f"{n} selected" if n else "Click pictures to select them"
        if hidden:
            text += f" ({hidden} hidden by the filter)"
        self.sel_label.configure(text=text)
        self.sel_remove.configure(text=f"Remove selected ({n})" if n else "Remove selected",
                                  state="normal" if n else "disabled")

    def _on_menu(self, e):
        t, _ = self._tile_at(e)
        if t:
            self.menu_path = t[4]
            try:
                self.menu.tk_popup(e.x_root, e.y_root)
            finally:
                self.menu.grab_release()

    def _open(self, path):
        try:
            open_file(path)
        except OSError as e:
            messagebox.showerror(self.app.title(), f"Couldn't open the image:\n{e}")

    # ----------------------------------------------------------- restitch
    def restitch(self, path):
        """Open the Restitch window for one image (finds its tiles, stitches it again)."""
        if not path or not os.path.exists(path):
            return
        from restitch_dialog import RestitchDialog
        RestitchDialog(self.app, self.C, path, on_done=self._restitched)

    def _restitched(self, old, new):
        """A picture was stitched again: show the new image (same place in the list)."""
        for p in {old, new}:
            self.thumbs.pop(p, None)
            self.photos.pop(p, None)
            self.queued.discard(p)
        if os.path.normcase(old) != os.path.normcase(new):
            self.groups = [(c, [new if p == old else p for p in ps]) for c, ps in self.groups]
            if old in self.info:
                self.info[new] = self.info.pop(old)
                self.captions[new] = self.captions.pop(old, self.caption(new))
        self._schedule_layout(0)

    # ----------------------------------------------------------- remove
    def remove(self, path):
        """Move an image out of the dataset and queue its sample to be stitched again."""
        root = self.collection_root()
        if not path or not os.path.exists(path):
            return
        source = stitch_source(path, [self.app.folder.get()])
        text = (f"Remove {os.path.basename(path)} from the dataset?\n\n"
                f"It's moved to the \"{REMOVED_DIR}\" folder inside {os.path.basename(root)}, not deleted, "
                f"so you can put it back. The Gallery and Identify tabs ignore that folder.")
        if source:
            text += (f"\n\nIts tiles were found, so sample \"{source[1]}\" will be marked Redo on the "
                     f"Stitch tab, ready to stitch again.")
        else:
            text += "\n\n(Its original tiles weren't found, so nothing is queued for re-stitching.)"
        if not messagebox.askyesno(self.app.title(), text, icon="warning"):
            return
        try:
            dest = move_to_removed(path, root)
        except OSError as e:
            messagebox.showerror(self.app.title(), f"Couldn't move the image:\n{e}")
            return
        self.log(f"Removed {os.path.relpath(path, root)} from the dataset (moved to {dest})")
        self.thumbs.pop(path, None)
        self.photos.pop(path, None)
        self._forget([path])
        if source:
            if self.app.mark_redo(*source):
                self.app.status_label.configure(text=f"\"{source[1]}\" is marked Redo – press Stitch to redo it")
            else:
                messagebox.showinfo(self.app.title(), "Stitching is running, so the sample couldn't be "
                                                      f"marked Redo yet. Open {source[0]} on the Stitch tab "
                                                      "and stitch it again.")

    def _forget(self, paths):
        """Images were moved out: update the cached scans instead of re-reading the folders."""
        if self.swri():
            gone = set(paths)
            self.groups = [(c, [p for p in ps if p not in gone]) for c, ps in self.groups]
            self.groups = [(c, ps) for c, ps in self.groups if ps]
            with cc._lock:
                cc._scan_cache.clear()           # every SWRI collection may hold these images
            self._schedule_layout(0)
            return
        idc.forget_images(self.folder(), paths)
        self.shown_folder = None
        self.refresh()
        self.identify._scan_database(quiet=True)

    def remove_selected(self):
        """Move every selected image out of the dataset in one go."""
        root = self.collection_root()
        paths = [p for p in self.selected if os.path.exists(p)]
        if not paths:
            return
        n = len(paths)
        hidden = len(set(paths) - set(self._visible_paths()))
        text = (f"Remove {n} image{'s' if n != 1 else ''} from the dataset?\n\n"
                f"They're moved to the \"{REMOVED_DIR}\" folder inside {os.path.basename(root)}, not deleted, "
                f"so you can put them back. The Gallery and Identify tabs ignore that folder.")
        if hidden:
            text += f"\n\n{hidden} of them are hidden by the current filter."
        if not messagebox.askyesno(self.app.title(), text, icon="warning"):
            return
        self.app.configure(cursor="watch")
        self.app.update_idletasks()
        cache, sources, failed, moved_paths = {}, [], [], []
        extra = [self.app.folder.get()]
        try:
            for path in paths:
                source = stitch_source(path, extra, cache)
                try:
                    move_to_removed(path, root)
                except OSError as e:
                    failed.append(f"{os.path.basename(path)}: {e}")
                    continue
                self.thumbs.pop(path, None)
                self.photos.pop(path, None)
                moved_paths.append(path)
                if source:
                    sources.append(source)
        finally:
            self.app.configure(cursor="")
        moved = n - len(failed)
        self.log(f"Removed {moved} image{'s' if moved != 1 else ''} from the dataset "
                 f"(moved to {os.path.join(root, REMOVED_DIR)})")
        self.selected.clear()
        self._anchor = None
        self._forget(moved_paths)

        # Mark Redo for the tiles that were found. The Stitch tab shows one folder at a time,
        # so use the one already open there if it has any, otherwise the most common one.
        msg = []
        if sources:
            folders = [os.path.normcase(os.path.normpath(f)) for f, _ in sources]
            current = os.path.normcase(os.path.normpath(self.app.folder.get() or "."))
            target = current if current in folders else max(set(folders), key=folders.count)
            names = [s for (f, s), nf in zip(sources, folders) if nf == target]
            marked = [s for s in names if self.app.mark_redo(sources[folders.index(target)][0], s)]
            if marked:
                self.app.status_label.configure(
                    text=f"{len(marked)} sample{'s' if len(marked) != 1 else ''} marked Redo – "
                         f"press Stitch to redo {'them' if len(marked) != 1 else 'it'}")
            elif names:
                msg.append("Stitching is running, so no samples could be marked Redo yet.")
            other = len(sources) - len(names)
            if other:
                msg.append(f"{other} other sample{'s are' if other != 1 else ' is'} in different tile "
                           f"folders; open those on the Stitch tab to re-stitch them.")
        if failed:
            msg.insert(0, f"{len(failed)} image{'s' if len(failed) != 1 else ''} couldn't be moved:\n"
                       + "\n".join(failed[:10]) + ("\n…" if len(failed) > 10 else ""))
        if msg:
            messagebox.showinfo(self.app.title(), "\n\n".join(msg))


def tkfont_width(family, size, text, widget):
    import tkinter.font as tkfont
    return tkfont.Font(root=widget, family=family, size=size).measure(text)


def _ellipsize(text, px, char_px):
    n = max(4, int(px // char_px))
    return text if len(text) <= n else text[:n - 1] + "…"
