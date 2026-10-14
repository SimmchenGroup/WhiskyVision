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

Click a picture to open it full size in your usual image viewer. Type in
the filter box to find a whisky or file name, and use Small / Medium /
Large to change the picture size.

REMOVING A BAD IMAGE

Point at a picture and click its ✕ (or right-click it) to remove it from
the dataset. It's moved to a "_removed" folder inside the reference
folder, not deleted, so you can put it back. If the tiles it was stitched
from can be found (the folder above its "Stitched" folder, or the folder
open on the Stitch tab), that sample is marked Redo on the Stitch tab so
you can stitch it again."""


# --------------------------------------------------------------------------- #
#  Finding images
# --------------------------------------------------------------------------- #
def collect(folder):
    """([(whisky code, [image paths])], [codes with only unstitched tiles]).
    The same images the Identify tab uses as references."""
    return idc.scan_classes(folder)


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


def stitch_source(path, extra_folders=()):
    """(tile folder, sample name) of the tiles a "<sample>_stitched" image was made from, or None.
    Looks next to the image's "Stitched" folder, beside the image, then in `extra_folders`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if not stem.lower().endswith("_stitched"):
        return None
    sample = stem[:-len("_stitched")]
    parent = os.path.dirname(path)
    folders = [os.path.dirname(parent)] if os.path.basename(parent) == core.OUT_DIR_NAME else []
    folders += [parent] + [f for f in extra_folders if f]
    for folder in folders:
        try:
            samples, _ = core.scan_folder(folder)
        except OSError:
            continue
        if any(s.name == sample for s in samples):
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
        self.summary = ctk.CTkLabel(head, text="", font=F.body(15), text_color=C["soft"], anchor="w")
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
        app._link(head, "Change folder", self.change_folder, 14).grid(row=0, column=5)
        app._link(head, "Refresh", lambda: self.refresh(force=True), 14).grid(row=0, column=6)

        # -- navigation pane
        nav = ctk.CTkFrame(self.body, fg_color=C["surface"], corner_radius=28, width=300)
        nav.grid(row=1, column=0, sticky="nsw", padx=(0, 16))
        nav.grid_propagate(False)
        nav.grid_rowconfigure(1, weight=1)
        nav.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(nav, text="Whiskies", font=F.semi(15), text_color=C["text"], anchor="w").grid(
            row=0, column=0, sticky="ew", padx=20, pady=(16, 6))
        self.nav_list = ctk.CTkScrollableFrame(nav, fg_color=C["surface"], corner_radius=0,
                                               scrollbar_button_color=C["line"],
                                               scrollbar_button_hover_color=C["muted"])
        self.nav_list.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 16))
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
        return dict(gallery_size=self.size.get())

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

    def refresh(self, force=False):
        folder = self.folder()
        if not force and folder == self.shown_folder:
            self._schedule_layout(0)
            return
        self.shown_folder = folder
        self.generation += 1
        self.queued.clear()
        if force:
            self.thumbs.clear()
            self.photos.clear()
        self.groups, self.unstitched = [], []
        if folder and os.path.isdir(folder):
            try:
                self.groups, self.unstitched = collect(folder)
            except OSError as e:
                self.summary.configure(text=f"Couldn't read the folder: {e}")
        self._schedule_layout(0)

    # ------------------------------------------------------------- loading
    def _loader(self):
        """Background thread: make thumbnails one at a time, in display order."""
        while True:
            gen, path = self.jobs.get()
            if gen != self.generation or path in self.thumbs:
                continue
            try:
                im = cached_thumb(path)
            except Exception:
                im = None
            self.app.events.put(("gal_thumb", (gen, path, im)))

    def handle(self, kind, data):
        if kind == "gal_thumb":
            gen, path, im = data
            self.thumbs[path] = im
            self.queued.discard(path)
            if gen == self.generation:
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
        for row, (code, name, n) in enumerate([("", "All whiskies", total)] + list(sig)):
            label = self.identify.whisky_full(code) if code else name
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

    def _visible_groups(self):
        q = self.search.get().strip().lower()
        out = []
        for code, paths in self.groups:
            name = self.identify.whisky(code)
            if not q or q in code.lower() or q in name.lower():
                out.append((code, name, paths))
            else:
                hits = [p for p in paths if q in os.path.basename(p).lower()]
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

        if not folder:
            self.summary.configure(text="")
            self.note.configure(text="")
            return message("No folder chosen yet.\n\nClick \"Change folder\" and choose the folder with "
                           "one sub-folder per whisky (the same one the Identify tab uses).")
        if not os.path.isdir(folder):
            self.summary.configure(text="")
            return message(f"This folder doesn't exist:\n{folder}")
        groups = self._visible_groups()
        n = sum(len(p) for _, _, p in groups)
        total = sum(len(p) for _, p in self.groups)
        text = (f"{n} of {total} images" if n != total else f"{total} images") \
            + f" · {len(groups)} whisk{'y' if len(groups) == 1 else 'ies'}"
        if self.unstitched:
            text += f" · not stitched yet: {', '.join(self.unstitched)}"
        self.summary.configure(text=text)
        shown = folder if len(folder) <= 90 else "…" + folder[-89:]
        self.note.configure(text=f"{shown}   ·   Click a picture to open it full size, ✕ or right-click "
                                 f"to remove it from the dataset.")
        if not groups:
            return message("No images here." if not self.groups else "Nothing matches the filter.")
        self._build_nav(groups)

        cell_w = px + GAP
        cols = max(1, int((W - MARGIN) // cell_w))
        y = 6
        order = []
        for code, name, paths in groups:
            self.group_y.append((y, code))
            cv.create_text(MARGIN / 2, y + 18, anchor="w", text=name, fill=C["text"],
                           font=(self.F.head_family, 16))
            sub = f"{self.identify.code(code)} · {len(paths)} image{'s' if len(paths) != 1 else ''}"
            cv.create_text(MARGIN / 2 + tkfont_width(self.F.head_family, 16, name, cv) + 14, y + 19,
                           anchor="w", text=sub, fill=C["muted"], font=(self.F.body_family, 11))
            y += HEAD_H
            for i, path in enumerate(paths):
                r, c = divmod(i, cols)
                x0 = MARGIN / 2 + c * cell_w
                y0 = y + r * (px + LABEL_H + GAP)
                tag = f"p{len(order)}"
                cv.create_rectangle(x0, y0, x0 + px, y0 + px, fill=C["well"], outline=C["well"],
                                    width=2, tags=(tag, tag + "box"))
                self.items[path] = (cv.create_image(x0 + px / 2, y0 + px / 2, tags=(tag,)), tag)
                label = os.path.splitext(os.path.basename(path))[0]
                cv.create_text(x0 + px / 2, y0 + px + 13, text=_ellipsize(label, px, 7),
                               fill=C["soft"], font=(self.F.body_family, 10), tags=(tag,))
                # remove button, shown while the mouse is over the picture
                bx, by = x0 + px - X_R - 6, y0 + X_R + 6
                cv.create_oval(bx - X_R, by - X_R, bx + X_R, by + X_R, fill=C["text"], outline=C["bg"],
                               width=1.5, state="hidden", tags=(tag + "x",))
                cv.create_text(bx, by, text="✕", fill=C["bg"], font=(self.F.body_family, 10),
                               state="hidden", tags=(tag + "x",))
                self.tiles.append((x0, y0, x0 + px, y0 + px + LABEL_H, path, tag, bx, by))
                order.append(path)
                self._place_thumb(path)
            rows = -(-len(paths) // cols)
            y += rows * (px + LABEL_H + GAP) + 8
        self.content_h = y + 10
        cv.configure(scrollregion=(0, 0, W, self.content_h))
        gen = self.generation
        for path in order:                     # queue missing thumbnails, top first
            if path not in self.thumbs and path not in self.queued:
                self.queued.add(path)
                self.jobs.put((gen, path))

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
        px = SIZES.get(self.size.get(), 190)
        ph = self.photos.get(path)
        if ph is None:
            t = im.copy()
            if max(t.size) > px:
                t.thumbnail((px, px), Image.LANCZOS)
            else:
                scale = px / max(t.size)
                t = t.resize((max(1, int(t.width * scale)), max(1, int(t.height * scale))), Image.BILINEAR)
            ph = self.photos[path] = ImageTk.PhotoImage(t)
        cv.itemconfigure(item, image=ph)
        cv.tag_raise(tag + "x")

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
        cv = self.canvas
        if self.hover:
            cv.itemconfigure(self.hover + "box", outline=self.C["well"])
            cv.itemconfigure(self.hover + "x", state="hidden")
        if tag:
            cv.itemconfigure(tag + "box", outline=self.C["accent"])
            cv.itemconfigure(tag + "x", state="normal")
        cv.configure(cursor="hand2" if tag else "")
        self.hover = tag

    def _on_motion(self, e):
        t, _ = self._tile_at(e)
        self._set_hover(t[5] if t else None)

    def _on_click(self, e):
        t, on_x = self._tile_at(e)
        if t:
            self.remove(t[4]) if on_x else self._open(t[4])

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

    # ----------------------------------------------------------- remove
    def remove(self, path):
        """Move an image out of the dataset and queue its sample to be stitched again."""
        root = self.folder()
        if not path or not os.path.exists(path):
            return
        source = stitch_source(path, [self.app.folder.get()])
        text = (f"Remove {os.path.basename(path)} from the dataset?\n\n"
                f"It's moved to the \"{REMOVED_DIR}\" folder inside the reference folder, not deleted, "
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
        self.shown_folder = None              # re-read the folder
        self.refresh()
        self.identify._scan_database(quiet=True)
        if source:
            if self.app.mark_redo(*source):
                self.app.status_label.configure(text=f"\"{source[1]}\" is marked Redo – press Stitch to redo it")
            else:
                messagebox.showinfo(self.app.title(), "Stitching is running, so the sample couldn't be "
                                                      f"marked Redo yet. Open {source[0]} on the Stitch tab "
                                                      "and stitch it again.")


def tkfont_width(family, size, text, widget):
    import tkinter.font as tkfont
    return tkfont.Font(root=widget, family=family, size=size).measure(text)


def _ellipsize(text, px, char_px):
    n = max(4, int(px // char_px))
    return text if len(text) <= n else text[:n - 1] + "…"
