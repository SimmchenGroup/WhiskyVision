"""
split_dialog.py — the "Split sample" window of the Stitch tab.

For when tiles of two (or more) objects were saved under one sample name.
Every tile gets a group number; group 1 keeps the sample's name, and each
other group becomes a new sample numbered after the highest in the list
(stitcher_core.plan_split / split_sample do the renaming).
"""
import tkinter as tk

import customtkinter as ctk
from PIL import Image

import stitcher_core as core

THUMB = 150
COLS = 4
MAX_GROUPS = 12


class SplitDialog(ctk.CTkToplevel):
    def __init__(self, app, C, sample, on_split):
        super().__init__(app, fg_color=C["bg"])
        self.app, self.C, self.F, self.sample, self.on_split = app, C, app.F, sample, on_split
        self.groups = [1] * sample.n
        self.n_groups = tk.IntVar(value=2)
        self.cards, self.segs, self._imgs = [], [], []
        self.palette = [C["accent"], C["sage"], "#5b7fa6", "#9a5ba6", C["accent_deep"], "#b8952e",
                        "#3f8f8a", "#c25b7a", C["sage_ink"], "#6b6bb8", C["muted"], "#8a5a3c"]
        self.title(f"Split {sample.name}")
        self.geometry("860x720")
        self.transient(app)
        self._build()
        self._load_next(0)
        self.after(50, self._grab)

    def _grab(self):
        self.lift()
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _build(self):
        C, F = self.C, self.F
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(self, text=f"Split “{self.sample.name}”", font=F.head(22), text_color=C["accent"],
                     anchor="w").grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 2))
        ctk.CTkLabel(self, text=f"Give each tile a group. Group 1 keeps the name “{self.sample.name}”; "
                                "every other group becomes a new sample, numbered after the highest "
                                "sample in the list. Tiles keep their order and are renumbered "
                                "within each group. The files are renamed (listed in "
                                f"{core.SPLIT_LOG} in the tile folder).",
                     font=F.body(13), text_color=C["soft"], anchor="w", justify="left",
                     wraplength=800).grid(row=1, column=0, sticky="new", padx=22, pady=(0, 10))

        # -- tiles
        self.grid_rowconfigure(2, weight=1)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=22, pady=(10, 0))
        ctk.CTkLabel(bar, text="Groups", font=F.semi(14), text_color=C["text"]).pack(side="left")
        for txt, d in (("−", -1), ("+", 1)):
            ctk.CTkButton(bar, text=txt, width=32, height=32, corner_radius=16, fg_color=C["surface"],
                          hover_color=C["hover_tint"], text_color=C["text"], font=F.semi(16),
                          command=lambda d=d: self._change_groups(d)).pack(side="left", padx=(8, 0))
            if d == -1:
                self.n_label = ctk.CTkLabel(bar, text="2", width=28, font=F.semi(15), text_color=C["text"])
                self.n_label.pack(side="left", padx=(8, 0))
        self.app._link(bar, "One tile per group", self._each_own, 13).pack(side="left", padx=(16, 0))
        self.app._link(bar, "Reset", self._reset, 13).pack(side="left")

        self.list = ctk.CTkScrollableFrame(self, fg_color=C["surface"], corner_radius=24,
                                           scrollbar_button_color=C["line"],
                                           scrollbar_button_hover_color=C["muted"])
        self.list.grid(row=2, column=0, sticky="nsew", padx=22, pady=(0, 0))
        for c in range(COLS):
            self.list.grid_columnconfigure(c, weight=1)
        for i, (index, path) in enumerate(self.sample.tiles):
            card = ctk.CTkFrame(self.list, fg_color=C["bg"], corner_radius=16, border_width=3,
                                border_color=self.palette[0])
            card.grid(row=i // COLS, column=i % COLS, padx=6, pady=6, sticky="n")
            img = ctk.CTkLabel(card, text="…", width=THUMB, height=THUMB, fg_color=C["well"],
                               corner_radius=10, text_color=C["well_text"], font=F.body(11))
            img.grid(row=0, column=0, padx=8, pady=(8, 4))
            img.bind("<Button-1>", lambda e, i=i: self._cycle(i))
            ctk.CTkLabel(card, text=f"tile {index}", font=F.body(12), text_color=C["soft"]).grid(
                row=1, column=0)
            seg = ctk.CTkSegmentedButton(card, values=["1", "2"], font=F.body(12), height=26,
                                         fg_color=C["surface"], selected_color=self.palette[0],
                                         selected_hover_color=self.palette[0],
                                         unselected_color=C["surface"],
                                         unselected_hover_color=C["hover_tint"], text_color=C["text"],
                                         command=lambda v, i=i: self._set(i, int(v)))
            seg.set("1")
            seg.grid(row=2, column=0, padx=8, pady=(2, 10))
            self.cards.append((card, img))
            self.segs.append(seg)

        # -- bottom
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=4, column=0, sticky="ew", padx=22, pady=(10, 18))
        foot.grid_columnconfigure(0, weight=1)
        self.summary = ctk.CTkLabel(foot, text="", font=F.body(13), text_color=C["text"], anchor="w",
                                    justify="left", wraplength=560)
        self.summary.grid(row=0, column=0, sticky="w")
        self.app._pill_button(foot, "Cancel", self.destroy, height=40, size=14, width=100).grid(
            row=0, column=1, padx=(10, 8))
        self.ok = ctk.CTkButton(foot, text="Split", command=self._split, height=40, width=120,
                                corner_radius=20, fg_color=C["accent"], hover_color=C["accent_hover"],
                                text_color=C["bg"], font=F.head(14))
        self.ok.grid(row=0, column=2)
        self._update()

    # ------------------------------------------------------------- loading
    def _load_next(self, i):
        """Decode one thumbnail per idle step so the window opens straight away."""
        if i >= self.sample.n or not self.winfo_exists():
            return
        im = self.app._thumb(self.sample.tiles[i][1], 400)
        _, label = self.cards[i]
        if im is None:
            label.configure(text="Couldn't open")
        else:
            t = im.copy()
            t.thumbnail((THUMB, THUMB), Image.LANCZOS)
            ci = ctk.CTkImage(light_image=t, dark_image=t, size=t.size)
            self._imgs.append(ci)
            label.configure(image=ci, text="")
        self.after(1, lambda: self._load_next(i + 1))

    # ------------------------------------------------------------- groups
    def _change_groups(self, d):
        n = max(2, min(MAX_GROUPS, self.n_groups.get() + d, max(2, self.sample.n)))
        self.n_groups.set(n)
        self.groups = [min(g, n) for g in self.groups]
        self._update()

    def _set(self, i, g):
        self.groups[i] = g
        self._update()

    def _cycle(self, i):
        """Clicking a picture moves it to the next group."""
        self.groups[i] = self.groups[i] % self.n_groups.get() + 1
        self._update()

    def _each_own(self):
        n = min(MAX_GROUPS, self.sample.n)
        self.n_groups.set(max(2, n))
        self.groups = [min(i + 1, n) for i in range(self.sample.n)]
        self._update()

    def _reset(self):
        self.n_groups.set(2)
        self.groups = [1] * self.sample.n
        self._update()

    def _update(self):
        n = self.n_groups.get()
        self.n_label.configure(text=str(n))
        values = [str(k) for k in range(1, n + 1)]
        for i, seg in enumerate(self.segs):
            g = self.groups[i]
            colour = self.palette[(g - 1) % len(self.palette)]
            if seg.cget("values") != values:
                seg.configure(values=values)
            seg.configure(selected_color=colour, selected_hover_color=colour)
            seg.set(str(g))
            self.cards[i][0].configure(border_color=colour)
        plan, err = core.plan_split(self.sample, self.groups, self.app.samples.keys())
        if err:
            self.summary.configure(text=err, text_color=self.C["muted"])
            self.ok.configure(state="disabled")
            return
        parts = [f"{g} → {name} ({len(m)} tile{'s' if len(m) != 1 else ''})" for g, name, m in plan]
        if 1 not in {g for g, _, _ in plan}:
            parts.append(f"(“{self.sample.name}” will have no tiles left)")
        self.summary.configure(text="   ·   ".join(parts), text_color=self.C["text"])
        self.ok.configure(state="normal")
        self.plan = plan

    def _split(self):
        plan, err = core.plan_split(self.sample, self.groups, self.app.samples.keys())
        if err:
            return
        self.grab_release()
        self.destroy()
        self.on_split(self.sample, plan)
