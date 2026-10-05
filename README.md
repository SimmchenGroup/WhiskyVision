# Whisky Webs Stitcher

Joins the separate microscope images (tiles) of one round object, such as a dried whisky droplet, into a single
picture. It uses ImageJ's own stitching method, and works for any layout: 1×2, 2×1, 2×2, 3×3,
4×4, 2×3, 5×5 and so on.

No coding is needed.

---

## One-time setup

Download this folder (on GitHub: **Code → Download ZIP**) and **unzip it** somewhere, for
example your Desktop. Keep all the files together; the launcher needs the files next to it.

**Windows**

1. **Install Python** from <https://www.python.org/downloads/>.
   On the first installer screen, tick **"Add python.exe to PATH"**.
2. **Install Fiji** from <https://fiji.sc>, unzip it somewhere (for example your Desktop), and
   open it once so it can finish setting up. *(You can skip this if you only use the "Python" engine, see below.)*

**Mac**

1. **Install Python** from <https://www.python.org/downloads/macos/> (the standard installer).
   Don't rely on the `python3` that comes with macOS; it can't show windows properly.
2. **Install Fiji** from <https://fiji.sc>: drag `Fiji.app` into *Applications*, then open it once
   by **right-clicking it → Open → Open**. That tells macOS it's safe. *(Skip this if you only use the "Python" engine.)*

You need an internet connection the first time you start the stitcher. It downloads about
100 MB of add-ons once, and after that it works offline.

## Every time

1. Start the stitcher:
   - **Windows:** double-click **`Start Stitcher.bat`**.
   - **Mac:** double-click **`Start Stitcher.command`**. The first time, macOS may say it
     "cannot be opened". If so, **right-click it → Open → Open**, or on newer macOS go to
     *System Settings → Privacy & Security* and click **Open Anyway**. A Terminal window opens
     alongside the stitcher, and you can close it once the stitcher appears.

   The first time, it installs a few extra parts automatically (about a minute).
2. Click **Browse…** and pick the folder that holds your tiles.
3. Every sample in the folder is listed under **Samples**. **Click one** to see it side by side:
   - **Before:** its tiles, with numbered dots joined in the order they were taken, and the
     overlapping strips shaded green.
   - **After:** a quick, low-resolution stitch showing what the final image will look like.

   Zoom with **− / +** or the mouse wheel, and drag to move around. Both pictures zoom and move
   together; **Fit** (or a double-click) resets them.

   If the After picture looks right, you're done with this step. If not, change **rows**,
   **cols** or the order under **Options › This sample**, and both pictures update straight
   away. Changes apply to every selected sample, so you can select several (Ctrl-click or
   Shift-click) and change them all at once. Click the ☑ next to a sample to leave it out.
   The **‹** button folds the list away to give the pictures more room.
4. Press **Stitch N samples**. Each sample's status changes to *Done* as it finishes, and
   **Show log** has the details.
5. Press **Open results folder**, or switch After to **Saved** to check each result.

Results are saved in a new **`Stitched`** folder inside your tile folder, as
`<sample>_stitched.tif`. **Your original tiles are never renamed or changed.**

If you run it again on the same folder, samples that are already stitched are skipped. Untick
*"Skip already stitched"* to redo them.

## Naming your files

Tiles must end in `_` and a number. Everything before the last `_` is the sample name:

| File                         | Sample              | Tile |
|------------------------------|---------------------|------|
| `S24-1273_4X_1uL_01.tiff`    | `S24-1273_4X_1uL`   | 1    |
| `S24-1273_4X_1uL_02.tiff`    | `S24-1273_4X_1uL`   | 2    |
| `large_10_0.5uL_3.bmp`       | `large_10_0.5uL`    | 3    |
| `001_0.tif`                  | `001`               | 0    |

The numbers don't need leading zeros, and they don't have to start at 1.

## Tile order

The **Order** setting is the path the microscope took. It always starts at the top-left tile:

| Order              | Path                                             |
|--------------------|--------------------------------------------------|
| Snake by rows      | → → → then ← ← ← then → → → … (the usual one)   |
| Row by row         | → → → then → → → …                               |
| Snake by columns   | ↓ ↓ ↓ then ↑ ↑ ↑ …                               |
| Column by column   | ↓ ↓ ↓ then ↓ ↓ ↓ …                               |

For **2-tile samples** you don't need to set anything. Whether the tiles are side by side or one above
the other is detected automatically.

## Options

- **Engine**
  - *ImageJ (Fiji)*: runs ImageJ's "Grid/Collection stitching" (or "Pairwise stitching" for
    2 tiles) in the background, with linear blending. The Fiji window never opens.
  - *Python*: a built-in stitcher that works the same way (phase correlation + linear blending)
    and doesn't need Fiji.
- **Overlap**: roughly how much neighbouring tiles overlap (the green strips in Before). 20 % is typical. It's only a
  starting guess when *Fine-tune positions* is on.
- **Fine-tune positions**: leave this on. Turn it off only if the stitcher keeps
  misplacing tiles of a featureless sample. It will then use exactly the overlap you entered.

## Identify tab: which whisky made this web?

Switch to **Identify** at the top of the window to test how similar a web is to webs of known
whiskies. It uses the method from Khalifa Mohamed's *Whisky Web Identification* script: every
image is turned grey, shrunk to 128 × 128, described by its HOG features (edge directions), and
compared with every reference image by cosine similarity. The closest reference wins.

1. **Reference images:** click **Browse…** and pick a folder with one sub-folder per whisky:

   ```
   Learning_Database/
       A/   image1.tif  image2.tif …
       B/   …
   Whisky_Key.xlsx      (optional: column 1 = folder name, column 2 = whisky name)
   ```

   A spreadsheet with "key" in its name, in or next to that folder, is picked up automatically
   (first row = headings). Without one, the folder names are shown, so you can also just name
   the folders after the whiskies. **Check database** matches every reference image against all
   the others and shows how often each whisky is recognised.
2. **Test images:** **Add folder** (for example the `Stitched` folder) or **Add files**.
3. Press **Identify N images**. Click an image to see it next to its closest reference, with the
   best score for each whisky (1.000 = identical).

If a test image's name starts with the class code and `_` (`O_4.tif` is class `O`), or it's in a
folder named after the class, the **Result** column shows ✓ or ✗ and the overall percentage.
**Save results (CSV)** writes everything to a spreadsheet.

## Troubleshooting

| Problem | What to do |
|---|---|
| "No tiles found" | Check the file names end in `_<number>` (see above). |
| A sample shows *Fix layout* | Set rows × cols so there's a place for every tile. |
| A sample shows *Redo* | Its old result file is broken. Press Stitch to make it again. |
| Stitched image is scrambled | Wrong order or rows/cols. Change them until the After quick preview looks right. |
| Tiles slightly misaligned | Set **Overlap** closer to the real overlap, or try the other engine. |
| Mac: "Start Stitcher.command" does nothing / "permission denied" | Open *Terminal*, type `chmod +x ` (with a space), drag the file into the window, and press Return. Then double-click it again. |
| Mac: macOS asks whether Fiji may access Desktop/Documents/Downloads | Click **Allow**. Fiji needs to read your tiles and save the results. |
| "Fiji wasn't found" | Click **Change** next to Fiji and choose `fiji-windows-x64.exe` (or `ImageJ-win64.exe`) inside the Fiji folder. On a Mac, choose `Fiji.app`. |
| *Failed* | Click **Show log** for the reason. One failed sample doesn't stop the others. |

---

## For developers

| File | Purpose |
|---|---|
| `Start Stitcher.bat` / `Start Stitcher.command` | Windows / macOS launchers. The Windows one installs packages with `pip --user`. The Mac one makes a private `.venv` inside the folder, because Homebrew/macOS Python refuse user installs. |
| `stitch_gui.py` | The window (CustomTkinter). Layout "1c Before / after" from the Claude Design mockups, in the Organic style. |
| `fonts/` | Caprasimo and Figtree (SIL Open Font License, see the `OFL-*.txt` files). They're loaded privately when the app starts (GDI on Windows, CoreText on macOS), so nothing is installed on the computer. |
| `stitcher_core.py` | Scanning and grouping files, grid layout, ImageJ macro generation, and running Fiji headless. |
| `identify_gui.py` | The Identify tab, built the first time it's opened. |
| `identify_core.py` | HOG + cosine-similarity matching, adapted from *Whisky Web Identification.py*. It shows the whisky of the **matched** reference (the original script looked up the test file's own prefix, so it printed the expected answer). scikit-learn/pandas are replaced by the equivalent numpy/openpyxl code, and the scores are identical. Command line: `python identify_core.py <reference folder> <test images or folder> [--key Whisky_Key.xlsx]` |
| `stitch_native.py` | Pure-Python engine. Can also be used from the command line: `python stitch_native.py <folder> [--rows R --cols C --order "Row by row"]` |
| `legacy/` | The earlier scripts (`prepare.py`, `stitch.py`, `rename.py`), which generated a macro to run by hand in Fiji. |

How the ImageJ engine works: for 3+ tiles it writes a `TileConfiguration` with approximate
positions to `Stitched/_layouts/`. It then runs `Grid/Collection stitching` with
`type=[Positions from file]` and `compute_overlap`, so no files need renaming and any grid
size works. Fiji runs as `fiji-windows-x64.exe --headless -macro …`. The macro reports progress by
appending to a text file, because the new Fiji launcher's console output can't be captured.
