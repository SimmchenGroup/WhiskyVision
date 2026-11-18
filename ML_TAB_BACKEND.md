# ML tab: backend reference (`ml_core.py`)

Handoff for whoever builds the **ML** tab UI. The backend is finished and tested; it has no UI code.
Written 13 Nov 2026.

## What it is

Identify matches whole images by HOG + cosine similarity (nearest neighbour). `ml_core` instead:

1. **Finds the droplet by its coffee ring** (annulus matched filter, then a robust ellipse fitted to the rim).
2. **QC:** rejects images with no rim, ghosting or several droplets, or a droplet mostly outside the image.
3. **Unwraps it radially** into an r/R × angle image, so droplet size and rotation stop mattering.
4. **Measures** 26 interpretable scalar features and a 192-number radial profile (the "radial feature axis").
5. **Optionally normalises** each image (subtract background, divide by the droplet's own mean) to remove exposure and lamp colour.
6. **Trains** a classifier (standardise → PCA → multinomial logistic regression, numpy only) with cross-validation, saves and loads models, and predicts single droplets or **whole samples** (several droplets averaged, with an "unlike any trained whisky" check).

**Dependencies:** numpy + OpenCV, like `identify_core`. **matplotlib** is needed only for the `figure_*` functions (added to `requirements.txt`). Check `ml_core.figures_available()` and grey out figure buttons if it returns False.

**Reference results (research run, 18 Khalifa whiskies, 1,654 droplets):**
- scalar + radial: balanced accuracy ≈ 0.89
- 6-whisky smoke test: 0.967 for one droplet, 0.997 for 5 droplets averaged

These are *within-slip* figures (all droplets of a whisky are on one cover slip), so show them as such and don't present them as a guarantee.

## Threading rules

- **Run everything heavier than a view function in a worker thread,** as the Identify tab does: `analyse`, `features*`, `build_dataset`, `cross_validate`, `train_model`, `predict*` and every `figure_*`.
- **Progress, logging and stop:** long calls take `log(str)`, `progress(done, total)` and `should_stop() -> bool`, with the same meaning as `identify_core.build_database`. Marshal callbacks to the Tk thread yourself (the existing tabs use a queue and `after`).
- **Stop** raises `InterruptedError("stopped")`.
- **Matplotlib** runs with the Agg backend and never opens windows. Figures come back as **PIL images**: show them with `ImageTk.PhotoImage(fit_to(img, px))`, or save them with `img.save(path)`.

## Timings and caching

- **One stitched image:** about 1–3 s to analyse (reading at full size, then working at ¼ size). Network reads from J: add about 1–2 s.
- **`features(path)`** caches its result to disk in `~/.whisky_stitcher_ml/`, keyed on path + modified time + size + `VERSION`. A second call takes milliseconds. Re-stitching an image invalidates its cache automatically.
- **The first `build_dataset` on the full Khalifa set** (about 2,400 images) takes roughly 30–60 min. Later runs take seconds. Say so in the UI and show progress.
- **`cross_validate`:** under 1 s per feature set (it uses cached features). **`train_model`:** under 1 s.

## API

### Names
```python
import identify_core as idc
names = idc.load_key(idc.find_key(folder))      # {"H": "Chivas 25 YO", ...}; pass as names= to figures
```

### One droplet (the "views" for selected Gallery images)
```python
d = ml_core.analyse(path)            # -> Droplet. Full analysis with all images kept (not cached; 1-3 s)
d.label, d.name                      # whisky letter (folder above "Stitched"), image name
d.qc_ok, d.qc_reason                 # e.g. False, "several droplets or ghosting (62% of the image outside the droplet)"
d.seg is None                        # True if no rim was found at all: views fall back to the raw image
d.scalars, d.scalars_norm            # {feature name: value}, raw and normalised
d.profiles, d.profiles_norm          # {"mean_R","mean_G","mean_B","mean_L","std_L","p95_L","hue_ratio"}: arrays of N_R=128 over r/R 0..1.15
```
View functions return **uint8 RGB numpy arrays** (wrap with `Image.fromarray`). Each is cheap once `d` exists:

| Function | Shows |
|---|---|
| `view_raw(d, gain=2.5)` | Stitched image at ¼ size, brightened |
| `view_rim(d)` | Raw image + fitted rim (green = passes QC, red = fails) |
| `view_ring_map(d)` | Ring map (magma colours) + rim (cyan) + centre |
| `view_background(d)` | Normalisation step 1: background region (blue) beyond 1.25 R (dashed) |
| `view_subtracted(d)` | Step 2: background subtracted (same gain as raw, so brightness is comparable) |
| `view_normalised(d)` | Step 3: divided by the droplet's own mean |
| `view_unwrap(d, normalised=True, height=256)` | Radial unwrap; angle 0–360° left to right, **centre at the bottom**, rim at the dashed line; 720 × 256 px |
| `view_crop(d, normalised=False)` | Square droplet-centred crop |

`normalisation_numbers(d)` returns R/G/B values (0–255) for a small table under the views:
```python
{"background": [19.0, 16.0, 18.0], "droplet_mean_raw": [...], "droplet_mean_after_subtraction": [...],
 "gain": [8.15, 8.63, 12.29], "background_share_of_image": 0.187}
```

### Average of a whole sample
```python
recs, skipped = ml_core.features_many(paths, log=..., progress=..., should_stop=...)   # cached Records
s = ml_core.summarise(recs, label="H")      # ValueError if no droplet passed QC
s.n_images, s.n_used, s.rejected            # rejected = [(image name, reason)]
s.mean_droplet, s.mean_droplet_norm         # 128×128 RGB "average droplet" (rotations are random -> radial structure)
s.mean_unwrap, s.mean_unwrap_norm           # 64×180 RGB mean unwrap (row 0 = centre; flip vertically to match view_unwrap)
s.profile_mean / profile_sd (and _norm)     # dicts like Droplet.profiles, over s.r (r/R)
s.scalar_mean, s.scalar_sd                  # {feature: value}
```
Show the 128 px images enlarged, brightened ×2.5 for raw and ×1.6 for normalised (matching the figures). Option: `summarise(recs, include_failed=True)` also averages droplets that failed QC.

### Figures (matplotlib → PIL image)
| Function | Content |
|---|---|
| `figure_pipeline(droplets, names)` | One row per droplet: stitched, ring map + rim, normalised, unwrap, radial profile |
| `figure_normalisation(droplets, names)` | One row per droplet: raw, background region, background histograms with medians, subtracted, droplet means before/after with gains, normalised |
| `figure_profiles(items, names, keys=("mean_L","std_L","hue_ratio"), normalised=False)` | Overlay of radial profiles. `items` can mix `Droplet`s, `Record`s and `SampleSummary`s (drawn as mean ± SD bands), e.g. to compare samples |
| `figure_sample_average(summary, names)` | Average droplet (raw and normalised), mean unwrap, mean ± SD profiles |
| `figure_confusion(cv_result, names)` | Row-normalised confusion matrix |
| `figure_cv_comparison([cv_results])` | Balanced accuracy bars for several runs (nuisance in grey, chance line) |

All raise `ValueError` if no selected image has a droplet, and `RuntimeError` if matplotlib is missing. Rows in the per-droplet figures are one per selected image, so cap the selection (4–6 rows is readable).

### Training and evaluation
```python
ds = ml_core.build_dataset(folder, log, progress, should_stop)  # same images as Identify/Gallery
ds.qc_table()             # [(label, images, passed QC)], for a table before training
res = ml_core.cross_validate(ds, feature_set="scalar+radial", normalised=False, folds=5,
                             groups=None, min_per_class=5, progress=..., should_stop=...)
res.summary()             # one-line text
res.balanced_accuracy, res.accuracy, res.macro_f1, res.chance
res.confusion, res.classes, res.recall      # recall = {label: per-whisky recall}
res.excluded              # {reason: n droplets} (QC failures, whiskies with < min_per_class good droplets)
res.sample_accuracy(n)    # balanced accuracy when n droplets of a sample are averaged (n = 1, 3, 5…)
model = ml_core.train_model(ds, feature_set=..., normalised=..., cv_result=res)
ml_core.save_model(model, "something.wwml.npz"); model = ml_core.load_model(path)
model.classes, model.info  # info: trained date, folder, n_per_class, left_out, cv scores, unknown_threshold
```
- **Feature sets** (`ml_core.FEATURE_SETS`):
  - `"scalar"` (26 features)
  - `"radial"` (192)
  - `"scalar+radial"` (the recommended default)
  - `"nuisance"`: a control (droplet size, ring score, image size). If this scores close to the real features, the model is reading imaging artefacts, not webs. Worth a "Run control" button.
- **`normalised=True`** uses the per-image-normalised features, which ignore exposure and lamp colour. Expect a few points lower.
- **`groups`:** a function `path -> group id` (e.g. a slip or day parsed from the file name) holds out whole groups. This is the honest test once slip IDs exist. With `None` (random folds), the result means "recognises another droplet from the same slip".
- **Always pass `cv_result=res` to `train_model`.** That calibrates the sample-level "unknown" check. Without it, the check is off (`unknown_threshold` returns None).

### Prediction
```python
preds = ml_core.predict(model, paths)        # one Prediction per image
p.label, p.probability, p.top, p.margin       # top = [(label, prob)] best first
p.qc_ok, p.qc_reason, p.unknown, p.confidence # confidence: "high" | "medium" | "low (close call)" |
                                              # "possibly not a trained whisky" | "unreliable (failed QC)"
sp = ml_core.predict_sample(model, paths)     # several droplets of ONE sample, judged together
sp.verdict                                    # e.g. "H", "H, but close to K", "not like any trained whisky (closest: K)"
sp.label, sp.probability, sp.top, sp.margin, sp.n_used, sp.n_images
sp.agreement                                  # share of single droplets that agree
sp.unknown, sp.unknown_score, sp.unknown_threshold
sp.droplets                                   # the per-image Predictions
```
**UI guidance:**
- Prefer `predict_sample` whenever the user selects several images of one sample, and say how many droplets were used. In the Nov 2026 test (6 trained whiskies), an unseen whisky was flagged 53% of the time from one droplet and 92–94% from 3–5.
- A single-droplet `unknown` is weak; label it "possibly".
- A whisky closely related to a trained one (Glengoyne 15 vs a trained Glengoyne 10) can still be called as its sibling. A closed-set model always names one of its trained whiskies.

### Constants worth showing
- `ml_core.QC_MIN_SCORE` (0.12), `QC_MAX_OUTSIDE` (0.45), `QC_MIN_VALID` (0.60): the QC rules.
- `ml_core.SCALAR_NAMES` and `radial_vector_names()`: feature names, for any feature table.
- `ml_core.VERSION`: bump it if feature code changes, so caches and models aren't silently reused.

## Suggested tab layout (front end's call)
1. **Explore:** pick images (or reuse the Gallery selection). Show a view switcher (raw / rim / ring map / background / subtracted / normalised / unwrap), the QC verdict and the normalisation numbers. Buttons: "Pipeline figure", "Normalisation figure", "Profiles figure", "Save PNG".
2. **Sample average:** pick a whisky (or a set of images), then show `summarise` (average droplet, mean unwrap, profile bands and rejected list) and `figure_sample_average`. Add a "Compare samples" option using `figure_profiles` with several summaries.
3. **Train & evaluate:** reference folder (same as Identify), a QC table, feature-set and normalised options, and "Run cross-validation" (summary, `figure_confusion`, `sample_accuracy` for 1/3/5). Also "Run control (nuisance)", "Compare feature sets" (`figure_cv_comparison`), and "Train & save model".
4. **Predict:** load a model, pick images, get per-droplet results and a sample verdict. Export to CSV like Identify's `write_csv`.

## Not in the backend yet
- CNN models (EfficientNet embedding, unwrap CNN): these need PyTorch and stay in the research scripts in `ml/` for now. They reached about 0.95–0.97 vs 0.89 for features in the research run.
- Slip or day IDs: once file names carry a slip ID, pass a `groups` function to `cross_validate`.
- The launcher (`Start Stitcher.bat`) doesn't check for matplotlib, so on another PC figures fail until `pip install -r requirements.txt` is run. Add `matplotlib` to its import check if figures should always work.

## Changes 14–15 Nov 2026 (front end please check)

**Display brightness (15 Nov).** The old fixed ×2.5 gain with a hard clip blew out bright droplets: at ×2.5, 79% of images had more than 1% pure-white pixels, and M, O, V, W and Y lost their whole centre.
- **All `view_*` functions now default to `gain="auto"`.** Each image gets its own gain: the 90th percentile of its pixels is mapped to 0.7, capped at ×2.5 and never below ×1. The brightest pixels then roll off smoothly (`soft_clip`) instead of hard-clipping. Dim webs (A, K, S) still get ×2.5; bright ones get about ×1.1–1.4.
- **`view_background` and `view_subtracted` use the raw image's auto gain** (`raw_gain(d)`), so they stay comparable with `view_raw`.
- **Pass a number** to any view for a fixed gain.
- **New `ml_core.display(img, max_gain=2.5)`:** use it for any other array you show. Use `max_gain=1.6` for normalised images.
- **`ml_gui.py` still applies a fixed ×2.5 itself** in the sample-average panel (around the `show(a, gain)` helper near line 956: `show(s.mean_droplet, 2.5)` and so on), and its notes text says "brightened ×2.5". Replace both with `ml_core.display(...)` and "auto gain".

**Normalised features (14 Nov).** These are now measured on the unclipped normalised image (`norm_info["float"]`), and `in_mean_R/G/B` are dropped in normalised mode. `VERSION` is now `ml-v2`, so cached features are recomputed once. The CNN's normalised crop is tone-mapped (`tone_map`), and `embed_crops` takes a `gain` argument.

**CNN raw input stays ×2.5 with a hard clamp (decided 15 Nov).** A soft roll-off was tested on all 2,438 droplets and didn't help: CNN raw balanced accuracy went from 0.958 to 0.953, V recall from 0.93 to 0.84, and every other whisky stayed within ±0.01. So `CNN_VERSION` is back to the original, existing cached embeddings remain valid, and no re-embed is needed. Only the *display* uses the soft roll-off.

## Explanations: what the model uses on each image (17 Nov 2026)

These are exact attribution maps for the **scalar, radial and scalar + radial** models; CNN sets aren't covered. The classifier is linear in its features, so a droplet's score splits exactly into one term per feature, and each term is painted onto the pixels that feature is computed from. (Checked: the terms plus the bias equal the model's score gap.)

```python
d = ml_core.analyse(path)
e = ml_core.explain(model, d)                    # why the predicted whisky (vs the average of the others)
e = ml_core.explain(model, d, target="W", versus="Y")   # why W rather than Y (e.g. predicted vs runner-up)
e.target, e.versus, e.probability, e.top
e.groups      # [(group, contribution)], ranked by size; e.g. "rim colour (R-B)/(R+B)", "interior bright particles"
e.features    # the same at single-feature level, e.g. "hue_ratio@r0.99"
ml_core.view_explain_heat(e)                   # RGB: red = evidence FOR the target, blue = AGAINST (sqrt scale)
img, legend = ml_core.view_explain_groups(e, k=6, sign=+1)   # top 6 groups FOR, one colour each
img, legend = ml_core.view_explain_groups(e, k=6, sign=-1)   # top 6 AGAINST; legend = [(group, contribution, (r,g,b))]
ml_core.figure_explain([e1, e2, ...], names)   # rows: image | evidence map | FOR | AGAINST | ranked bars
```
- **Groups:** radial features are grouped as zone (centre < 0.35, interior 0.35–0.85, rim 0.85–1.10, outside > 1.10) × profile (red, green, blue, patchiness, bright particles, colour). Scalars are grouped by meaning (coffee ring, interior contrast, centre deposit…).
- **Units:** contributions are in logit units (log-odds), so +2.3 multiplies the odds by about 10.
- **Suggested UI:** an "Explain" button next to a prediction, with a target drop-down defaulting to the prediction and a "versus" drop-down defaulting to the average or the runner-up. Show `view_explain_heat`, plus the FOR and AGAINST overlays with their colour legend.
- **Timing:** about 1–2 s per droplet, fine in a worker thread.

### CNN maps (17 Nov 2026)
```python
cm = ml_core.cnn_map(model, path, target=None, versus=None)   # model with "cnn" in feature_set
cm.cells          # 8 × 8 signed contributions (exact: they sum to the CNN part of the score)
ml_core.view_cnn_map(cm)   # red = CNN evidence for the target, blue = against, over the 256 px crop
```
- **Coarse by nature:** the true resolution is 8 × 8 (each cell is about 1/8 of the crop). Label it "where the CNN looks", not "what".
- **Cost:** about 1–2 s (8 rotated or flipped CNN passes).
- **Pair it with `explain()`** on a scalar + radial model for the named, pixel-level view.
