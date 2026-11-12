# WildPRIMA: Metric 3D Shape and Gait of Wild Quadrupeds from Video and LiDAR

Metric 3D quadrupeds from wild video and single images: an extended SMAL body (bone scales, horns, humps) fitted per
clip with LiDAR, and a feed-forward student distilled from those fits (weights: Hugging Face, `wildprima-student-0.1.0`).

```bash
bash setup_env.sh && conda activate prima
python predict.py photo.jpg --weights <hf user>/wildprima-student --out predictions/   # single image, student
python run.py <frames folder>                                                           # per-clip fitting
python student_refine.py --split test --w-relief 10                                     # student + refinement (eval)
python baseline_animer.py                                                               # AniMer baseline (needs third_party/AniMer)
```

Code: MIT. Weights: PRIMA's academic non-commercial licence. See NOTICE.

---

# WildPRIMA

Builds a 3D model of the animals in an image, a folder of images, a video, or an RGB + LiDAR snippet, using
[PRIMA](https://github.com/AdaptiveMotorControlLab/PRIMA). The body is measured from the images with no
knowledge of the species, so one pipeline handles any quadruped, in principle. The species only decides which horns to
build.

## Setup and run

```bash
./setup_env.sh                 # creates the `prima` conda env, clones PRIMA, downloads ~6 GB of weights
conda activate prima
python app.py                  # the GUI, at http://127.0.0.1:7860
```

From the command line:

```bash
python run.py photo.jpg
python run.py photos/                    # one species per run: all animals share one body shape
python run.py 1.mp4 --species giraffe
python run.py input_30s/giraffe_4        # rgb/ + depthm/ + metadata.txt: adds sizes in metres
python run.py 1.mp4 --baseline-only      # PRIMA alone, fast
pytest tests
```

About 10-15 minutes for a 300-frame clip on an RTX PRO 4000 while other jobs share the GPU.

## GUI

Choose **Image**, **Folder of images** or **Video** and upload it, or use **Path on this machine** for data
already here, which is faster for large folders. Results appear on the page: a comparison video, a 3D model to
rotate, a turntable and downloads. Each run writes to a new `outputs/gui/<name>_<time>/`. Stop ends the job.

A folder may hold colour images only, or also 16-bit depth PNGs of the same names and a `metadata.txt`. The GUI
binds to this machine only; `--host 0.0.0.0` exposes it, with no login, to anyone who can reach the port.

## How it works

1. **Detect** (`animal3d/detection.py`): Mask R-CNN, counting every COCO mammal class as "animal", because COCO
   has no class for most of these species. PRIMA's own filter drops elephants and giraffes.
2. **PRIMA** gives each animal a starting pose and camera. It has one SMAL template, whose shape space comes from
   toy scans of cats, dogs, horses, cows and hippos, so it cannot produce every species' proportions.
3. **Species → horns** (`animal3d/species.py`): the species comes from `--species` or `metadata.txt` when it names one
   species, otherwise from [BioCLIP](https://huggingface.co/imageomics/bioclip) on the 16 largest clean views. A
   table, curated from the sources linked in it, gives the horn layout: paired (cattle, goats, antelope), on the nose
   (two for white, black and Sumatran rhinos; one for Indian and Javan), or none (camels, horses, cats; giraffe
   ossicones are not modelled). The classifier's probabilities are added up per layout, since it often cannot tell
   close species apart in small crops but the layout is what matters. An unknown or uncertain animal falls back to
   letting the masks decide. Camels also get their humps from it (two for Bactrian, one for dromedary) and small ears, held at 0.45 of SMAL's
   horse-like ears: masks barely show ears, and a fitted ear only pads the outline. The species is
   used for nothing else.
4. **Search** (`animal3d/fit.py`): tries 12 starting body plans (neck, leg and thickness) on the masks and keeps the
   best. With a known layout, the horns are always built; if they do not cut the mask error by at least 13%, they are
   a typical shape for the layout, held fixed and reported as "species prior, not measured". With no known layout,
   paired horns are tried and kept only if they clear that 13%. `--horns masks|none|paired|nasal_two|nasal_one`
   overrides. `--humps auto|0|1|2` and `--ears auto|masks|<length>` do the same for humps and ears. Animals under 40 px tall are not used to measure the body.
5. **Fit** (`AnimalBody` in `animal3d/body.py`): length and thickness per bone group, then smooth symmetric surface
   offsets, shared by all animals in the input. Torso length is held fixed, because images cannot separate
   overall size from distance. Horns are tapered tubes appended after SMAL's vertices and skinned to the head joint,
   so PRIMA's network, checkpoint and keypoints are untouched: a mirrored pair from the ears, or one or two on the
   midline of the snout, standing on its top surface. The ears have tight bone limits and
   no surface offsets, otherwise they stretch to cover a horn.
6. **Pose** every instance with that body fixed, using the real camera when it is known. With LiDAR, half of each
   animal's points (a checkerboard of 16 px blocks) also pull the mesh's depth relief towards the LiDAR's
   (`--depth-weight`, default 0.3; 0 turns it off). Outlines cannot tell which way a body turns in depth; LiDAR can.
   The body's shape is still measured from the masks alone: with LiDAR in that stage too, the shared shape absorbed
   the strain, and a rhino's neck grew to giraffe length.
7. **Scale and check with LiDAR** (`animal3d/lidar.py`): LiDAR depth over mesh depth gives metres per model unit.
   The check uses the other half of the points, which the fit never sees, beside what a flat surface would score.
   Points on one animal are not fully independent of each other, so this check is weaker than before, when LiDAR
   played no part in the fit.

### Sensor calibration

Depth is only used with a known calibration, because units differ by site (`animal3d/sources.py`). `metadata.txt`
selects it: `source: July..` is Longleat (millimetres), `source: Zimbabwe` is Bubye Valley (centimetres). For other
data, add a `calibration.json` with `fx, fy, cx, cy, width, height, metres_per_unit` to the input folder, or
pass `--calibration`.

The local docs disagree on Longleat's unit (`HANDOFF.md` says millimetres, `WILDDEPTH_REMOTE_SETUP.md` says
centimetres). Millimetres it is: values top out near 49,000 under the rig's 50 m cutoff, and they give a giraffe about
4 m tall, where centimetres would give 40 m. These cameras also have non-square pixels (fx 903, fy 1069), which
the fit accounts for.

## Outputs (`outputs/<input>/`)

| File | Contents |
|---|---|
| `comparison.mp4` | PRIMA beside the fitted body, with per-animal crops and profiles (photos: `overlays/*.jpg`) |
| `overlay.mp4` | The fitted body over the input |
| `model.glb`, `.obj`, `model_turntable.mp4` | The standalone model in its mean pose, in metres when LiDAR gives scale |
| `meshes/*.obj` | Meshes every 10th frame, in camera coordinates |
| `metrics.csv`, `summary.json`, `fit_vs_size.png` | Per-instance measurements and their summary |
| `fit.npz`, `baseline.npz`, `masks.pkl`, `detections.csv` | Fitted parameters, PRIMA's results, masks |

## Results

Medians over instances neither cut off by the frame edge nor overlapping another animal's box. "Held out" means
tracks not used to measure the body. No species information was used.

| | Giraffe (`giraffe_4`) | Ankole cow (`Ankle_cow_5`) | Giraffe (`1.mp4`, no LiDAR) |
|---|---:|---:|---:|
| Instances (clean), tracks | 924 (809), 7 | 1,291 (1,247), 18 | 636 (573), 8 |
| Starting plan chosen (neck, legs, thickness) | 2.0, 1.0, 0.7 | 1.0, 1.5, 1.0 | 2.0, 1.0, 1.0 |
| Neck/body, PRIMA → fitted | 0.27 → 0.69 | 0.24 → 0.30 | 0.27 → 0.69 |
| Leg/body, PRIMA → fitted | 0.89 → 1.16 | 0.64 → 0.80 | 0.88 → 1.18 |
| Silhouette IoU, PRIMA → fitted | 0.51 → 0.73 | 0.70 → 0.81 | 0.52 → 0.78 |
| IoU, held-out tracks | 0.48 → 0.71 | 0.73 → 0.80 | 0.53 → 0.78 |
| Instances with IoU below 0.5 | 425 → 8 | 54 → 0 | 267 → 1 |
| Mesh/mask area | 1.24 → 0.81 | 0.88 → 0.87 | 1.24 → 0.87 |
| LiDAR surface error: PRIMA → fitted (flat surface) | 0.31 → 0.37 m (0.23) | 0.16 → 0.10 m (0.12) | no LiDAR |
| Shoulder height, per well-seen animal | 2.1-2.5 m | 1.2-1.3 m | unknown |

**The body plan is found, not assumed.** The search picks a long neck for giraffes and a short one for cattle from
masks alone. On `1.mp4`, the species-blind fit matches masks better than the earlier giraffe-specific fit did (IoU 0.78
against 0.74), with a shorter neck (0.69 against 0.88). There is no ground truth to say which neck is right.

### Horns

Same clips, before → after adding horns and limiting the ears. `Ankle_cow_2` is the catalogue's Bull 2 (`July07/snippet_12`),
a single animal at about 7 m. It used that release snippet's depth maps and RealSense intrinsics, as its input folder has
only raw `.pcd` files.

| | `Ankle_cow_2` | `Ankle_cow_5` | `Rhino_1` | `giraffe_4` |
|---|---:|---:|---:|---:|
| Mask error reduction from horns | 52% | 17% | 1% | 9% |
| Horns kept | yes | yes | no | no |
| Neck/body | 0.48 → 0.22 | 0.30 → 0.32 | 0.34 → 0.34 | 0.69 → 0.71 |
| Silhouette IoU | 0.82 → 0.82 | 0.81 → 0.80 | 0.84 → 0.85 | 0.73 → 0.73 |
| LiDAR surface error (PRIMA; flat surface) | 0.31 → 0.07 m (0.12; 0.28) | 0.10 → 0.09 m (0.16; 0.12) | 0.41 → 0.44 m (0.22; 0.36) | 0.37 → 0.37 m (0.31; 0.25) |
| Fitted horn length | 0.60 m | 0.68 m | none | none |

The horn limits were set using these same clips, so they are not an independent test: the radius cap (0.07 model units)
and the 13% threshold were both chosen after a giraffe was wrongly given a large horn-like blob, which even improved its
LiDAR score. With so few clips the threshold is only provisional. A horned animal can fall below it, and short-horned
species probably will. In that case it gets no horns, not false ones.

Outputs are in `outputs/<input>_horns/`. `outputs/<input>/` holds the earlier fits.

### Rhinos, camel and species-aware horns

The species picks the horn layout (`outputs/<input>_species/`, `outputs/Camel_1_min40/`). Before → after, where "before"
is the masks-only horn search.

| | `Rhino_1` | `Rhino_3` | `Camel_1` |
|---|---:|---:|---:|
| Label; BioCLIP's answer | Rhino; black rhino 0.49 (wrong: Longleat's are white) | Rhino; white rhino 0.99 | camel; Bactrian camel 0.36 + dromedary 0.24 |
| Horns | two on the nose, typical shape (masks: -1% gain) | two on the nose, typical shape (masks: 5% gain) | none |
| IoU, PRIMA → fitted | 0.69 → 0.85 → 0.83 | 0.71 → 0.88 → 0.88 | camel's own track 0.51 → 0.72 → 0.77 |
| LiDAR surface error, fitted (PRIMA) | 0.44 → 0.43 m (0.22) | 0.13 → 0.13 m (0.14) | 0.16 → 0.15 m (0.12) |
| Animals, distance | 2 white rhinos, 18-36 m | 2 white rhinos, 18-29 m | 1 camel at 19 m + 4 specks ~10 px tall |

- **Rhino masks cut the horn off**, so the horns cannot be measured. Fitting them anyway turned them into thick stubs
  that padded the snout (front 0.17 m at the radius cap). They are now a typical white-rhino shape, front horn about
  0.8 head lengths and rear 0.4 of that, held fixed and reported as "species prior, not measured". Their sizes in metres
  only restate that prior.
- **Rhino_1's poor LiDAR score is a pose problem, not a horn or data problem.** An outline looks almost the same
  whichever way the body turns in depth. On the rhino with the most LiDAR points (track 5), the LiDAR shows the body
  turned; PRIMA's mesh leans the same way in 87% of frames, the fitted one in 30%. In Rhino_3 both lean correctly in
  every frame of its main rhino, and the fit beats PRIMA. The giraffe is near chance for both meshes (42-49%), and the
  camel's fit leans correctly in 64% of frames against PRIMA's 82%.
- **Camel_1 was measured mostly on specks.** 434 of its 640 clean instances were under 40 px tall, probably not
  camels. Animals under 40 px are now left out of the shape fit (this removes 70 small instances from `Ankle_cow_5`,
  which has not been rerun, and nothing from the other clips). The camel still has no humps, which the body cannot
  make, and walks mostly toward or away from the camera.

### LiDAR-guided pose and camel humps

Final results: `outputs/<input>_lidar/`. "Masks only" is the previous fit; "with LiDAR" adds the depth term when
posing. The LiDAR error here is on the held-out half of the points only, for both, in metres.

| | `Rhino_1` | `Rhino_3` | `Camel_1` | `giraffe_4` | `Ankle_cow_2` | `Ankle_cow_5` |
|---|---:|---:|---:|---:|---:|---:|
| LiDAR error, masks only → with LiDAR | 0.40 → **0.17** | 0.14 → 0.14 | 0.15 → **0.06** | 0.36 → **0.12** | 0.08 → **0.05** | 0.10 → **0.05** |
| PRIMA; flat surface | 0.22; 0.30 | 0.16; 0.32 | 0.12; 0.15 | 0.26; 0.23 | 0.13; 0.28 | 0.15; 0.11 |
| IoU, masks only → with LiDAR | 0.84 → 0.82 | 0.88 → 0.86 | 0.81 → 0.78 (camel) | 0.73 → 0.72 | 0.82 → 0.81 | 0.80 → 0.80 |

- **Every clip now beats PRIMA and a flat surface on LiDAR**, the giraffe and Rhino_1 for the first time. The cost is
  0.00-0.03 IoU: the masks give a little to the depth.
- **The lean is fixed.** On Rhino_1's best-seen rhino the fitted mesh leans the way the LiDAR does in 91% of frames,
  from 30% with masks alone (PRIMA 87%).
- **Sizes stay steady**: 2-14% spread per animal, except one distant rhino in `Rhino_1` at 23%, as without LiDAR. `Ankle_cow_5`'s shoulder heights fell from 1.19-1.27 m to
  1.03-1.09 m; that run also used the 40 px minimum and the species' paired horns, so the cause is not separated.
- **Humps** (`Camel_1`, Bactrian camel from BioCLIP, 0.61): the camel's own IoU rose from 0.77 to 0.81 without
  LiDAR; with LiDAR it is 0.78. See the limits below for their shape.
- **Two dead ends, kept out:** LiDAR in the shape stage too made a rhino's neck grow from 0.41 to 1.16 body lengths;
  and a first version of the depth term pushed animals along the view axis (Rhino_3's IoU fell to 0.67, the bull's
  scale swung 65% per frame) until the mesh's median depth kept its gradient. Both are covered by tests now.

### Dome humps, camel ears, and Rhino_1 with new frames

`outputs/Camel_1_domes/` replaces the displaced-vertex humps: IoU on clean instances 0.71 → 0.74, held-out LiDAR error
0.061 → 0.052 m (PRIMA 0.116, flat 0.147), with two separate humps and short ears.

`Rhino_1`'s input was replaced with 151 consecutive frames (73910-74060) of two grazing rhinos; the old run is in
`outputs/Rhino_1_lidar_old/`. More and larger views help everywhere:

| `Rhino_1` | old frames | new frames |
|---|---:|---:|
| Clean instances, tracks | 73, 5 | 280, 2 |
| Species | black rhino 0.49 (layout right, species unsure) | white rhino 0.97 |
| Silhouette IoU, PRIMA → fitted | 0.71 → 0.82 | 0.69 → 0.86 |
| Held-out LiDAR error: fitted (PRIMA; flat) | 0.17 m (0.22; 0.30) | 0.14 m (0.19; 0.35) |
| Scale spread per rhino | 4-23% | 7% |
| Shoulder height per rhino | 1.5-2.1 m | 1.19 m and 1.53 m |

The new body is heavier and lower, closer to a white rhino. The 1.19 m rhino is short for an adult white rhino
(about 1.5-1.8 m): it grazes throughout, so its standing pose is a guess, and it may also be a young animal.

### Lion_1

`input_30s/Lion_1` is 88 frames (13705-13800) of `July08/snippet_12` from release 19103277: the 640x480 RealSense
camera, like `Ankle_cow_2`. It has no `metadata.txt`, so it was run with `--species lion --calibration <json>` holding
the RealSense intrinsics from `wilddepth-v2/configs/calib/intrinsic_realsense.txt`, depth in millimetres. Output:
`outputs/Lion_1_lidar/`.

- **One animal measured the body:** a lion about 13 m away, 57 px tall, walking mostly away from the camera. The
  others are 25 px or less, under the 40 px minimum.
- **IoU 0.60 → 0.79** (the near lion 0.56 → 0.82). The body is cat-like: no horns, as the label says.
- **LiDAR: no better than PRIMA.** Held-out error 0.27 m on the near lion (PRIMA 0.26, flat 0.36), 0.10 m on the far
  lioness (PRIMA 0.09, flat 0.11). A rear view gives the depth term little body length to correct.
- **Sizes:** shoulder 0.87 m (near lion) and 1.03 m (far lioness), 6-7% spread. That is within a lioness's range
  (about 0.9-1.1 m) and short for a male (about 1.2 m); unverified.
- **BioCLIP alone would have said cattle** (0.34 against lion 0.28) on these small, low-resolution crops. The label
  decided. Without it, the horned species already add up to at least 0.46 in the top five, so paired horns could have
  passed the 0.5 bar. A known layout is always built, so the lion would then have got typical fixed horns. Label small or
  low-resolution inputs.

### What these results do not show

- **The LiDAR check is no longer fully independent.** The fit uses half of each animal's points and is checked on the
  other half, in alternate 16 px blocks. Neighbouring points on one animal are related, so this check flatters the
  fit somewhat. The masks-only fits (`--depth-weight 0`) remain the independent comparison: there, the cattle beat
  PRIMA and a flat surface, and the rhino, giraffe and camel did not, because outlines left them turned the wrong way
  in depth.
- **IoU is not independent.** It is measured against the masks the fit was optimised on, so some gain is expected.
  Held-out tracks test only the shape, as their poses are still fitted to their own masks.
- **Sizes are unverified.** There is no ground-truth measurement of these animals. Scale is steady per animal (spread
  3-13%), and the values are plausible, but a 2.1-2.5 m giraffe shoulder is only a measurement of the mean pose of
  this fit. Mask heights reach 3.4-4.3 m.

### Limits

- **Few views give poor shapes.** The same giraffes gave neck/body 0.34 from 6 frames and 0.44 from 1 frame, against 0.69
  from 300. The GUI warns below 20 views. A single photo cannot determine depth or thickness.
- **Horns.** All animals in an input share one set, so mixed herds get an average. Horn shape rests on outlines:
  masks clip the tips, and a horn in front of the body adds nothing. On `Ankle_cow_5` the fit is straight and
  sideways, and on `Ankle_cow_2` the horn base sits at its outward limit. Rhino horns are a typical shape, not a
  measurement. Horn sizes are unverified. Males-only horns (steenbok, grysbok) are given to every animal.
- **Species recognition** is reliable on large views (giraffe 1.00, Ankole 0.96, white rhino 0.99) but not on small
  ones (Rhino_1's 100 px rhinos: black rhino, white rhino and hippo were close). It only matters through the horn
  layout, and an uncertain answer falls back to the masks. The table covers the species listed in the GUI and a few
  more; other species fall back to the masks too.
- **Humps** are their own geometry: a peaked dome per hump, standing on the back's midline and skinned like the back
  beneath it, appended after SMAL's vertices like the horns. Their limits are tight on purpose: with wider ones both
  domes grew to their maximum and merged into one mass, padding the furry outline. On `Camel_1` (one camel, 19 m away)
  the front hump fits taller (0.38 m) than the rear (0.16 m); the sizes are unverified. Humps do not sag or lean.
- **Other anatomy.** There are no ossicones, and the neck is one rigid segment, so it cannot curve.
- **One body per input.** Calves and adults share a shape, and only overall size differs by animal. Mixed-species
  inputs will fit badly.
- **The standing model is a mean pose.** On grazing animals, its head is low; on walkers, its legs lean.
- **The fit runs thin on giraffes** (area 0.81), and width is weakly constrained in mostly side-on footage.
- **Distant animals are included.** In `Ankle_cow_5`, 14 of the 18 tracks are background animals 11-23 px tall (7% of
  instances). They feed the shape fit with little information; a minimum size for it would be a sensible change.
- **LiDAR coverage is partial.** Scale came from 5 of 7 giraffe tracks and 3 of the 4 large cows. One cow has no
  LiDAR points on it at all.
- **Tracks are not re-identified**, so an animal that leaves and returns gets a new ID.
- **Colours are approximate** and carry the footage's colour cast; markings blur away.

### Other species

Only giraffes and Ankole cattle have been through the full pipeline. For the rest, this is detection only: the first
snippet of each species in the local WildDepth release, 30 evenly spaced frames each, counting frames with at least one
detection. An animal may truly be absent from some of them, so these are lower bounds.

| Species | Found at 0.7 | Found at 0.5 | Most often labelled |
|---|---:|---:|---|
| Goat, camel, zebra, oryx + Ankole, Ankole cow, giraffe | 30 | 30 | sheep, sheep, giraffe, cow, sheep, giraffe |
| Donkey | 28 | 30 | horse |
| Lion | 23 | 25 | cow |
| Lioness | 23 | 30 | cow |
| Rhino | 23 | 26 | elephant |
| Tiger | 13 | 18 | zebra |
| Grysbok (closest local data to steenbok) | 12 | 16 | giraffe |
| Wolf | 5 | 6 | dog |
| Cheetah | 3 | 10 | dog |

Lower the threshold for tigers, wolves and cheetahs. If that is not enough, the fix is a detector that can be told what
to look for, such as Grounding DINO with SAM. Horse, steenbok and wildebeest have no local footage, so are untested.

## Differences from PRIMA's demos

- Any species is detected, and several animals per frame are tracked and rendered together.
- Mask R-CNN instead of Faster R-CNN, to get masks to fit to.
- torch 2.7.1 + CUDA 12.8 instead of 2.2.1, because Blackwell GPUs need it. DeepLabCut, pytorch3d, open3d and mmcv
  are left out: this path does not use them, but `demo_tta.py` and PRIMA's `app.py` do need DeepLabCut.

## License

PRIMA's weights are for **academic, non-commercial use only** under its modified MIT license; see
`third_party/PRIMA/LICENSE.md`. `assets/model-viewer.min.js` is @google/model-viewer 3.5.0 (Apache-2.0).

## Releasing the trained model

`python export_release.py --version 0.1.0` packages the trained student into `release/`, kept apart from the
experiment outputs:

- `release/huggingface/` is the model repo: `model.safetensors`, `pose_prior.safetensors`, `config.json`, a model card,
  PRIMA's licence (the weights derive from PRIMA's checkpoint), results and SHA256 checksums.
- `release/github/` is the code repo: MIT code, `predict.py`, tests, and a NOTICE.

PRIMA's backbone/checkpoint and the SMAL files are not redistributed; `setup_env.sh` fetches them. Single-image inference:
`python predict.py photo.jpg --weights release/huggingface` (or a Hugging Face repo id).
