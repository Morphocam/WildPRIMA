# Giraffe 3D with PRIMA

Recovers a 3D mesh for every giraffe in a video using
[PRIMA](https://github.com/AdaptiveMotorControlLab/PRIMA), and measures how well each mesh matches
the giraffe's size in the image.

## Setup and run

```bash
./setup_env.sh               # creates the `prima` conda env, clones PRIMA, downloads ~6 GB of weights
conda activate prima
python run.py 1.mp4          # all giraffes (about 80 s for this 170-frame video)
python run.py 1.mp4 --max-animals 1   # only the largest giraffe per frame
python fit_giraffe.py 1.mp4  # builds a real giraffe model; see "Giraffe model" below
pytest tests
```

## Outputs (`outputs/<video>/`)

| File | Contents |
|---|---|
| `dashboard.mp4` | Input beside overlay; for each giraffe, the mesh on its crop and a fixed-scale side profile |
| `overlay.mp4` | Full-resolution meshes on the video, with track IDs |
| `meshes/*.obj` | 3D meshes every 10th frame (`--mesh-stride`), in camera coordinates |
| `predictions.npz` | Vertices, faces, SMAL pose and shape, and camera translation for every instance |
| `metrics.csv`, `summary.json`, `fit_vs_size.png` | Per-instance measurements and their summary |

## Differences from PRIMA's own demo

- **Giraffes are detected.** PRIMA's `ANIMAL_COCO_IDS` omits COCO class 23 (giraffe). On `1.mp4` its
  filter kept 2 boxes in 170 frames, compared with 634 giraffe instances here.
- **Multiple animals in the video pipeline.** `demo_video.py` handles only the top-scoring animal per
  frame. This pipeline runs every giraffe, adds tracking, and renders them in one scene so that they
  occlude each other correctly.
- **Mask R-CNN instead of Faster R-CNN**, giving masks to measure the mesh fit against.
- **torch 2.7.1 + CUDA 12.8 instead of 2.2.1**, because Blackwell GPUs need it. DeepLabCut, pytorch3d,
  open3d and mmcv are left out: this inference path does not use them, but `demo_tta.py` and `app.py`
  do need DeepLabCut, so those will not run in this env.

## Results on `1.mp4`

Medians over the 571 instances that are neither cut off by the frame edge nor overlapping another
giraffe's box.

| Giraffe height in image | Instances | Mesh/mask height | Mesh/mask width | Mesh/mask area | Silhouette IoU | Neck recall |
|---|---:|---:|---:|---:|---:|---:|
| Small (<150 px) | 137 | 0.98 | 1.25 | 1.11 | 0.55 | 0.68 |
| Medium (150-250 px) | 354 | 0.98 | 1.17 | 1.25 | 0.50 | 0.62 |
| Large (>=250 px) | 80 | 0.93 | 1.19 | 1.29 | 0.55 | 0.60 |

**Apparent size and motion: handled.** Giraffes range from 95 to 335 px tall. Mesh height is within
15% of the mask height in 96% of instances (85% for small giraffes, 96-100% for the rest), and IoU
shows no trend with size. Predicted shape stays stable within a track (coefficient of variation of neck/body about 2%)
while size changes, so the change goes into camera depth instead (correlation of image height with
1/depth is 0.87-0.98 for tracks with a substantial size change).

**Giraffe body proportions: not handled.** The meshes look like a foal or horse. They are too bulky
(about 24% more area than the mask) and the neck is far too short and thick: the predicted neck/body
ratio is 0.27, barely above the generic mean SMAL animal's 0.23. A real giraffe's neck is roughly as
long as its body, so that ratio should be several times larger (a rough anatomical figure, not
measured here). The meshes reach the right height with longer legs instead (leg/body 0.88 versus
0.69 for the mean animal) and a body that overflows the mask.

**Other failures seen:** when a giraffe faces the camera or walks directly away, the orientation can
be wrong; frame 98, checked by eye, has a side-on mesh on a giraffe seen end-on. The 14 instances with
neck recall under 0.3 nearly all have narrow boxes (width about a quarter to a third of the height),
which fits this pattern, but the others were not individually inspected.

### Reading these numbers

- Height agreement is partly built in, because PRIMA regresses its camera from a crop tight around the
  detection. It shows the mesh fits its box, not that the anatomy is right.
- There is **no metric scale**: the output cannot give a giraffe's height in metres, as depth and size
  are ambiguous from one camera with an assumed focal length.
- The detector mask is a reference, not ground truth, and IoU is harsh on thin legs and necks.
- Track IDs are not re-identified: 8 tracks were produced with at most 6 giraffes visible at once, so
  a giraffe that leaves and returns gets a new ID. ID swaps during crossings were not audited.
- Scoring at threshold 0.7 drops 14 low-score boxes from the lower-threshold run, including one
  head-only and one spurious box, but also some real giraffes for single frames.

## Giraffe model (`fit_giraffe.py`)

PRIMA has no per-species models. It has one SMAL template and predicts 41 shape coefficients per image.
SMAL's shape space comes from 41 toy scans in 5 animal families, with no giraffe family, so the
predictions on this video sit next to one family's average (the closest of the 5 for every instance).

`python fit_giraffe.py 1.mp4` runs after `run.py` and builds a giraffe on top of SMAL's topology and
skeleton, by fitting to the instance masks:

1. Learnable length and thickness for each bone group, within anatomical limits (`giraffe3d/body.py`).
2. Smooth, left-right symmetric per-vertex offsets.
3. One shape shared by all giraffes; pose and camera per instance, starting from PRIMA's estimate.

About 4 minutes on this video. Outputs in `outputs/<video>_giraffe/`:

| File | Contents |
|---|---|
| `giraffe_model.glb`, `.obj` | The standalone model, standing, with colors from the video |
| `giraffe_turntable.mp4` | The model rotating |
| `comparison.mp4` | PRIMA as-is beside the fitted model, per-giraffe crops and profiles |
| `overlay.mp4`, `meshes/*.obj` | The fitted model over the video; meshes every 10th frame |
| `fit.npz`, `metrics.csv`, `summary.json` | All fitted parameters, measurements and summary |

| Median, 571 clean instances | PRIMA | Fitted giraffe |
|---|---:|---:|
| Silhouette IoU | 0.52 | 0.74 |
| Silhouette IoU, 3 tracks not used for shape (60) | 0.57 | 0.72 |
| Mesh/mask width | 1.18 | 0.97 |
| Mesh/mask area | 1.24 | 0.80 |
| Neck recall | 0.63 | 0.81 |
| Neck/body length ratio | 0.27 | 0.88 |

**Limits of this result:**

- **These scores are not independent.** The masks being scored against are also the fitting target, so
  the improvement partly reflects that. Held-out tracks test only the *shape*; their poses are still
  fitted to their own masks.
- **It is no longer pure PRIMA inference.** It needs masks and per-video optimization, not one network
  pass, and the starting proportions and limits come from general giraffe anatomy.
- **Width is weakly constrained.** Most views are side-on, so body thickness relies on the priors.
- **The mesh runs thin** (area 0.80) and legs still twist or cross in some frames.
- **The neck is one rigid segment**, because SMAL has a single neck joint, so it cannot curve.
- **No ossicones or fine detail**, and the same shape is shared by every animal, including any calves.
- **Colors are only approximate:** they carry the footage's magenta cast, and the spots blur away as the
  mesh is too coarse (3,889 vertices) and imperfectly aligned to keep them.
- **Still no metric scale:** real height in metres cannot be read from the model.

## Possible next steps

PRIMA's test-time adaptation (`demo_tta.py`) refines predictions on each image using keypoints. It was
**not** evaluated here and needs DeepLabCut installed. It could provide better pose starting points for
`fit_giraffe.py`, but it would not change PRIMA's limited shape space.

## License

PRIMA's weights are for **academic, non-commercial use only** under its modified MIT license; see
`third_party/PRIMA/LICENSE.md`.
