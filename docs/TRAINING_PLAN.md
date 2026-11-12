# Training plan: steps 3 and 4

Goal: turn the per-clip fits into a model that predicts the extended body (pose, bone scales, horns and humps, and metric
distance) from a single crop in one forward pass, with a learned pose prior that keeps motion smooth and gait-like.

Budget: 48 hours of GPU time from the start of label generation (2026-11-10, about 10:25), on one RTX PRO 4000
Blackwell (24 GB). Data: WildDepth release snippets of giraffe, Ankole cattle, rhino, camel, lion/lioness and donkey only
(`make_labels.py`). Release `19103277` has no species labels and is not used.

All results of these experiments go under `new_output/`: `labels/` (phase 2), `kinematics_existing/` (step 1 on the
earlier clips), `dataset/` (3a), `prior/` (3b), `student/` (4a-4b) and `eval/` (4c).

| Phase | Work | GPU hours | Ends by (if on time) |
|---|---|---:|---|
| 2 | Label generation: 63 windows from 37 snippets (`new_output/labels/runs/`) | ≤ 20 (hard stop) | Nov 11, 06:30 |
| 3a | Collect and filter labels into one dataset; cache backbone features | 2 | Nov 11, 08:30 |
| 3b | Pose prior (VAE) and latent smoothing | 2 | Nov 11, 10:30 |
| 4a | Train the student on cached features (frozen backbone) | 12 | Nov 11, 22:30 |
| 4b | Optional: unfreeze the last backbone blocks, short fine-tune | 4 | Nov 12, 02:30 |
| 4c | Evaluation on held-out snippets, ablations, kinematics | 4 | Nov 12, 06:30 |
| | Slack | 4 | Nov 12, 10:30 |

Steps 3a and 3b need only CPU and the GPU for minutes, so they can start on the windows finished so far while phase 2
is still running, and be rerun at the end.

## Data and splits (phase 3a)

- **Instances.** Each fitted instance gives: the crop, its mask, its fitted pose (34 joint rotations, global
  orientation), camera translation, the clip's bone scales and part parameters, the species, the held-out and fitting
  LiDAR points, and its LiDAR scale (metres per model unit) when the track has one. Expected: about 63 × 300 frames ×
  1-3 animals, roughly 20-40k instances before filtering.
- **Quality filter, as for the fits' own reliability:** not truncated, not overlapping, IoU ≥ 0.7, box at least 40 px
  tall. A window whose fit has IoU below 0.6 overall is dropped whole. The filter is applied before splitting.
- **Split by snippet, never by frame.** Neighbouring frames are nearly identical, so a frame split would leak.
  Test: one snippet per species, chosen before training and fixed in `new_output/labels/split.json`, including at least one
  Bubye Valley snippet (giraffe or donkey) so a second site is tested. Validation: one more window per species. Train:
  the rest. The clips already used to develop the method (`Rhino_1`, `Rhino_3`, `Camel_1`, `giraffe_4`,
  `Ankle_cow_2/5`) go to train, since their settings were tuned on them.
- **Features.** PRIMA's ViT-H backbone, frozen, run once per crop at 256 px; tokens stored in fp16 (about 0.5 MB per
  crop, 10-20 GB in total). This is what makes 12 hours of training enough.

## Step 3: pose prior and latent smoothing (phase 3b)

- **Pose VAE.** Input: the 34 joint rotations as 6D vectors (204 numbers), from all training instances. Encoder and
  decoder: MLPs of 512 units, latent size 32, conditioned on a species embedding (6 species, learned). Loss:
  rotation reconstruction (geodesic) plus KL divergence with weight annealed to 1e-3. Dropout 0.1 in both MLPs. A few
  minutes per run on the GPU; train 3 seeds and keep the best by validation reconstruction error.
- **Latent smoothing.** For each track, optimise one latent code per frame, initialised by the encoder from the
  fitted poses. The cost has three parts: the existing mask term, the LiDAR depth term, and smoothness in latent
  space (squared second differences of the codes, scaled by the frame interval, so dropped frames are handled).
  This replaces the per-joint temporal term used now.
- **Gait in the prior.** Add a periodic term: within a run of walking frames (stride periodicity ≥ 0.4 in
  `gait.json`), each code should be close to the code one stride period earlier. This term is optional: it is kept
  only if it improves the held-out masks, not just the gait plots.
- **Checks.** Held-out mask IoU and LiDAR error must not drop by more than 0.01 compared with the per-clip fit.
  Jitter, measured as the mean joint acceleration, should fall. Stride periodicity on walking tracks should rise.
  Samples from the prior should look like the species when rendered: one sheet of 16 samples per species.

## Step 4: the student network (phases 4a-4c)

- **Architecture.** Start from PRIMA's own decoder (6-layer transformer, BioCLIP taxonomic embedding), on the frozen
  backbone tokens. Its outputs stay the same: pose, global orientation, camera. New outputs, each a small MLP head:
  - bone scales (13 groups × length, thickness), within the same limits as the fits;
  - part parameters: horn layout logits (none / paired / nasal), horn shape, hump count logits and hump shape;
  - metric distance: log depth of the animal in metres, given the camera intrinsics, which is what turns a crop into a
    metric size.
- **Losses.**
  - Pseudo-labels from the fits: pose (geodesic), bone scales and parts (L1 within limits), camera translation.
    Weighted by fit quality (IoU), so a poor fit teaches less.
  - Masks: the same mask term as the fitting, on the predicted mesh, so the student is not limited to copying its
    teacher.
  - LiDAR: the depth-relief term on the fitting half of the points, and a metric depth term against the LiDAR median
    depth. The held-out half is never used in training, in any phase.
  - Prior: the frozen pose VAE's negative log-likelihood of the predicted pose, at a small weight.
  - Temporal, on pairs of neighbouring frames in a batch: the latent codes of their predicted poses should be close.
- **Regularisation and dropout.**
  - dropout 0.1 in the decoder and the heads; stochastic depth 0.1 if the backbone is unfrozen in 4b;
  - LiDAR point dropout: each step keeps a random 30-100% of the fitting points, so the model does not rely on
    dense depth, since far animals have few points;
  - occlusion: random rectangles cut from the crop and the mask (as background and unknown), on 30% of crops;
  - frame dropout in the temporal term: neighbour pairs 1-3 frames apart;
  - the usual crop jitter, scale, and colour jitter. No left-right flips: the fits' left and right legs would need
    swapping, and the gain is small.
- **Schedule.** AdamW, learning rate 1e-4 for the new heads and 1e-5 for PRIMA's decoder, cosine decay, batch 64 on cached
  features. Validate every epoch on the validation windows. Keep the best checkpoint by validation IoU plus LiDAR error.
  Hard stop at 12 hours.
- **4b, only if 4a finishes early.** Unfreeze the last 4 of the 32 ViT-H blocks with gradient checkpointing, batch 16,
  learning rate 5e-6, at most 4 hours. Kept only if validation improves.

## Evaluation (phase 4c)

On the test snippets, which have not been seen in any phase:

| Measure | Compared methods |
|---|---|
| Mask IoU (all clean instances; small, medium and large animals separately) | PRIMA; per-clip fit (teacher); student; student with latent smoothing |
| Held-out LiDAR relief error, and a flat-surface baseline | same |
| Metric size: shoulder height spread per animal, and against species ranges from the literature | per-clip fit; student |
| Metric distance error against LiDAR median depth | student (PRIMA and the fit need LiDAR for this) |
| Jitter (joint acceleration) and stride periodicity on walking tracks | fit; student; student with smoothing |
| Runtime per frame | all |

Ablations (on validation only, so the test set is used once): without the mask term, without LiDAR terms, without the
pose prior, without point dropout and occlusion.

Biomechanics: per species, stride frequency, duty factor (if the foot contacts can be read reliably) and ranges of
elbow, carpus, stifle and hock angles over walking cycles. Compare with published values where they exist (for
giraffes, Basu et al., J. Exp. Biol. 2019). SMAL's joints are approximate, so these are reported as skeleton angles,
not anatomical ones.

## Risks and what to do

- **Labels are the teacher's errors.** The quality filter, the IoU weighting and the mask and LiDAR terms limit this,
  and the test compares against masks and LiDAR, not against the teacher.
- **Few animals per species** (4 rhino snippets, 4 camel snippets): per-species results will be noisy. Report the
  number of animals with every figure.
- **Time.** If label generation is slower than 20 minutes per window, the remaining windows are thinned to one per
  snippet. If 4a has not converged by 12 hours, the best checkpoint is used and 4b is skipped.
- **Licence.** PRIMA's weights are for academic, non-commercial use only, and so is any model fine-tuned from them.
