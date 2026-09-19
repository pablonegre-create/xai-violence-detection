# Datasets and why the selection looks the way it does

## What is used

| dataset | clips | balance | protocol | metric |
|---|---|---|---|---|
| RLVS | 2000 | 1000 / 1000 | 70/10/20 stratified holdout | accuracy, P/R/F1, AUC |
| RWF-2000 | 2000 | 1000 / 1000 | official train/val split | accuracy, P/R/F1, AUC |
| Hockey Fights | 1000 | 500 / 500 | 70/10/20 stratified holdout | accuracy, P/R/F1, AUC |
| Movies Fight | 198 | 99 / 99 | 70/10/20 stratified holdout | accuracy, P/R/F1, AUC |
| Violent Flows | 246 | 123 / 123 | 70/10/20 stratified holdout | accuracy, P/R/F1, AUC |

All five are *trimmed clip classification*: a short video is labelled violent
or not, and the whole clip carries one label. That is the task this model
solves, so these are the datasets it can be evaluated on directly.

The splits are produced by `src/data/datasets.py` with a fixed seed so the
numbers are reproducible. Hockey, Movies and Violent Flows are small enough
that a single holdout split is noisy, which is why much of the literature uses
5-fold CV on them. We use a stratified holdout instead and absorb the noise a
different way: every configuration is trained with five seeds and reported as
mean +/- sd, with a Wilson interval on the pooled test accuracy. `kfold_split`
is in `datasets.py` for anyone who wants the CV protocol, but no number in the
paper comes from it.

## What is not used, and why

**UCF-Crime** (Sultani et al., CVPR 2018) and **XD-Violence** (Wu et al., ECCV
2020) are a different problem. They are long untrimmed surveillance videos with
*video-level* labels only: you are told a crime happens somewhere in a
13-minute recording, not when. The task is weakly-supervised temporal anomaly
detection, scored with frame-level ROC-AUC (UCF-Crime) or average precision
(XD-Violence), and the standard approach is multiple-instance learning over
snippet bags.

A trimmed binary clip classifier cannot be evaluated on either without being
wrapped in an MIL head — ranking snippets inside a bag, with a ranking loss
over positive and negative bags. That is a different model with a different
training objective, and its performance would say very little about the
detector described in the paper. Reporting a number from such a wrapper as if
it were "our model on UCF-Crime" would be misleading.

So the generalisation evidence here comes from the four additional trimmed
datasets plus the cross-dataset transfer matrix (train on one, test on the
others with no adaptation), which measures the same thing — does the decision
rule survive a domain change — on data where the comparison is meaningful.

## Getting the data

The RWF-2000 mirror will not extract with `--unzip`. Some of its entries have
names that were mangled on the way in (Cyrillic re-encoded through CP437) and
exceed the 255-byte limit Linux puts on a path component, so the extraction
aborts with `OSError: [Errno 36] File name too long`, leaving the dataset half
written. Download it without extracting and use the helper instead:

```bash
kaggle datasets download -d vulamnguyen/rwf2000 -p rwf2000        # no --unzip
python scripts/extract_dataset_zip.py rwf2000/rwf2000.zip -o rwf2000
```

The helper replaces over-long basenames with a short hash and records the
mapping in `_renamed.csv`. Only the parent directory carries the label, so the
names themselves do not matter.

```bash
export DATA_ROOT=$HOME/data
mkdir -p $DATA_ROOT && cd $DATA_ROOT

kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset -p rlvs --unzip
kaggle datasets download -d vulamnguyen/rwf2000 -p rwf2000 --unzip
kaggle datasets download -d yassershrief/hockey-fight-vidoes -p hockey --unzip
kaggle datasets download -d naveenk903/movies-fight-detection-dataset -p movies --unzip
# Violent Flows: request from https://www.openu.ac.il/home/hassner/data/violentflows/
# It arrives as movies.rar (158 MB). The classification benchmark is that file;
# 21VideosForDetection.rar is a different task and is not used here. The two
# CSVs are source metadata (YouTube URLs and time ranges), not needed to train.
#   tar -xf movies.rar -C violentflows/        # bsdtar reads RAR v4
```

Roughly 8 GB in total. The Kaggle mirrors nest the class folders differently
from each other; `index_flat()` walks the tree and matches directory names
case-insensitively against a list of known aliases (`Violence`, `fight`,
`NonViolence`, `nonfight`, ...), so an extra level of nesting is fine but an
unrecognised class name is not — add it to `VIOLENT_DIRS` / `PEACEFUL_DIRS`
if you hit one.

## Duplicate clips in the published mirrors

Every one of these benchmarks contains byte-identical clips stored under
different names. Counted with `scripts/check_dataset_integrity.py`:

| dataset | indexed | duplicate groups | unique |
|---|---|---|---|
| RLVS | 2000 | 14 (one spanning both classes) | 1986 |
| Hockey Fights | 1000 | 3 | 997 |
| Movies Fight | 198 | 0 after cleaning the mirror | 198 |
| Violent Flows | 246 | 1 | 245 |
| RWF-2000 train | 1600 | 5 | 1595 |
| RWF-2000 val | 400 | 0 | 400 |

A random 70/10/20 split puts roughly 46% of each duplicate pair into two
different partitions, so the model would be tested on clips it had memorised.
`stratified_split` and `kfold_split` therefore assign a whole duplicate group
to one partition. The clip counts stay as published, so the numbers remain
comparable with the literature, but the leakage is gone. Pass
`group_duplicates=False` to reproduce the naive split.

Two cases need judgement rather than a rule:

- **RLVS** contains one clip stored under both labels: `NV_226.mp4` and
  `V_504.mp4` are the same file. That is an annotation contradiction, not a
  duplicate. It is one pair in 2000 and is left in place; the group is assigned
  to a single partition by majority label.
- **Violent Flows** contains one clip harvested under two search keywords,
  `football_crowds__Brisbane_Lions...` and `stadium_crowds__Brisbane_Lions...`.
  This is in the original distribution, and both copies sit in the same fold
  and class, so the official protocol never splits them. Left untouched.

The Movies Fight mirror (`naveenk903`) is the one that needs fixing before use:
it ships three redundant copies, two in `noFights` and one in `fights`, and is
missing one clip per class relative to the 100+100 of Nievas et al. After
removing the redundant copies it holds 198 clips, 99 per class, and that is
what should be reported rather than 200.

## Preprocessing

40 frames per clip, sampled at equal intervals across the whole video, resized
to 128×128, scaled to [0, 1]. Uniform sampling rather than a fixed-rate window
means a 2-second clip and a 10-second clip both become 40 frames, at the cost
of a different effective frame rate — acceptable here because all five
datasets contain short clips (2–10 s).

Violent Flows is the one dataset where the 40-frame budget bites: its clips run
26-163 frames (median 90), and 11 of the 246 fall below 40. Uniform sampling
then repeats frames - the shortest clip yields 24 distinct frames out of 40 -
which is nearest-neighbour temporal upsampling rather than an error. It is
worth remembering when reading the per-dataset numbers, since the temporal
attribution has correspondingly less to work with there.

It also ships as five numbered directories, which are the official
cross-validation folds with the classes one level deeper (50/50/50/48/48,
balanced 123/123). `index_flat` pools them; `index_folds` returns the official
assignment if you want to reproduce the published protocol exactly rather than
regenerate folds.

128×128 rather than MobileNetV2's native 224×224 is a deliberate cost
decision: it cuts the per-frame FLOPs by a factor of ~3 and still leaves a
4×4×1280 feature map, which is enough spatial resolution for Grad-CAM to be
readable after upsampling.
