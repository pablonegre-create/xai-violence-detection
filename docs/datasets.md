# Datasets and why the selection looks the way it does

## What is used

| dataset | clips | balance | protocol | metric |
|---|---|---|---|---|
| RLVS | 2000 | 1000 / 1000 | 70/10/20 stratified holdout | accuracy, P/R/F1, AUC |
| RWF-2000 | 2000 | 1000 / 1000 | official train/val split | accuracy, P/R/F1, AUC |
| Hockey Fights | 1000 | 500 / 500 | 5-fold CV | accuracy |
| Movies Fight | 200 | 100 / 100 | 5-fold CV | accuracy |
| Violent Flows | 246 | 123 / 123 | 5-fold CV | accuracy |

All five are *trimmed clip classification*: a short video is labelled violent
or not, and the whole clip carries one label. That is the task this model
solves, so these are the datasets it can be evaluated on directly.

The splits are produced by `src/data/datasets.py` with a fixed seed so the
numbers are reproducible. Hockey, Movies and Violent Flows are small enough
that a single holdout split is noisy, which is why the literature settled on
5-fold CV for them and why we follow it.

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

```bash
export DATA_ROOT=$HOME/data
mkdir -p $DATA_ROOT && cd $DATA_ROOT

kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset -p rlvs --unzip
kaggle datasets download -d vulamnguyen/rwf2000 -p rwf2000 --unzip
kaggle datasets download -d yassershrief/hockey-fight-vidoes -p hockey --unzip
kaggle datasets download -d naveenk903/movies-fight-detection-dataset -p movies --unzip
# Violent Flows: request from https://www.openu.ac.il/home/hassner/data/violentflows/
```

Roughly 8 GB in total. The Kaggle mirrors nest the class folders differently
from each other; `index_flat()` walks the tree and matches directory names
case-insensitively against a list of known aliases (`Violence`, `fight`,
`NonViolence`, `nonfight`, ...), so an extra level of nesting is fine but an
unrecognised class name is not — add it to `VIOLENT_DIRS` / `PEACEFUL_DIRS`
if you hit one.

## Preprocessing

40 frames per clip, sampled at equal intervals across the whole video, resized
to 128×128, scaled to [0, 1]. Uniform sampling rather than a fixed-rate window
means a 2-second clip and a 10-second clip both become 40 frames, at the cost
of a different effective frame rate — acceptable here because all five
datasets contain short clips (2–10 s).

128×128 rather than MobileNetV2's native 224×224 is a deliberate cost
decision: it cuts the per-frame FLOPs by a factor of ~3 and still leaves a
4×4×1280 feature map, which is enough spatial resolution for Grad-CAM to be
readable after upsampling.
