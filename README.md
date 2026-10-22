# Frame order does not matter

Code for the paper *Frame order does not matter: the limits of temporal
explanation in lightweight video violence detection* (Negre, Alonso, Chamoso,
Prieto), under review at Springer *Signal, Image and Video Processing*.

The detector is a MobileNetV2 backbone that produces one descriptor per frame,
followed by a Bi-LSTM head that classifies the sequence. It reaches 96.4% on
RLVS. We set out to explain it along the time axis and ended up reporting
something else: **once trained, it barely uses the order of its input
frames.** Over twenty random permutations of the 40 frames of each test clip,
accuracy moves by less than the binomial noise of the test split on all six
benchmarks, and at least 96.7% of clips keep their predicted label.

That leaves temporal attribution of this model with almost nothing to find,
and the repository is laid out so you can check it rather than take our word
for it:

- `scripts/run_temporal_probe.py` is the experiment the claim rests on. It
  needs no training (it loads cached descriptors and a trained head, permutes
  the frame axis and re-runs the head), so the whole sweep takes minutes.
- `scripts/run_synthetic_xai_benchmark.py` is the control. On sequences with an
  event planted at a known offset, the same attribution methods recover it
  almost perfectly, which is what separates "the model has no temporal
  dependence" from "the methods do not work".
- `results/` holds every number in the paper, and `scripts/make_tables.py` and
  `scripts/make_figures.py` turn those files into the tables and figures.
  Nothing in the paper was typed in by hand.

## Layout

```
src/
  data/         dataset indexing, clip loading, duplicate-aware splits
  models/       backbone, recurrent head, ablation variants
  keyframe/     frame differencing, YOLO person detection
  xai/          Grad-CAM, frame importance, temporal baselines, metrics
  robustness/   corruption suite and adversarial attacks
  eval/         params/FLOPs/latency, confidence intervals, significance tests
scripts/        one entry point per experiment
slurm/          job files for the BISITE cluster
results/        the JSON every table and figure is generated from
figures/        the generated figures
legacy/         the original thesis-era scripts, kept verbatim for reference
```

`legacy/` is the code as it was written during the first round of this work
(VGG19 features, YOLOv8, the first version of the frame-removal loop). It is
not wired into the current pipeline and is only here so the provenance of the
method is visible. The `src/` tree is the cleaned-up reimplementation that the
reported numbers come from.

## Setup

```bash
conda create -n xai_vd python=3.10
conda activate xai_vd
pip install -r requirements.txt
```

TensorFlow 2.15 with `numpy<2`; newer numpy breaks the prebuilt wheels.
`ultralytics` and `timeshap` are optional; the keyframe and TimeSHAP-comparison
scripts degrade gracefully without them (the temporal comparison falls back to
a sampling-based Shapley estimator and says so).

## Data

None of the datasets are redistributed here. Download them and point
`DATA_ROOT` at the parent directory:

| dataset | clips | where |
|---|---|---|
| RLVS | 2000 | kaggle: `mohamedmustafa/real-life-violence-situations-dataset` |
| RWF-2000 | 2000 | kaggle: `vulamnguyen/rwf2000` |
| Hockey Fights | 1000 | kaggle: `yassershrief/hockey-fight-vidoes` |
| Movies Fight | 198 | kaggle: `naveenk903/movies-fight-detection-dataset` |
| Violent Flows | 246 | https://www.openu.ac.il/home/hassner/data/violentflows/ |
| Industrial Surveillance | 300 | github: `FathUMinUllah3797/VD_ConvLSTM_GRU` |

The loaders expect two class subdirectories per dataset (`Violence/` and
`NonViolence/`, or the equivalent names the Kaggle mirrors use; see
`src/data/datasets.py`). RWF-2000 keeps its official train/val split.

Run `scripts/check_dataset_integrity.py` before anything else. Every mirror we
downloaded ships byte-identical clips under different names (14 groups in
RLVS, 5 in the RWF-2000 training split, 3 in Hockey, 3 in Movies, 1 in Violent
Flows), and one RLVS pair carries opposite labels. A random split puts about
46% of each pair on opposite sides of the train/test boundary, so the splits in
`src/data/datasets.py` are computed over duplicate *groups* instead of files.
The Movies mirror has three redundant copies, which is why the table above says
198 and not the usual 200.

UCF-Crime and XD-Violence are deliberately absent. They are untrimmed,
weakly-labelled anomaly benchmarks scored with frame-level AUC / AP, so a
trimmed binary clip classifier cannot be evaluated on them without wrapping it
in a multiple-instance-learning head, which is a different model and not a
different test set. See `docs/datasets.md`.

## Running the experiments

The pipeline is two-stage on purpose: the backbone is fine-tuned once, every
clip is pushed through it once, and everything afterwards works on the cached
`(40, 1280)` descriptor arrays. That is what makes the explanation sweep and
the order probe cost fractions of a second per clip instead of minutes.

```bash
# 1. adapt the backbone (once per fine-tuning depth)
python scripts/finetune_backbone.py --root $DATA_ROOT/rlvs --finetune 5 \
    --out ckpt/mnv2_ft5.h5

# 2. cache descriptors (once per dataset)
python scripts/extract_features.py --dataset rlvs --root $DATA_ROOT/rlvs \
    --out features/rlvs --backbone-weights ckpt/mnv2_ft5.h5

# 3. train the head over 5 seeds
python scripts/train.py --features features/rlvs --seeds 5 \
    --ckpt-prefix ckpt/head_rlvs --out results/main_rlvs.json

# 4. the rest
python scripts/run_temporal_probe.py --features features/rlvs \
    --head-weights ckpt/head_rlvs_seed0.h5 --n-perm 20 \
    --out results/temporal_probe.json
python scripts/run_kfold.py         --features features/rlvs --k 10
python scripts/run_ablation.py      --features features/rlvs
python scripts/run_cross_dataset.py --features features
python scripts/run_xai_eval.py      --features features/rlvs --head-weights ckpt/head_rlvs_seed0.h5
python scripts/run_robustness.py    --root $DATA_ROOT/rlvs --backbone-weights ckpt/mnv2_ft5.h5 --head-weights ckpt/head_rlvs_seed0.h5
python scripts/measure_complexity.py --cpu

# 5. regenerate the tables and figures from whatever is in results/
python scripts/make_tables.py  --results results --out tables.tex
python scripts/make_figures.py --results results --out figures/fig
```

RWF-2000 gets its own backbone, fine-tuned on its official training split only,
so that the 400 held-out clips never reach the CNN. Step 1 takes the dataset
name for that: `--root $DATA_ROOT/rwf2000 --dataset rwf2000`.

On the cluster, `slurm/` has one job file per step; they assume a conda env
called `xai_vd` and the repo checked out at `~/xai-vd`.

```bash
sbatch slurm/01_finetune_backbone.sh 5
for d in rlvs rwf2000 hockey movies violentflows; do
    sbatch slurm/02_extract_features.sh $d
done
...
sbatch slurm/09_temporal_probe.sh      # all five datasets, ~2 min
```

`docs/hpc.md` has the full chain with dependencies, and a table of which job
fills which table. Partitions follow the current cluster layout (`short` ≤ 4 h,
`medium` ≤ 1 day, `long` ≤ 7 days). Only the robustness sweep needs `long`.

## Things that do not need a trained model

Two experiments are self-contained and run on a laptop in minutes, which is
useful for checking the setup:

```bash
python scripts/measure_complexity.py --cpu --out results/complexity_cpu.json
python scripts/run_synthetic_xai_benchmark.py --out results/synthetic_xai_benchmark.json
```

The first reports parameters, FLOPs and latency, all fixed by the
architecture, so weights are irrelevant. The second plants a short event at a
known position in a synthetic sequence and checks whether each temporal
explanation method recovers it. It is the only setting in which "correct"
attribution is actually defined, and it is why the paper can say the methods
are sound and the model is not.

`scripts/make_figures.py` also runs anywhere: it only needs `results/` and
matplotlib.

## Is the RWF-2000 gap really the missing motion?

Showing that the detector ignores frame order does not by itself prove that the
points it loses on RWF-2000 are the motion it cannot see. The control keeps the
whole pipeline fixed and changes only whether motion is in the input:

```bash
# cache a second set of descriptors from frame-to-frame differences
python scripts/extract_features.py --dataset rwf2000 --root $DATA_ROOT/rwf2000 \
    --input-mode diff --backbone-weights ckpt/mnv2_ft5_rwf2000.h5 \
    --out features/rwf2000_diff

# same head, same split, same seeds; appearance vs motion vs both
python scripts/run_motion_control.py --dataset rwf2000 \
    --features-rgb features/rwf2000 --features-diff features/rwf2000_diff \
    --out results/motion_control_rwf2000.json
```

It probes each arm too. On RWF-2000 the difference arm gains 13.6 points and
is still flat under the probe: the motion sits inside each per-frame
descriptor, so a head that ignores order can use it. On RLVS the difference
arm gains nothing.

`scripts/run_semisynthetic_xai.py` is the other half of that argument. It
scores the attribution methods against an event placed in real footage: a real
non-violent clip with a window spliced in from a donor, violent for positives
and non-violent for negatives, so the cut carries no label information and only
the window content does.

## Notes

- FLOPs come from the TF profiler, which counts a multiply-accumulate as two
  operations. Papers quoting MACs report roughly half these numbers; both are
  printed to avoid confusion.
- Efficiency is reported on CPU. On an H100 the comparison is meaningless at
  this batch size: VGG-19's 65× arithmetic disadvantage shrinks to 1.7× in
  latency because the GPU spends its time on kernel launches. Both measurements
  are in `results/`.
- The Bi-LSTM head is `(64, 32)` units with `(64, 32)` dense layers. The first
  version of this work did not record the layer widths, so these were fixed
  when the code was rewritten and are what all reported numbers use.
- Clip length is 40 frames sampled at equal intervals over the whole video,
  resized to 128×128.
- The keyframe gate decides which frames are shown to an analyst. It does not
  change what the classifier sees, which is always the fixed 40-frame sample.

## Citation

```bibtex
@article{negre2026xaivd,
  title   = {Frame order does not matter: the limits of temporal explanation
             in lightweight video violence detection},
  author  = {Negre, Pablo and Alonso, Ricardo S. and Chamoso, Pablo and Prieto, Javier},
  journal = {Signal, Image and Video Processing},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT, see `LICENSE`. The datasets keep their own licences.
