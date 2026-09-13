# Explainable lightweight video violence detection

Code for the paper *Explainable Lightweight Video Violence Detection*
(Negre, Alonso, Chamoso, Prieto), under review at Springer *Signal, Image and
Video Processing*.

The detector is a MobileNetV2 backbone that produces one descriptor per frame
plus a Bi-LSTM head that classifies the sequence. Around it sit three
explanation components: YOLO + frame differencing to justify which frames are
analysed at all, Grad-CAM over the backbone blocks to show *where* the network
is looking, and a block-ablation method that says *when* in the clip the
decision is made.

## Layout

```
src/
  data/         dataset indexing and clip loading
  models/       backbone, recurrent head, ablation variants
  keyframe/     frame differencing, YOLO person detection
  xai/          Grad-CAM, frame importance, temporal baselines, metrics
  robustness/   corruption suite and adversarial attacks
  eval/         params/FLOPs/latency, confidence intervals, significance tests
scripts/        one entry point per experiment
slurm/          job files for the BISITE cluster
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

TensorFlow 2.15 with `numpy<2` — newer numpy breaks the prebuilt wheels.
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
| Movies Fight | 200 | kaggle: `naveenk903/movies-fight-detection-dataset` |
| Violent Flows | 246 | https://www.openu.ac.il/home/hassner/data/violentflows/ |

The loaders expect two class subdirectories per dataset (`Violence/` and
`NonViolence/`, or the equivalent names the Kaggle mirrors use — see
`src/data/datasets.py`). RWF-2000 keeps its official train/val split.

UCF-Crime and XD-Violence are deliberately absent. They are untrimmed,
weakly-labelled anomaly benchmarks scored with frame-level AUC / AP, so a
trimmed binary clip classifier cannot be evaluated on them without wrapping it
in a multiple-instance-learning head — a different model, not a different test
set. See `docs/datasets.md`.

## Running the experiments

The pipeline is two-stage on purpose: the backbone is fine-tuned once, every
clip is pushed through it once, and everything afterwards works on the cached
`(40, 1280)` descriptor arrays. That is what makes the explanation sweep cost
fractions of a second per clip instead of minutes.

```bash
# 1. adapt the backbone (once per fine-tuning depth)
python scripts/finetune_backbone.py --root $DATA_ROOT/rlvs --finetune 5 \
    --out ckpt/mnv2_ft5.h5

# 2. cache descriptors (once per dataset)
python scripts/extract_features.py --dataset rlvs --root $DATA_ROOT/rlvs \
    --out features/rlvs --backbone-weights ckpt/mnv2_ft5.h5

# 3. train the head over 5 seeds
python scripts/train.py --features features/rlvs --seeds 5 \
    --out results/main_rlvs.json

# 4. the rest
python scripts/run_ablation.py      --features features/rlvs
python scripts/run_cross_dataset.py --features features
python scripts/run_xai_eval.py      --features features/rlvs --head-weights ckpt/head_rlvs_seed0.h5
python scripts/run_robustness.py    --root $DATA_ROOT/rlvs --backbone-weights ckpt/mnv2_ft5.h5 --head-weights ckpt/head_rlvs_seed0.h5
python scripts/measure_complexity.py --cpu
```

On the cluster, `slurm/` has one job file per step; they assume a conda env
called `xai_vd` and the repo checked out at `~/xai-vd`.

```bash
sbatch slurm/01_finetune_backbone.sh 5
for d in rlvs rwf2000 hockey movies violentflows; do
    sbatch slurm/02_extract_features.sh $d
done
```

Partitions follow the current cluster layout (`short` ≤ 4 h, `medium` ≤ 1 day,
`long` ≤ 7 days). Only the robustness sweep needs `long`.

## Things that do not need a trained model

Two experiments are self-contained and run on a laptop in minutes, which is
useful for checking the setup:

```bash
python scripts/measure_complexity.py --cpu --out results/complexity_cpu.json
python scripts/run_synthetic_xai_benchmark.py --out results/synth_xai.json
```

The first reports parameters, FLOPs and latency — all fixed by the
architecture, so weights are irrelevant. The second plants a short event at a
known position in a synthetic sequence and checks whether each temporal
explanation method recovers it; it is the only setting in which "correct"
attribution is actually defined, so it is used as the sanity check for the
block-ablation method.

## Notes

- FLOPs come from the TF profiler, which counts a multiply-accumulate as two
  operations. Papers quoting MACs report roughly half these numbers; both are
  printed to avoid confusion.
- The Bi-LSTM head is `(64, 32)` units with `(64, 32)` dense layers. The first
  version of this work did not record the layer widths, so these were fixed
  when the code was rewritten and are what all reported numbers use.
- Clip length is 40 frames sampled at equal intervals over the whole video,
  resized to 128×128.

## Citation

```bibtex
@article{negre2026xaivd,
  title   = {Explainable Lightweight Video Violence Detection},
  author  = {Negre, Pablo and Alonso, Ricardo S. and Chamoso, Pablo and Prieto, Javier},
  journal = {Signal, Image and Video Processing},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT, see `LICENSE`. The datasets keep their own licences.
