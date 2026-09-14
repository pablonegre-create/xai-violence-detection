# Running on the BISITE cluster

## 0. Access

The login node moved in the last upgrade:

```
ssh <user>@hpc-bisite.usal.es      # 212.128.132.30
```

Only reachable from the university network or through the USAL VPN
(https://vpn.usal.es/). The old `bisite-hpc-vm-login.usal.es` / 212.128.132.66
no longer answers, so an SSH alias pointing there needs updating:

```
Host atos
    HostName hpc-bisite.usal.es
    User <user>
```

## 1. Get the repository onto the cluster

The repository is private, so a plain `git clone` will fail. Two options.

**With a token** (preferred — you can `git pull` later):

```bash
# on your laptop
gh auth token                       # copy the value

# on the cluster
cd $HOME
git clone https://<TOKEN>@github.com/ralorin/xai-violence-detection.git xai-vd
```

Do not leave the token in the remote URL afterwards:

```bash
cd ~/xai-vd
git remote set-url origin https://github.com/ralorin/xai-violence-detection.git
```

**By copy** (no token needed):

```bash
# from the laptop, inside R1/
scp -r code <user>@hpc-bisite.usal.es:~/xai-vd
```

If you copy from Windows rather than cloning, check the line endings before
submitting anything — `sbatch` rejects CRLF with *"Batch script contains DOS
line breaks"*:

```bash
cd ~/xai-vd
file slurm/*.sh                     # should say "ASCII text", not "with CRLF"
sed -i 's/\r$//' slurm/*.sh         # fix if needed
```

A clone is safe: `.gitattributes` forces LF on `.sh` and `.py`.

## 2. Environment

Home directories survived the reinstall, so miniconda may still be there:

```bash
ls ~/miniconda3/etc/profile.d/conda.sh && echo present || echo missing
```

If missing:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh      # accept the default path, answer yes to init
rm Miniconda3-latest-Linux-x86_64.sh
exit                                        # reconnect so conda is on PATH
```

Conda's package cache is configured to `/scratch/$USER/conda_pkgs`, but
`/scratch` currently has no per-user directories and is not writable, so
`conda create` fails with `NoWritablePkgsDirError`. Add a writable cache in
your home directory:

```bash
mkdir -p $HOME/.conda/pkgs
conda config --add pkgs_dirs $HOME/.conda/pkgs
```

Then:

```bash
conda create -n xai_vd python=3.10 -y
conda activate xai_vd
cd ~/xai-vd
pip install -r requirements.txt
```

### CUDA

The plain `tensorflow` wheel ships no CUDA on Linux, so out of the box it runs
on CPU and a 4 h job becomes days. Install the CUDA 12 wheels **from the login
node** — compute nodes have no outbound network and the download will fail
there:

```bash
# login node
conda activate xai_vd
pip install "tensorflow[and-cuda]==2.15.1"      # ~2 GB of nvidia-*-cu12 wheels
```

Do not use the `CUDA/13.3.0` and `cuDNN/9.23.0.39` environment modules for
this: TensorFlow 2.15 links against `libcudart.so.12` and `libcudnn.so.8`,
while those modules provide the `.13` and `.9` sonames. The GPU driver itself
is backward compatible, so the pip CUDA 12 runtime works on these H100s.

Then verify on a compute node, never on login:

```bash
srun --partition=short --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
     --pty bash -i
nvidia-smi                                      # driver visible?
conda activate xai_vd
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
exit                                            # frees the GPU
```

Expect one `PhysicalDevice(..., device_type='GPU')` entry, together with this
warning:

```
TensorFlow was not built with CUDA kernel binaries compatible with compute
capability 9.0. CUDA kernels will be jit-compiled from PTX, which could take
30 minutes or longer.
```

That is expected: the TF 2.15 wheel carries no sm_90 cubins, so on an H100
every kernel is compiled from PTX at first use, in every new process. Persist
the compiled kernels so only the first run pays for it — the job files already
export these, but set them in your interactive session too:

```bash
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296          # 4 GB, the maximum CUDA accepts
python scripts/warm_cuda_cache.py
```

The first call takes tens of minutes; run the script a second time and it
should finish in seconds. Do this once, from an interactive session, before
submitting anything — otherwise the first batch job burns that time out of its
own walltime, and `run_ablation.py`, which spawns one subprocess per variant,
would pay it repeatedly.

### Pretrained weights

Also because compute nodes have no network, Keras cannot download the ImageNet
backbone weights from inside a job. Populate the cache once from login:

```bash
python scripts/prefetch_weights.py
```

That writes `~/.keras/models/mobilenet_v2_weights_..._1.0_128_no_top.h5`, which
is in your home directory and therefore visible from the compute nodes. Jobs
01, 02 and 07 fail without it. Add `--backbones mobilenetv2 mobilenetv3small
efficientnetb0 resnet50 vgg19` if you intend to run the backbone comparison
with trained weights.

`requirements-keyframe.txt` (ultralytics, torch) is deliberately separate: no
job from 01 to 08 uses it, and it pulls a full CUDA stack whose major version
may not match the one TensorFlow wants. Install it in its own environment only
if you need the keyframe stage.

Large caches: `/scratch` is not writable, so keep them in home, which is on
Lustre:

```bash
echo 'export HF_HOME=$HOME/.cache/huggingface' >> ~/.bashrc
```

## 3. Datasets

Not redistributed with the code. Roughly 8 GB in total.

```bash
export DATA_ROOT=$HOME/data          # /scratch is not writable
mkdir -p $DATA_ROOT && cd $DATA_ROOT

pip install kaggle                   # needs ~/.kaggle/kaggle.json from your account
kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset -p rlvs --unzip
kaggle datasets download -d vulamnguyen/rwf2000 -p rwf2000 --unzip
kaggle datasets download -d yassershrief/hockey-fight-vidoes -p hockey --unzip
kaggle datasets download -d naveenk903/movies-fight-detection-dataset -p movies --unzip
# Violent Flows: request from https://www.openu.ac.il/home/hassner/data/violentflows/
```

Check the loaders find them before queueing anything — the Kaggle mirrors nest
the class folders differently from each other:

```bash
cd ~/xai-vd
conda activate xai_vd
python -c "
from src.data.datasets import index_flat
for d in ['rlvs','rwf2000','hockey','movies']:
    try:
        it = index_flat('$DATA_ROOT/'+d)
        print('%-10s %5d clips, %d violent' % (d, len(it), sum(y for _,y in it)))
    except Exception as e:
        print('%-10s FAILED: %s' % (d, e))
"
```

Expected roughly: rlvs 2000, rwf2000 2000, hockey 1000, movies 200, all
balanced. If a dataset fails, the class directory name is not in the alias list
— add it to `VIOLENT_DIRS` / `PEACEFUL_DIRS` in `src/data/datasets.py`.

RWF-2000 keeps its own `train/` and `val/` folders; `index_rwf2000` handles it.

## 4. Partitions

The old `gpu-small` / `gpu-large` / `all` layout is gone.

| partition | max walltime |
|---|---|
| `short` (default) | 4 h |
| `medium` | 1 day |
| `long` | 7 days |

All sit on `cn001` (224 logical cores, ~4 TB RAM, 8× H100 80 GB).
`--gres=gpu:N` is still required. The job files in `slurm/` already use this
layout: `short` for head training and complexity, `medium` for fine-tuning,
feature extraction, ablation, cross-dataset and the XAI comparison, `long` only
for robustness.

## 5. Submit

Jobs have real dependencies: 02 needs the checkpoint from 01, everything else
needs the feature caches from 02, and 06/07 need the head checkpoints from 03.
Chain them with `--dependency` rather than watching the queue:

```bash
cd ~/xai-vd
mkdir -p logs ckpt features results
export DATA_ROOT=$HOME/data

# 1. fine-tune the backbone (~4 h)
FT=$(sbatch --parsable slurm/01_finetune_backbone.sh 5)

# 2. cache descriptors, one job per dataset (~1 h each, run in parallel)
FEAT=""
for d in rlvs rwf2000 hockey movies violentflows; do
    J=$(sbatch --parsable --dependency=afterok:$FT slurm/02_extract_features.sh $d)
    FEAT="$FEAT:$J"
done

# 3. train the head (minutes) -> writes ckpt/head_rlvs_seed*.h5
TR=$(sbatch --parsable --dependency=afterok${FEAT} slurm/03_train_head.sh rlvs)

# 4-8. the rest
sbatch --dependency=afterok:$TR   slurm/04_ablation.sh rlvs        # ~1 h
sbatch --dependency=afterok${FEAT} slurm/05_cross_dataset.sh       # ~2 h
sbatch --dependency=afterok:$TR   slurm/06_xai_eval.sh rlvs        # ~6 h
sbatch --dependency=afterok:$TR   slurm/07_robustness.sh rlvs      # ~20 h
sbatch                            slurm/08_complexity.sh           # minutes
```

If you would rather go step by step, drop the `--dependency` flags and submit
each one once the previous has finished.

To also get per-dataset numbers for the other four, repeat job 03 for each:

```bash
for d in rwf2000 hockey movies violentflows; do
    sbatch slurm/03_train_head.sh $d
done
```

## 6. Monitor

```bash
squeue -u $USER
squeue -u $USER -o "%6i %8j %3t %10M %10L %R"     # id, name, state, used, left, reason
tail -f logs/xai_vd_train_<jobid>.out
scancel <jobid>
sacct -j <jobid> -o JobID,JobName,State,Elapsed,ReqMem,MaxRSS
```

`PENDING` with reason `Dependency` is expected while the chain waits.

## 7. Which job fills which table

| Job | Output file | Fills |
|---|---|---|
| 03 | `results/main_<ds>.json` | Table 1 left (acc/F1/AUC/CI) |
| 04 | `results/ablation_rlvs.json` | Table 2 (Holm-corrected p) |
| 05 | `results/cross_dataset.json` | Table 1 right (transfer matrix) |
| 06 | `results/xai_temporal_rlvs.json`, `results/synthetic_xai_benchmark.json` | Table 5, ESM S5/S6 |
| 07 | `results/robustness_rlvs.json` | Table 6, ESM S4/S5 |
| 08 | `results/complexity_h100.json` | Table 4, H100 column |

The block-size study (ESM Table S3) is not a job; run it once the head exists:

```bash
python - <<'PY'
import json, numpy as np
from scripts.train import load_split
from src.models.build import build_temporal_head
from src.xai.frame_importance import sweep_block_sizes

X, y, split, _ = load_split('features/rlvs')
te = np.where(split == 'test')[0][:100]
head = build_temporal_head(X.shape[1], X.shape[2])
head.load_weights('ckpt/head_rlvs_seed0.h5')
out = sweep_block_sizes([X[i] for i in te], head, blocks=(1, 2, 5, 10))
json.dump(out, open('results/block_size.json', 'w'), indent=2)
for r in out:
    print(r)
PY
```

## 8. Bring the results back

Only the JSON files are needed to fill the tables; they are small.

```bash
# from the laptop
scp -r <user>@hpc-bisite.usal.es:~/xai-vd/results ./R1/results_hpc
```

Checkpoints and feature caches stay on the cluster — the caches are several GB.

## Gotchas

- `sbatch` runs a non-interactive shell, which reads neither `.bashrc` nor
  `.bash_profile`. Every job file therefore sources conda explicitly before
  activating the env; the first lines are not redundant.
- Compute nodes have no outbound network: pip, Kaggle and Keras weight
  downloads all fail inside a job. Do every download from the login node.
- Slurm is configured with `TMPDIR=/scratch/$USER/conda_tmp`, which does not
  exist; it falls back to `/tmp` with a warning at the top of every job. It is
  harmless, but `export TMPDIR=$HOME/tmp` silences it if anything trips on it.
- Anything written to a compute node's local disk disappears when the job ends.
  Write to `$HOME`; `/scratch` has no per-user directories at present.
- `--partition=all` no longer exists; jobs referring to it are rejected.
- Video decoding, not the GPU, is the bottleneck in job 02 — give it the CPUs.
