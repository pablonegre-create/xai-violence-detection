# Running on the BISITE cluster

The login node moved in the last upgrade:

```
ssh <user>@hpc-bisite.usal.es      # 212.128.132.30
```

Only reachable from the university network or through the USAL VPN
(https://vpn.usal.es/).

## Partitions

The old `gpu-small` / `gpu-large` / `all` layout is gone. Current queues:

| partition | max walltime |
|---|---|
| `short` (default) | 4 h |
| `medium` | 1 day |
| `long` | 7 days |

All of them sit on `cn001` (224 logical cores, ~4 TB RAM, 8x H100 80 GB).
`--gres=gpu:N` is still required.

The job files in `slurm/` are already on this layout: `short` for head
training and the complexity measurements, `medium` for fine-tuning, feature
extraction, ablation, cross-dataset and the XAI comparison, `long` only for
the robustness sweep.

## Environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh          # accept the default path
# reconnect, then
conda create -n xai_vd python=3.10 -y
conda activate xai_vd
pip install -r requirements.txt
```

`sbatch` runs a non-interactive shell, which reads neither `.bashrc` nor
`.bash_profile`, so every job file sources conda explicitly before activating
the env. That is why the first lines of each script look redundant — they are
not.

Keep the Hugging Face cache off `/home`:

```bash
mkdir -p /scratch/$USER/hf_cache
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
```

## Order

```bash
sbatch slurm/01_finetune_backbone.sh 5
# wait for the checkpoint, then
for d in rlvs rwf2000 hockey movies violentflows; do
    sbatch slurm/02_extract_features.sh $d
done
# once the caches exist everything else is independent
sbatch slurm/03_train_head.sh rlvs
sbatch slurm/04_ablation.sh rlvs
sbatch slurm/05_cross_dataset.sh
sbatch slurm/06_xai_eval.sh rlvs
sbatch slurm/07_robustness.sh rlvs
sbatch slurm/08_complexity.sh
```

Rough cost on one H100: fine-tuning ~4 h, feature extraction ~1 h per dataset
(video decoding is the bottleneck, not the GPU — give it the 8 CPUs), head
training minutes, ablation ~1 h, cross-dataset ~2 h, XAI comparison ~6 h
(dominated by the Shapley baselines, which is the point), robustness ~20 h.

## Gotchas

- Edit the `.sh` files on Linux or set the line endings to LF. `sbatch` rejects
  CRLF with `Batch script contains DOS line breaks`.
- `--partition=all` no longer exists; jobs referring to it will be rejected.
- Anything written to a compute node's local disk disappears when the job
  ends. Write to `$HOME` or `/scratch/$USER`.
