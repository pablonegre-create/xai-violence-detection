"""Efficiency table: params, FLOPs, latency and FPS for the proposed model
and for the CNN+Bi-LSTM baselines it is compared against.

Weights are irrelevant for all three quantities, so this runs anywhere and
does not need a trained checkpoint. Device is whatever TF picks up; pass
--cpu to force CPU so the numbers describe a low-resource deployment.

    python scripts/measure_complexity.py --cpu --out results/complexity_cpu.json
"""

import argparse
import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKBONES = ["mobilenetv2", "mobilenetv3small", "efficientnetb0",
             "resnet50", "vgg19"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="results/complexity.json")
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf
    from src.models.build import build_backbone, build_temporal_head, FEATURE_DIMS
    from src.eval import complexity as cx

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    env = {
        "tensorflow": tf.__version__,
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "n_frames": args.frames,
        "input_size": args.input_size,
    }
    print(json.dumps(env, indent=2))

    rows = []
    for name in args.backbones:
        print("\n--- %s ---" % name)
        tf.keras.backend.clear_session()

        cnn = build_backbone(name, input_size=args.input_size, weights=None)
        head = build_temporal_head(n_frames=args.frames,
                                   feature_dim=FEATURE_DIMS[name])

        p_cnn = cx.count_params(cnn)
        p_head = cx.count_params(head)

        f_cnn = cx.count_flops(cnn)
        f_head = cx.count_flops(head)
        f_clip = cx.total_clip_flops(f_cnn, f_head, args.frames)

        t_cnn = cx.time_model(cnn, n_runs=args.runs)
        t_head = cx.time_model(head, n_runs=args.runs)
        # Streaming case: frames arrive one at a time, worst case for latency.
        t_clip_stream_ms = args.frames * t_cnn["median_ms"] + t_head["median_ms"]
        # Batched case: the whole clip is pushed through the CNN at once, which
        # is what the offline pipeline actually does.
        t_cnn_batch = cx.time_model(cnn, n_runs=max(5, args.runs // 5),
                                    batch=args.frames)
        t_clip_batch_ms = t_cnn_batch["median_ms"] + t_head["median_ms"]

        row = {
            "backbone": name,
            "params_cnn_M": p_cnn["total"] / 1e6,
            "params_head_M": p_head["total"] / 1e6,
            "params_total_M": (p_cnn["total"] + p_head["total"]) / 1e6,
            "size_fp32_MB": p_cnn["fp32_mb"] + p_head["fp32_mb"],
            "flops_frame_G": f_cnn / 1e9,
            "macs_frame_G": f_cnn / 2e9,
            "flops_head_G": f_head / 1e9,
            "flops_clip_G": f_clip / 1e9,
            "macs_clip_G": f_clip / 2e9,
            "latency_frame_ms": t_cnn["median_ms"],
            "latency_frame_std_ms": t_cnn["std_ms"],
            "latency_head_ms": t_head["median_ms"],
            "latency_clip_stream_ms": t_clip_stream_ms,
            "latency_clip_batch_ms": t_clip_batch_ms,
            "fps_stream": args.frames / (t_clip_stream_ms / 1000.0),
            "fps_batch": args.frames / (t_clip_batch_ms / 1000.0),
            "throughput_frames_per_s": 1000.0 / t_cnn["median_ms"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"env": env, "rows": rows}, fh, indent=2)
    print("\nwrote", args.out)

    # quick LaTeX dump so the table can be pasted straight into the paper
    print("\n% --- LaTeX ---")
    for r in rows:
        print("%s & %.2f & %.1f & %.2f & %.1f & %.0f & %.1f \\\\" % (
            r["backbone"], r["params_total_M"], r["size_fp32_MB"],
            r["flops_clip_G"], r["latency_frame_ms"],
            r["latency_clip_batch_ms"], r["fps_batch"]))


if __name__ == "__main__":
    main()
