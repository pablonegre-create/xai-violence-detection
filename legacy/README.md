# Original scripts

These are the scripts as they were written while the method was being worked
out, kept unchanged (comments in Spanish, absolute Windows paths and all).
They are here for provenance, not to be run — the maintained versions live in
`src/` and `scripts/`.

| file | what it did | where it went |
|---|---|---|
| `get_vgg19_extracted_features.py` | cached VGG19 `block5_pool` features per clip | `scripts/extract_features.py`, backbone now configurable |
| `iteracion_5f.py` | first version of the frame-removal loop, 5 frames at a time, plus the animated importance plot | `src/xai/frame_importance.py` |
| `gradcam_definitive.py` | Grad-CAM per conv block, overlays, grid video | `src/xai/gradcam.py` |
| `frame_diff.py` | mean absolute difference between consecutive frames | `src/keyframe/frame_difference.py` |
| `yolov8.py` | YOLOv8 person tracking with per-ID colours | `src/keyframe/yolo_person.py` (YOLOv12) |

Two things changed substantively in the rewrite rather than just being tidied:

- The backbone moved from VGG19 to MobileNetV2. VGG19 needs 6.4 GMACs per
  frame at 128×128 against MobileNetV2's 0.098, which is what the
  "lightweight" claim rests on.
- The importance loop became configurable in block size and stride, and gained
  the additive form (`importance_attribution`) that returns a probability drop
  instead of a ratio, so it can be compared against Shapley-style attributions
  on the same scale.

The model reported in the paper was trained separately and its training code
is not ours to release; `scripts/finetune_backbone.py` and `scripts/train.py`
reproduce the same architecture and procedure from scratch.
