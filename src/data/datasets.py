"""Dataset discovery and clip loading.

Every dataset here is trimmed binary clip classification: a short video is
either violent or not. The loaders only need a root directory laid out as

    root/
      Violence/    (or fight/, violent/, ...)
      NonViolence/ (or nonfight/, normal/, ...)

The mirrors are not consistent about this, so index_flat() tries, in order:
class directories at the top; the same behind one or more wrapper directories;
numbered cross-validation folds with the classes one level deeper (Violent
Flows); the class encoded in the file name with every clip in one directory
(Hockey Fights, fi*.avi versus no*.avi); and finally a search for class folders
anywhere under the root. RWF-2000 ships its own train/val split and is handled
by index_rwf2000().

UCF-Crime and XD-Violence are deliberately *not* here: they are untrimmed,
weakly labelled anomaly-detection benchmarks scored with AUC/AP, so a trimmed
clip classifier cannot be run on them without a multiple-instance-learning
head. See docs/datasets.md.
"""

import os
import glob

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Bare "0"/"1" are deliberately absent: Violent Flows is sometimes distributed
# as five numbered cross-validation folds, and treating "1" as a class name
# would silently label a whole fold violent.
VIOLENT_DIRS = {"violence", "fight", "fights", "violent", "vio"}
PEACEFUL_DIRS = {"nonviolence", "nofight", "nonfight", "non-violence",
                 "nonviolent", "normal", "noviolence", "noviolent"}

VIDEO_EXT = ("*.mp4", "*.avi", "*.mpg", "*.mpeg", "*.mov", "*.mkv")

# Some datasets put every clip in a single directory and encode the class in
# the file name instead. Hockey Fights is the canonical case: fi1_xvid.avi
# versus no1_xvid.avi. A pair is only accepted when the two prefixes partition
# the whole set, so a coincidental prefix cannot mislabel anything.
FILENAME_PREFIXES = [
    ("fi", "no"),            # Hockey Fights
    ("fight", "nofight"),
    ("v_", "nv_"),
    ("violence", "nonviolence"),
]


def _label_of(dirname):
    d = dirname.lower().replace("_", "").replace(" ", "").replace("-", "")
    if d in VIOLENT_DIRS or d.startswith("fi"):
        return 1
    if d in PEACEFUL_DIRS or d.startswith("no"):
        return 0
    return None


def _descend(root, wanted):
    """Follow single-child wrapper directories until `wanted` appears.

    Kaggle mirrors are inconsistent about how deeply they nest: one ships
    `Violence/` at the top, another wraps everything in `RWF-2000/`, a third
    adds a duplicate of the dataset name. Rather than make the caller guess,
    walk down while there is exactly one plausible way to go.
    """
    cur = root
    for _ in range(4):
        if any(os.path.isdir(os.path.join(cur, w)) for w in wanted):
            return cur
        subs = [s for s in sorted(os.listdir(cur))
                if os.path.isdir(os.path.join(cur, s))]
        if len(subs) != 1:
            break
        cur = os.path.join(cur, subs[0])
    return root


def _class_dirs(root):
    """[(path, label)] for the immediate class subdirectories of root."""
    out = []
    for sub in sorted(os.listdir(root)):
        full = os.path.join(root, sub)
        if os.path.isdir(full):
            label = _label_of(sub)
            if label is not None:
                out.append((full, label))
    return out


def _videos_under(d):
    found = []
    for ext in VIDEO_EXT:
        found += glob.glob(os.path.join(d, "**", ext), recursive=True)
    return sorted(found)


def index_folds(root):
    """Official cross-validation folds, when the dataset ships with them.

    Violent Flows is distributed as five numbered directories, each holding the
    two class folders. Returns {fold_name: [(path, label), ...]} or None if the
    layout is not folded.
    """
    root = _descend(root, VIOLENT_DIRS | PEACEFUL_DIRS)
    if _class_dirs(root):
        return None

    folds = {}
    for sub in sorted(os.listdir(root)):
        full = os.path.join(root, sub)
        if not os.path.isdir(full):
            continue
        pairs = _class_dirs(full)
        if not pairs:
            return None
        items = [(p, lab) for d, lab in pairs for p in _videos_under(d)]
        if items:
            folds[sub] = items
    return folds or None


def index_by_filename(root):
    """Label by file-name prefix, for datasets with no class directories.

    Returns [] unless one of FILENAME_PREFIXES partitions every video found
    under root. Partial coverage is rejected rather than guessed at: a clip
    labelled by accident is worse than a loader that refuses to run.
    """
    vids = _videos_under(root)
    if not vids:
        return []

    names = [os.path.basename(p).lower() for p in vids]
    for vpre, ppre in FILENAME_PREFIXES:
        v = [n.startswith(vpre) for n in names]
        p = [n.startswith(ppre) for n in names]
        if any(a and b for a, b in zip(v, p)):
            continue                     # ambiguous, prefixes overlap
        if not (any(v) and any(p)):
            continue
        if all(a or b for a, b in zip(v, p)):
            return [(path, 1 if a else 0) for path, a, b in zip(vids, v, p)]
    return []


def _search_class_root(root, max_depth=4):
    """Shallowest directory anywhere under root that holds class folders."""
    from collections import deque

    q = deque([(root, 0)])
    while q:
        cur, d = q.popleft()
        if _class_dirs(cur):
            return cur
        if d >= max_depth:
            continue
        try:
            for s in sorted(os.listdir(cur)):
                full = os.path.join(cur, s)
                if os.path.isdir(full):
                    q.append((full, d + 1))
        except OSError:
            continue
    return None


def index_flat(root):
    """Walk a two-class directory tree and return [(path, label), ...]."""
    base = _descend(root, VIOLENT_DIRS | PEACEFUL_DIRS)

    items = [(p, lab) for d, lab in _class_dirs(base) for p in _videos_under(d)]
    if items:
        return items

    # distributed as numbered folds, classes one level deeper; pool them.
    # The official assignment stays available through index_folds().
    folds = index_folds(root)
    if folds:
        return [x for f in sorted(folds) for x in folds[f]]

    # every clip in one directory, class encoded in the file name
    items = index_by_filename(base)
    if items:
        return items

    # last resort: the class folders are nested behind something that is not a
    # single-child chain, so go looking for them
    found = _search_class_root(root)
    if found:
        items = [(p, lab) for d, lab in _class_dirs(found)
                 for p in _videos_under(d)]
        if items:
            return items

    raise RuntimeError(
        "no videos found under %s -- run scripts/inspect_dataset.py on it; "
        "if the class directories have unusual names, add them to "
        "VIOLENT_DIRS / PEACEFUL_DIRS in this module" % root)


def index_rwf2000(root):
    """RWF-2000 ships with its own split, so honour it."""
    root = _descend(root, {"train", "val", "test"})

    out = {}
    for split, name in (("train", "train"), ("test", "val")):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            d = os.path.join(root, split)
        if not os.path.isdir(d):
            raise RuntimeError("no %s/ split under %s" % (name, root))
        out[split] = index_flat(d)
    return out


def load_clip(path, n_frames=40, size=128, sampling="uniform"):
    """Read a video and return (n_frames, size, size, 3) float32 in [0, 1].

    Frames are taken at equal intervals over the whole clip, which is what the
    original implementation did and keeps short and long clips comparable.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # some AVIs report 0; fall back to reading everything
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            raise RuntimeError("unreadable video: %s" % path)
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        sel = [frames[i] for i in idx]
    else:
        if sampling == "uniform":
            idx = np.linspace(0, total - 1, n_frames).astype(int)
        else:
            start = np.random.randint(0, max(1, total - n_frames))
            idx = np.arange(start, start + n_frames).clip(0, total - 1)
        sel = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                f = sel[-1] if sel else np.zeros((size, size, 3), np.uint8)
            sel.append(f)
        cap.release()

    out = np.stack([cv2.resize(f, (size, size)) for f in sel])
    out = cv2.cvtColor(out.reshape(-1, size, 3), cv2.COLOR_BGR2RGB)
    return (out.reshape(n_frames, size, size, 3).astype("float32") / 255.0)


def stratified_split(items, fractions=(0.7, 0.1, 0.2), seed=0):
    """Split preserving class balance. Returns (train, val, test) lists."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for p, y in items:
        by_class.setdefault(y, []).append((p, y))

    train, val, test = [], [], []
    for y, lst in by_class.items():
        lst = list(lst)
        rng.shuffle(lst)
        n = len(lst)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        train += lst[:n_tr]
        val += lst[n_tr:n_tr + n_va]
        test += lst[n_tr + n_va:]

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def kfold_split(items, k=5, seed=0):
    """5-fold CV, the convention for Hockey Fights and Violent Flows."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for p, y in items:
        by_class.setdefault(y, []).append((p, y))

    folds = [[] for _ in range(k)]
    for y, lst in by_class.items():
        lst = list(lst)
        rng.shuffle(lst)
        for i, item in enumerate(lst):
            folds[i % k].append(item)

    for i in range(k):
        test = folds[i]
        train = [x for j in range(k) if j != i for x in folds[j]]
        yield train, test


DATASETS = {
    "rlvs": dict(loader=index_flat, protocol="holdout",
                 note="Real Life Violence Situations, Soliman et al. 2019"),
    "hockey": dict(loader=index_flat, protocol="5fold",
                   note="Hockey Fights, Nievas et al. 2011"),
    "movies": dict(loader=index_flat, protocol="5fold",
                   note="Movies Fight, Nievas et al. 2011"),
    "violentflows": dict(loader=index_flat, protocol="5fold",
                         note="Violent Flows, Hassner et al. 2012"),
    "rwf2000": dict(loader=index_rwf2000, protocol="official",
                    note="RWF-2000, Cheng et al. 2021"),
}
