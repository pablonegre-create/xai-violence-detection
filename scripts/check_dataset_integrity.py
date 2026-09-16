"""Find duplicated and unreadable clips before spending GPU hours on them.

The loader's cheap guard compares file name and size, which catches a mirror
that ships the whole tree twice. It does not catch the other failure these
archives have: the same clip stored under two different names. The Movies Fight
mirror does exactly that in both classes, which would put identical clips in
different cross-validation folds.

Grouping by size first means only files that could possibly match are ever
read, so this costs seconds rather than a full pass over the dataset.

    python scripts/check_dataset_integrity.py ~/data/movies
    python scripts/check_dataset_integrity.py ~/data/rlvs --delete
"""

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHUNK = 1 << 20          # 1 MB is plenty to separate distinct videos


def partial_hash(path, nbytes=CHUNK):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--delete", action="store_true",
                    help="remove the redundant copies, keeping the first by "
                         "sorted path")
    ap.add_argument("--check-readable", action="store_true",
                    help="also open every clip with OpenCV (slow)")
    args = ap.parse_args()

    from src.data.datasets import index_flat

    try:
        items = index_flat(args.root, allow_duplicates=True)
    except Exception as e:
        print("could not index %s: %s" % (args.root, e))
        return 1

    c = Counter(y for _, y in items)
    print("%d clips indexed (violent %d, non-violent %d)"
          % (len(items), c[1], c[0]))

    # group by size, then hash only within groups that could collide
    by_size = defaultdict(list)
    for p, lab in items:
        try:
            by_size[os.path.getsize(p)].append((p, lab))
        except OSError as e:
            print("  unreadable: %s (%s)" % (p, e))

    groups = defaultdict(list)
    n_hashed = 0
    for size, entries in by_size.items():
        if len(entries) < 2:
            continue
        for p, lab in entries:
            try:
                groups[(size, partial_hash(p))].append((p, lab))
                n_hashed += 1
            except OSError as e:
                print("  unreadable: %s (%s)" % (p, e))

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    print("hashed %d candidate files, %d duplicate group(s)"
          % (n_hashed, len(dupes)))

    removable, cross_class = [], 0
    for _, entries in sorted(dupes.items()):
        entries.sort()
        labels = {lab for _, lab in entries}
        marker = ""
        if len(labels) > 1:
            cross_class += 1
            marker = "   <-- SAME CLIP IN BOTH CLASSES"
        print("\n  duplicate group (%d copies)%s" % (len(entries), marker))
        for p, lab in entries:
            print("    [%d] %s" % (lab, os.path.relpath(p, args.root)))
        removable += [p for p, _ in entries[1:]]

    if cross_class:
        print("\n%d group(s) span both classes. Those are label errors in the"
              " mirror, not\nmere duplicates - inspect them before deleting"
              " anything." % cross_class)

    if not dupes:
        print("\nno duplicates. unique clips: %d" % len(items))
    else:
        uniq = len(items) - len(removable)
        print("\n%d redundant file(s); %d unique clips remain"
              % (len(removable), uniq))
        if args.delete and not cross_class:
            for p in removable:
                os.remove(p)
            print("deleted %d file(s)" % len(removable))
        elif args.delete:
            print("refusing to delete while groups span both classes")
        else:
            print("re-run with --delete to remove them")

    if args.check_readable:
        import cv2
        bad = 0
        for i, (p, _) in enumerate(items):
            cap = cv2.VideoCapture(p)
            ok, _ = cap.read()
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if not ok or n <= 0:
                print("  not decodable: %s" % os.path.relpath(p, args.root))
                bad += 1
            if (i + 1) % 500 == 0:
                print("  checked %d/%d" % (i + 1, len(items)), flush=True)
        print("undecodable clips: %d" % bad)

    return 0


if __name__ == "__main__":
    sys.exit(main())
