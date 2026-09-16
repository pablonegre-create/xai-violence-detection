"""Show how a dataset directory is laid out and what the loader makes of it.

The Kaggle and academic mirrors disagree about nesting depth and class folder
names, and a wrong guess is silent: the loader would happily return a balanced
set with the labels inverted, or with a cross-validation fold treated as a
class. Run this before the first feature extraction of any new dataset.

    python scripts/inspect_dataset.py ~/data/violentflows
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VIDEO_EXT = (".mp4", ".avi", ".mpg", ".mpeg", ".mov", ".mkv")


def tree(root, max_depth=3, max_entries=12):
    print("directory tree (depth %d):" % max_depth)
    root = os.path.abspath(root)
    for cur, dirs, files in os.walk(root):
        depth = cur[len(root):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs.sort()
        vids = [f for f in files if f.lower().endswith(VIDEO_EXT)]
        indent = "  " * depth
        name = os.path.basename(cur) or root
        note = "  <- %d video files" % len(vids) if vids else ""
        print("%s%s/%s" % (indent, name, note))
        if vids and depth <= max_depth:
            for f in sorted(vids)[:3]:
                print("%s    e.g. %s" % (indent, f[:70]))
        if len(dirs) > max_entries:
            print("%s  ... %d more directories" % (indent, len(dirs) - max_entries))
            dirs[:] = dirs[:max_entries]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print("not a directory:", args.root)
        return 1

    from src.data.datasets import (_descend, _label_of, index_flat,
                                   index_rwf2000, VIOLENT_DIRS, PEACEFUL_DIRS)

    tree(args.root, args.depth)

    print("\nwhat the loader sees")
    landed = _descend(args.root, VIOLENT_DIRS | PEACEFUL_DIRS |
                      {"Violence", "NonViolence", "fight", "nofight"})
    if os.path.abspath(landed) != os.path.abspath(args.root):
        print("  descends into: %s" % landed)

    subs = [s for s in sorted(os.listdir(landed))
            if os.path.isdir(os.path.join(landed, s))]
    print("  top-level directories: %s" % (", ".join(subs) or "(none)"))
    for s in subs:
        lab = _label_of(s)
        verdict = {1: "VIOLENT", 0: "non-violent", None: "ignored"}[lab]
        print("    %-24s -> %s" % (s, verdict))

    unmatched = [s for s in subs if _label_of(s) is None]
    if unmatched:
        print("\n  %d directory name(s) not recognised. If they are classes,"
              % len(unmatched))
        print("  add them to VIOLENT_DIRS / PEACEFUL_DIRS in"
              " src/data/datasets.py.")
        if all(s.strip().isdigit() for s in unmatched):
            print("  They look like cross-validation folds. Violent Flows is"
                  " distributed\n  that way; the classes are one level deeper.")

    print("\nindex_flat():")
    try:
        items = index_flat(args.root)
        c = Counter(y for _, y in items)
        print("  %d clips  (violent %d, non-violent %d)"
              % (len(items), c[1], c[0]))
        if c[1] == 0 or c[0] == 0:
            print("  WARNING: one class is empty, the layout is not what the"
                  " loader expects")
        if c[1] != c[0]:
            print("  note: the classes are not balanced (%d vs %d). All five"
                  % (c[1], c[0]))
            print("  datasets used here should be, so check for a stray or"
                  " duplicated file.")

        # where did each clip come from, and are any duplicated
        by_dir = Counter(os.path.dirname(os.path.relpath(p, args.root))
                         for p, _ in items)
        print("  clips per source directory:")
        for d, n in sorted(by_dir.items()):
            print("    %-52s %4d" % ((d or ".")[:52], n))

        names = Counter(os.path.basename(p) for p, _ in items)
        dupes = [n for n, k in names.items() if k > 1]
        if dupes:
            print("  %d file name(s) appear more than once:" % len(dupes))
            for n in dupes[:5]:
                print("    %s x%d" % (n[:60], names[n]))

        for p, y in items[:3]:
            print("    e.g. [%d] %s" % (y, os.path.relpath(p, args.root)[:76]))
    except Exception as e:
        print("  failed: %s" % e)

    print("\nindex_rwf2000():")
    try:
        sp = index_rwf2000(args.root)
        for k, v in sp.items():
            c = Counter(y for _, y in v)
            print("  %-6s %d clips (violent %d, non-violent %d)"
                  % (k, len(v), c[1], c[0]))
    except Exception as e:
        print("  not an official-split layout (%s)" % str(e)[:60])

    return 0


if __name__ == "__main__":
    sys.exit(main())
