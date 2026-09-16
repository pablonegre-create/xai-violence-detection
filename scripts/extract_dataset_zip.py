"""Extract a dataset archive, shortening names the filesystem cannot take.

The RWF-2000 mirror on Kaggle contains entries whose names were mangled on the
way in (Cyrillic re-encoded through CP437), and several exceed the 255-byte
limit Linux puts on a single path component. `kaggle --unzip` aborts partway
through with `OSError: [Errno 36] File name too long`, leaving the dataset half
extracted.

Only the directory structure carries meaning here - the class is the parent
folder, not the file name - so over-long basenames are replaced by a short
hash. The mapping is written to `_renamed.csv` next to the output so nothing is
lost.

    python scripts/extract_dataset_zip.py ~/data/rwf2000/rwf2000.zip -o ~/data/rwf2000
"""

import argparse
import csv
import hashlib
import os
import sys
import zipfile

MAX_COMPONENT = 200          # bytes; real limit is 255, leave headroom


def shorten(component, keep_ext=True):
    """Replace an over-long path component with a hashed stand-in."""
    raw = component.encode("utf-8", errors="surrogateescape")
    if len(raw) <= MAX_COMPONENT:
        return component, False

    stem, ext = os.path.splitext(component)
    if not keep_ext or len(ext) > 12:
        ext = ""
    digest = hashlib.sha1(raw).hexdigest()[:16]
    # keep a readable prefix so files stay sortable by their original order
    prefix = stem.encode("utf-8", errors="surrogateescape")[:40]
    prefix = prefix.decode("utf-8", errors="ignore")
    prefix = "".join(c for c in prefix if c.isalnum() or c in "-_")
    return "%s_%s%s" % (prefix or "clip", digest, ext), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not zipfile.is_zipfile(args.archive):
        print("not a zip file:", args.archive)
        return 1

    renamed, extracted, skipped = [], 0, 0
    os.makedirs(args.out, exist_ok=True)

    with zipfile.ZipFile(args.archive) as z:
        members = z.infolist()
        print("%d entries in %s" % (len(members), os.path.basename(args.archive)))

        for info in members:
            if info.is_dir():
                continue

            parts = [p for p in info.filename.split("/") if p not in ("", ".", "..")]
            if not parts:
                continue

            new_parts, changed = [], False
            for i, p in enumerate(parts):
                np_, ch = shorten(p, keep_ext=(i == len(parts) - 1))
                new_parts.append(np_)
                changed |= ch

            target = os.path.join(args.out, *new_parts)
            if changed:
                renamed.append((info.filename, os.path.relpath(target, args.out)))

            if args.dry_run:
                continue

            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                skipped += 1
                continue
            try:
                with z.open(info) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted += 1
            except OSError as e:
                print("  could not write %s: %s" % (target, e))

            if (extracted + skipped) % 500 == 0:
                print("  %d done" % (extracted + skipped), flush=True)

    print("\nextracted %d, already present %d, renamed %d"
          % (extracted, skipped, len(renamed)))

    if renamed and not args.dry_run:
        path = os.path.join(args.out, "_renamed.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["original", "written_as"])
            w.writerows(renamed)
        print("rename map:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
