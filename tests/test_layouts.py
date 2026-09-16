"""Exercise index_flat against every directory layout the mirrors use."""
import os, shutil, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.data.datasets import index_flat, index_folds, index_rwf2000

BASE = tempfile.mkdtemp(prefix="layouts_")


def mk(paths):
    """paths: list of relative file paths to create as empty .avi"""
    root = tempfile.mkdtemp(dir=BASE)
    for p in paths:
        full = os.path.join(root, p.replace("/", os.sep))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"x")
    return root


CASES = []

# 1. plain two-class at the top
CASES.append(("plano", mk(
    ["Violence/a.avi", "Violence/b.avi", "NonViolence/c.avi", "NonViolence/d.avi"]
), 4, 2))

# 2. one wrapper directory
CASES.append(("un envoltorio", mk(
    ["Wrap/Violence/a.avi", "Wrap/NonViolence/b.avi"]
), 2, 1))

# 3. two nested wrappers (RLVS-style)
CASES.append(("dos envoltorios", mk(
    ["real life violence situations/Real Life Violence Dataset/Violence/a.avi",
     "real life violence situations/Real Life Violence Dataset/NonViolence/b.avi"]
), 2, 1))

# 4. wrapper plus a stray sibling directory -> not a single-child chain
CASES.append(("envoltorio + hermano suelto", mk(
    ["Data/Violence/a.avi", "Data/NonViolence/b.avi", "extras/readme/none.txt"]
), 2, 1))

# 5. Hockey: one directory, class in the file name
CASES.append(("nombre de fichero (hockey)", mk(
    ["HockeyFights/fi1_xvid.avi", "HockeyFights/fi2_xvid.avi",
     "HockeyFights/no1_xvid.avi", "HockeyFights/no2_xvid.avi",
     "HockeyFights/no3_xvid.avi"]
), 5, 2))

# 6. Hockey with no wrapper at all
CASES.append(("nombre de fichero sin envoltorio", mk(
    ["fi1.avi", "fi2.avi", "no1.avi", "no2.avi"]
), 4, 2))

# 7. numbered CV folds (Violent Flows)
# real fold distributions have unique clip names across folds
CASES.append(("folds numerados", mk(
    ["%d/%s/%s_f%d_%d.avi" % (f, c, c, f, i)
     for f in (1, 2, 3) for c in ("Violence", "NonViolence") for i in (1, 2)]
), 12, 6))

# 8. spanish-ish class names (Movies mirror)
CASES.append(("fights / noFights", mk(
    ["Peliculas/fights/f1.avi", "Peliculas/fights/f2.avi",
     "Peliculas/noFights/n1.avi", "Peliculas/noFights/n2.avi"]
), 4, 2))

# 9. mixed extensions and nested subfolders inside a class
CASES.append(("subcarpetas dentro de la clase", mk(
    ["Violence/g1/a.mp4", "Violence/g2/b.avi", "NonViolence/g1/c.mkv"]
), 3, 2))

print("%-34s %8s %8s  %s" % ("caso", "clips", "violent", "resultado"))
print("-" * 70)
fails = 0
for name, root, want_n, want_v in CASES:
    try:
        items = index_flat(root)
        n = len(items)
        v = sum(y for _, y in items)
        ok = (n == want_n and v == want_v)
        fails += (not ok)
        print("%-34s %8d %8d  %s" % (
            name, n, v, "OK" if ok else "MAL (esperaba %d/%d)" % (want_n, want_v)))
    except Exception as e:
        fails += 1
        print("%-34s %8s %8s  EXCEPCION: %s" % (name, "-", "-", str(e)[:40]))

# duplicate tree (RLVS mirror): same names under two wrappers -> must be refused
dup = mk(["Real Life Violence Dataset/%s/%s_%d.mp4" % (c, c, i)
          for c in ("Violence", "NonViolence") for i in range(1, 6)] +
         ["real life violence situations/Real Life Violence Dataset/%s/%s_%d.mp4" % (c, c, i)
          for c in ("Violence", "NonViolence") for i in range(1, 6)])
try:
    index_flat(dup)
    print("%-34s %19s" % ("arbol duplicado (rlvs)", "MAL: deberia rechazar"))
    fails += 1
except RuntimeError:
    print("%-34s %19s" % ("arbol duplicado (rlvs)", "rechazado correctamente"))

# ambiguity guard: prefixes that do NOT partition must be refused
amb = mk(["mix/fi1.avi", "mix/no1.avi", "mix/otracosa.avi"])
try:
    items = index_flat(amb)
    print("\n%-34s %8d           MAL: deberia rechazar" % ("cobertura parcial", len(items)))
    fails += 1
except RuntimeError:
    print("\n%-34s %19s" % ("cobertura parcial", "rechazada correctamente"))

# regression: rwf official split still works
rwf = mk(["RWF-2000/%s/%s/%s%d.avi" % (s, c, c, i)
          for s in ("train", "val") for c in ("Fight", "NonFight") for i in (1, 2)])
open(os.path.join(rwf, "_renamed.csv"), "w").write("a,b\n")
try:
    sp = index_rwf2000(rwf)
    ok = len(sp["train"]) == 4 and len(sp["test"]) == 4
    fails += (not ok)
    print("%-34s %8s %19s" % ("rwf2000 split oficial", "",
                              "OK" if ok else "MAL"))
except Exception as e:
    fails += 1
    print("%-34s EXCEPCION: %s" % ("rwf2000 split oficial", e))

print("\n%d fallos" % fails)
shutil.rmtree(BASE, ignore_errors=True)
sys.exit(1 if fails else 0)
