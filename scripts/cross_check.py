"""Cross-check corank1_walk and scale against the C reference (ref/cross_test).

Data is exchanged through plain text files in --dir (default: current dir):

    in_walk.txt   T (n^3 ints, T[a][b][c]) and u (n ints)    Sage -> C
    in_scale.txt  R (l^3 ints, R[a][b][c])                    Sage -> C
    out_walk.txt  output of `cross_test walk`                 C -> Sage
    out_scale.txt output of `cross_test scale`                C -> Sage

By hand:
    python scripts/cross_check.py gen --seed 1
    ./cross_test walk  < in_walk.txt  > out_walk.txt
    ./cross_test scale < in_scale.txt > out_scale.txt
    python scripts/cross_check.py check

Many cases at once:
    python scripts/cross_check.py run --bin /path/to/ref/cross_test --cases 100

What is compared:
    walk_ok      must be equal (both reject when a kernel is not 1-dim).
    LU, LV, LW   column by column after scaling each vector so its first
                 non-zero entry is 1 (C and Sage pick different kernel
                 representatives).
    scale_ok     must be equal (zero pattern of R is scaling invariant).
    CF           final canonical tensor, must be identical.
    R, f, g, h   in walk mode not compared (they depend on the representatives);
                 compared exactly in scale mode, where both get the same R.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sage.all import matrix, set_random_seed, vector

from minimeds_sage import Tensor, TensorSpace, corank1_point, corank1_walk, scale

# LEVEL -> (q, n, l); must match ref/config.h of the compiled cross_test
LEVELS = {0: (101, 6, 5), 1: (4093, 25, 9), 3: (4093, 34, 11), 5: (4093, 44, 12)}

def flat(T: Tensor) -> list[int]:
    n = T.n
    return [
        int(T.entry(a, b, c)) for a in range(n) for b in range(n) for c in range(n)
    ]


def from_flat(space: TensorSpace, vals: list[int]) -> Tensor:
    n, F = space.n, space.field.F
    slices = [matrix(F, n, n, vals[a * n * n : (a + 1) * n * n]) for a in range(n)]
    return Tensor(space.field, n, slices)


def write_ints(path: Path, *chunks: list[int]) -> None:
    path.write_text("\n".join(" ".join(map(str, c)) for c in chunks) + "\n")


def read_ints(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def parse_out(path: Path) -> dict[str, list[int]]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            key, *vals = line.split()
            out[key] = [int(v) for v in vals]
    return out


def normalize(v):
    """Scale v so that its first non-zero entry is 1."""
    for x in v:
        if x != 0:
            return v / x
    return v


def gen(args, dirpath: Path) -> None:
    q, n, l = LEVELS[args.level]
    set_random_seed(args.seed)

    space = TensorSpace(q, n)
    T = space.random_tensor()
    u = corank1_point(T)
    write_ints(dirpath / "in_walk.txt", flat(T), [int(x) for x in u])

    R = TensorSpace(q, l).random_tensor()
    write_ints(dirpath / "in_scale.txt", flat(R))


def check(args, dirpath: Path) -> list[str]:
    q, n, l = LEVELS[args.level]
    space, lspace = TensorSpace(q, n), TensorSpace(q, l)
    F = space.field.F
    errors: list[str] = []

    # ---- walk mode: Corank1Walk -> transform -> Scale -> transform ----
    vals = read_ints(dirpath / "in_walk.txt")
    T, u = from_flat(space, vals[: n**3]), vals[n**3 :]
    out = parse_out(dirpath / "out_walk.txt")

    if out.get("params") != [q, n, l]:
        return [f"params mismatch: C {out.get('params')} vs script {[q, n, l]}"]

    walk = corank1_walk(T, u, steps=l, check_span=False)
    c_ok, s_ok = out["walk_ok"][0] == 1, walk is not None

    if args.kernel == "fixed_piv" and not c_ok:
        pass  # fixed_piv may reject valid corank-1 matrices (Sec. 6.1)
    elif c_ok != s_ok:
        errors.append(f"walk_ok: C {c_ok} vs Sage {s_ok}")
    elif c_ok:
        for name, M in zip(("LU", "LV", "LW"), walk):
            cols = out[name]
            for j in range(l):
                vc = normalize(vector(F, cols[j * n : (j + 1) * n]))
                vs = normalize(M.column(j))
                if vc != vs:
                    errors.append(f"{name} column {j}: C {vc} vs Sage {vs}")

        R = T.transform(*walk)
        fgh = scale(R)
        sc_ok = out["scale_ok"][0] == 1
        if sc_ok != (fgh is not None):
            errors.append(f"walk scale_ok: C {sc_ok} vs Sage {fgh is not None}")
        elif sc_ok and out["CF"] != flat(R.transform(*fgh)):
            errors.append("walk CF differs")

    # ---- scale mode: same R on both sides ----
    R = from_flat(lspace, read_ints(dirpath / "in_scale.txt"))
    out = parse_out(dirpath / "out_scale.txt")
    fgh = scale(R)
    sc_ok = out["scale_ok"][0] == 1

    if sc_ok != (fgh is not None):
        errors.append(f"scale_ok: C {sc_ok} vs Sage {fgh is not None}")
    elif sc_ok:
        for name, D in zip("fgh", fgh):
            if out[name] != [int(x) for x in D.diagonal()]:
                errors.append(f"{name}: C {out[name]} vs Sage {list(D.diagonal())}")
        if out["CF"] != flat(R.transform(*fgh)):
            errors.append("scale CF differs")

    return errors


def run(args, dirpath: Path) -> list[str]:
    failures: list[str] = []
    for case in range(args.cases):
        args.seed = args.first_seed + case
        gen(args, dirpath)
        for mode, extra in (("walk", [args.kernel]), ("scale", [])):
            with (
                open(dirpath / f"in_{mode}.txt") as fin,
                open(dirpath / f"out_{mode}.txt", "w") as fout,
            ):
                subprocess.run(
                    [args.bin, mode, *extra], stdin=fin, stdout=fout, check=True
                )
        failures += [f"seed {args.seed}: {e}" for e in check(args, dirpath)]
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--level", type=int, default=0, choices=sorted(LEVELS))
    p.add_argument("--dir", type=Path, default=Path("."))
    p.add_argument("--kernel", default="fast", choices=("fast", "fixed_piv"))
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="write in_walk.txt and in_scale.txt")
    g.add_argument("--seed", type=int, default=0)

    sub.add_parser("check", help="compare out_*.txt with Sage")

    r = sub.add_parser("run", help="gen + run C + check, many seeds")
    r.add_argument("--bin", required=True, help="path to ref/cross_test")
    r.add_argument("--cases", type=int, default=100)
    r.add_argument("--first-seed", type=int, default=0)

    args = p.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "gen":
        gen(args, args.dir)
        print(f"wrote {args.dir / 'in_walk.txt'} and {args.dir / 'in_scale.txt'}")
        return

    errors = check(args, args.dir) if args.cmd == "check" else run(args, args.dir)
    for e in errors:
        print("MISMATCH", e)
    total = 1 if args.cmd == "check" else args.cases
    status = "FAIL" if errors else "OK"
    print(f"{status}: {len(errors)} mismatch(es) over {total} case(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
