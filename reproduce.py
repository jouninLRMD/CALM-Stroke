"""
Reproduces every result of the CALM-Stroke paper, from the public SOOP download to the tables, figures and numbers.

Stages, in order:
  download    SOOP metadata, lesion masks and DWI from OpenNeuro (about 2.8 GB)
  atlas       arterial territory atlas (Liu et al. 2023)
  preprocess  brain mask, relative ADC, affine registration to MNI152 2 mm, lesion features, QC images
  cohort      cohort table and frozen folds (5-fold CV, leave one scanner out, 1.5 T -> 3 T)
  features    lesion load per arterial territory and 4 mm lesion maps
  baselines   reference models R0, R1, R1b, R2a, R2b, I1, I2
  cache       memory-mapped image arrays for training
  train       all networks, seeds and ablations reported in the paper (GPU)
  evaluate    metrics, bootstrap intervals, subgroups and decision curves
  paper       manuscript figures, table bodies and number macros (results/paper)
  explain     lesion-map agreement, territory-token occlusion and the late-fusion comparator (GPU)

Usage (from the repository root):
  python reproduce.py                          # everything
  python reproduce.py --dry-run                # print the commands only
  python reproduce.py --from train             # start at a stage
  python reproduce.py --only evaluate paper    # run some stages
  python reproduce.py --only train --models calm25d   # one model family
Finished training runs are skipped; use --force to run them again.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC, SCRIPTS = REPO / "src", REPO / "scripts"
RES = Path(os.environ.get("CALM_RESULTS", str(REPO / "results")))
STAGES = ["download", "atlas", "preprocess", "cohort", "features", "baselines", "cache", "train", "evaluate", "paper",
          "explain"]
N_UNITS = {"cv5": 5, "loso": 4, "field": 1}

# (model, extra flags, {scheme: seeds}) exactly as used for the paper
RUNS = [
    ("resnet3d", [],                                {"cv5": [0], "loso": [0], "field": [0]}),
    ("resnet3d", ["--no-anchor"],                   {"cv5": [0]}),
    ("calm",     [],                                {"cv5": [0, 1, 2], "loso": [0], "field": [0]}),
    ("slice25d", [],                                {"cv5": [0, 1, 2, 3, 4], "loso": [0, 1, 2], "field": [0, 1, 2]}),
    ("slice25d", ["--anchor-vars", "clin_lesion"],  {"cv5": [0, 1, 2, 3, 4], "loso": [0, 1, 2], "field": [0, 1, 2]}),
    ("slice25d", ["--no-anchor"],                   {"cv5": [0, 1, 2]}),
    ("calm25d",  [],                                {"cv5": [0, 1, 2, 3, 4], "loso": [0, 1, 2], "field": [0, 1, 2]}),
    ("calm25d",  ["--no-seg"],                      {"cv5": [0, 1, 2]}),
    ("calm25d",  ["--no-aux"],                      {"cv5": [0, 1, 2]}),
    ("calm25d",  ["--no-anchor"],                   {"cv5": [0, 1, 2]}),
]


def tag(model, flags, scheme, seed):
    t = model + ("_noanchor" if "--no-anchor" in flags else "") + ("_lesanchor" if "clin_lesion" in flags else "")
    t += ("_noseg" if "--no-seg" in flags else "") + ("_noaux" if "--no-aux" in flags else "")
    return f"{t}__{scheme}__s{seed}"


def finished(t, scheme):
    f = RES / "oof" / f"{t}.csv"
    if not f.exists():
        return False
    import csv
    with open(f, newline="") as fh:
        units = {r["test_unit"] for r in csv.DictReader(fh)}
    return len(units) >= N_UNITS[scheme]


def run(cmd, log=None, dry=False):
    print("  $ " + " ".join(str(c) for c in cmd) + (f"  > {log}" if log else ""), flush=True)
    if dry:
        return
    t0 = time.time()
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
    else:
        rc = subprocess.call(cmd, cwd=REPO)
    if rc != 0:
        sys.exit(f"Command failed (exit code {rc})" + (f"; see {log}" if log else ""))
    print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STAGES)
    ap.add_argument("--from", dest="start", choices=STAGES)
    ap.add_argument("--models", nargs="+", choices=sorted({r[0] for r in RUNS}), help="limit the train stage")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="CPU workers")
    ap.add_argument("--loader-workers", type=int, default=4, help="data loader workers for training")
    ap.add_argument("--force", action="store_true", help="retrain runs whose predictions already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.only:
        stages = [s for s in STAGES if s in a.only]
    elif a.start:
        stages = STAGES[STAGES.index(a.start):]
    else:
        stages = STAGES
    py = sys.executable
    for st in stages:
        print(f"\n=== {st}", flush=True)
        if st == "download":
            run([py, SCRIPTS / "download_soop.py"], dry=a.dry_run)
        elif st == "atlas":
            run([py, SCRIPTS / "fetch_atlas.py"], dry=a.dry_run)
        elif st == "preprocess":
            run([py, SRC / "preprocess_dwi.py", "--workers", str(a.workers)], dry=a.dry_run)
        elif st == "cohort":
            run([py, SRC / "cohort_splits.py"], dry=a.dry_run)
        elif st == "features":
            run([py, SRC / "atlas_features.py"], dry=a.dry_run)
        elif st == "baselines":
            run([py, SRC / "baselines.py"], dry=a.dry_run)
        elif st == "cache":
            run([py, SRC / "build_cache.py"], dry=a.dry_run)
        elif st == "train":
            for model, flags, schemes in RUNS:
                if a.models and model not in a.models:
                    continue
                for scheme, seeds in schemes.items():
                    for seed in seeds:
                        t = tag(model, flags, scheme, seed)
                        if finished(t, scheme) and not a.force:
                            print(f"  {t}: predictions exist, skipped")
                            continue
                        cmd = [py, SRC / "train.py", "--model", model, "--scheme", scheme, "--seed", str(seed),
                               "--workers", str(a.loader_workers)] + flags
                        run(cmd, log=RES / "logs" / f"train_{t}.txt", dry=a.dry_run)
        elif st == "evaluate":
            run([py, SRC / "evaluate.py"], log=RES / "logs" / "evaluate.txt", dry=a.dry_run)
        elif st == "paper":
            run([py, SRC / "paper_outputs.py"], log=RES / "logs" / "paper_outputs.txt", dry=a.dry_run)
        elif st == "explain":
            run([py, SRC / "explain.py"], log=RES / "logs" / "explain.txt", dry=a.dry_run)
    print("\nFinished. Main report: results/report.md; manuscript outputs: results/paper and results/explain")


if __name__ == "__main__":
    main()
