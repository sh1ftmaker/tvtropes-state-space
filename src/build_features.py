"""Build the per-trope feature table for embedding + visualization.

Source: the extracted dhruvilgala/tvtropes CSVs (data/full/TVTropesData/...).
We read only the columns we need from the large occurrence tables so memory
stays modest despite the ~1.5GB of CSVs.

Output: out/trope_features.parquet, one row per trope:
  trope_id, trope, description,
  n_film, n_tv, n_lit, n_total,
  frac_film, frac_tv, frac_lit, dominant_media,
  gender_ratio

dominant_media is the categorical "type" used to color the state space; the
n_*/frac_* give each trope a media-mix signature.

Usage:
  python build_features.py --data ../data --out ../out/trope_features.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def find_dir(data: Path) -> Path:
    hits = list(data.rglob("tropes.csv"))
    if not hits:
        log(f"ERROR: tropes.csv not found under {data}")
        sys.exit(1)
    return hits[0].parent


def trope_id_counts(path: Path) -> pd.Series:
    """value_counts of trope_id, reading only that one column (chunked)."""
    if not path.exists():
        return pd.Series(dtype=int)
    total = None
    for chunk in pd.read_csv(path, usecols=["trope_id"], dtype=str, chunksize=500_000):
        vc = chunk["trope_id"].value_counts()
        total = vc if total is None else total.add(vc, fill_value=0)
    return total.astype(int) if total is not None else pd.Series(dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = find_dir(Path(args.data))
    log(f"data dir: {d}")

    # --- tropes (small enough to load whole) ---
    tr = pd.read_csv(d / "tropes.csv")
    tr.columns = [c.strip() for c in tr.columns]
    ren = {}
    for c in tr.columns:
        cl = c.lower()
        if cl == "tropeid":
            ren[c] = "trope_id"
        elif cl == "trope":
            ren[c] = "trope"
        elif cl == "description":
            ren[c] = "description"
    tr = tr.rename(columns=ren)
    tr = tr[["trope_id", "trope", "description"]].dropna(subset=["trope_id", "trope"])
    tr["trope_id"] = tr["trope_id"].astype(str)
    tr = tr.drop_duplicates("trope_id").reset_index(drop=True)
    log(f"tropes: {len(tr)}")

    # --- media counts (read only trope_id) ---
    cf = trope_id_counts(d / "film_tropes.csv")
    log(f"film occurrences mapped: {int(cf.sum()) if len(cf) else 0}")
    ct = trope_id_counts(d / "tv_tropes.csv")
    log(f"tv occurrences mapped: {int(ct.sum()) if len(ct) else 0}")
    cl_ = trope_id_counts(d / "lit_tropes.csv")
    log(f"lit occurrences mapped: {int(cl_.sum()) if len(cl_) else 0}")

    tr["n_film"] = tr["trope_id"].map(cf).fillna(0).astype(int)
    tr["n_tv"] = tr["trope_id"].map(ct).fillna(0).astype(int)
    tr["n_lit"] = tr["trope_id"].map(cl_).fillna(0).astype(int)
    tr["n_total"] = tr["n_film"] + tr["n_tv"] + tr["n_lit"]

    tot = tr["n_total"].replace(0, np.nan)
    tr["frac_film"] = (tr["n_film"] / tot).fillna(0).round(3)
    tr["frac_tv"] = (tr["n_tv"] / tot).fillna(0).round(3)
    tr["frac_lit"] = (tr["n_lit"] / tot).fillna(0).round(3)

    arr = tr[["n_film", "n_tv", "n_lit"]].to_numpy()
    labels = np.array(["film", "tv", "lit"])
    dom = labels[arr.argmax(1)]
    dom[arr.sum(1) == 0] = "unused"
    tr["dominant_media"] = dom

    # --- genderedness (read only name + normalized ratio) ---
    gpath = d / "genderedness_filtered.csv"
    tr["gender_ratio"] = np.nan
    if gpath.exists():
        # discover the columns first
        head = pd.read_csv(gpath, nrows=0)
        cols = [c.strip() for c in head.columns]
        name_col = "Trope" if "Trope" in cols else cols[0]
        gr_col = None
        for c in cols:
            if c.lower().replace(" ", "").startswith("normalizedgenderratio"):
                gr_col = c
                break
        if gr_col is None:
            for c in cols:
                if c.lower().startswith("gender ratio"):
                    gr_col = c
                    break
        if gr_col:
            gmap: dict[str, float] = {}
            for chunk in pd.read_csv(gpath, usecols=[name_col, gr_col], chunksize=500_000):
                chunk.columns = [c.strip() for c in chunk.columns]
                for n, v in zip(chunk[name_col].astype(str),
                                pd.to_numeric(chunk[gr_col], errors="coerce")):
                    if n not in gmap and pd.notna(v):
                        gmap[n] = float(v)
            tr["gender_ratio"] = tr["trope"].map(gmap)
            log(f"genderedness matched: {tr['gender_ratio'].notna().sum()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        tr.to_parquet(out, index=False)
    except Exception as e:  # noqa: BLE001
        log(f"parquet failed ({e}); writing csv instead")
        out = out.with_suffix(".csv")
        tr.to_csv(out, index=False)
    log(f"wrote {out}  ({len(tr)} rows)")
    log("dominant_media:\n" + tr["dominant_media"].value_counts().to_string())


if __name__ == "__main__":
    main()
