"""Project high-dim trope embeddings down to 2D/3D and prepare the viz payload.

Pipeline:
  1. load embeddings (.npz: ids, vectors) from embed_gemini.py or embed_local.py
  2. join per-trope metadata (name, media mix -> "type", genderedness, popularity)
  3. PCA pre-reduce -> UMAP to 2D (and 3D) for the state-space layout
  4. KMeans clusters (a discrete "semantic neighborhood" coloring)
  5. k-nearest-neighbors in full embedding space (the real similarity graph)
  6. emit out/points.json (+ points.bin) consumed by the web demo

Usage:
  python project.py --emb ../out/embeddings_local.npz --feat ../out/trope_features.parquet --out ../out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neighbors", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=24)
    ap.add_argument("--pca", type=int, default=50)
    args = ap.parse_args()

    import pandas as pd

    d = np.load(args.emb, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    X = d["vectors"].astype(np.float32)
    model = str(d["model"]) if "model" in d else "unknown"
    log(f"embeddings {X.shape} model={model}")

    feat = Path(args.feat)
    fdf = pd.read_parquet(feat) if feat.suffix == ".parquet" else pd.read_csv(feat)
    fdf["trope_id"] = fdf["trope_id"].astype(str)
    fdf = fdf.set_index("trope_id")
    fdf = fdf.reindex(ids)  # align to embedding order

    # ---- PCA pre-reduction (denoise + speed up UMAP) ----
    from sklearn.decomposition import PCA

    ncomp = min(args.pca, X.shape[1], X.shape[0] - 1)
    Xp = PCA(n_components=ncomp, random_state=42).fit_transform(X) if ncomp < X.shape[1] else X
    log(f"pca -> {Xp.shape}")

    # ---- UMAP 2D + 3D ----
    try:
        import umap

        reducer2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.12,
                             metric="cosine", random_state=42)
        xy = reducer2.fit_transform(Xp)
        reducer3 = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.12,
                             metric="cosine", random_state=42)
        xyz = reducer3.fit_transform(Xp)
        layout = "umap"
    except Exception as e:  # noqa: BLE001
        log(f"UMAP unavailable ({e}); falling back to PCA-2D/3D")
        xy = PCA(n_components=2, random_state=42).fit_transform(Xp)
        xyz = PCA(n_components=3, random_state=42).fit_transform(Xp)
        layout = "pca"

    # normalize coords into a friendly range
    def norm(a):
        a = np.asarray(a, dtype=np.float32)
        a = a - a.mean(0)
        s = np.abs(a).max() or 1.0
        return a / s

    xy = norm(xy)
    xyz = norm(xyz)

    # ---- clusters ----
    from sklearn.cluster import KMeans

    k = min(args.clusters, X.shape[0])
    cl = KMeans(n_clusters=k, n_init=4, random_state=42).fit_predict(Xp)
    log(f"kmeans k={k}")

    # ---- nearest neighbors in full embedding space ----
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=args.neighbors + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    neigh = idx[:, 1:].astype(np.int32)  # drop self

    # ---- assemble points ----
    def col(name, default=None):
        if name in fdf.columns:
            return fdf[name].tolist()
        return [default] * len(ids)

    names = col("trope")
    dtype = col("dominant_media", "unknown")
    n_film = col("n_film", 0)
    n_tv = col("n_tv", 0)
    n_lit = col("n_lit", 0)
    total = col("n_total", 0)
    gender = col("gender_ratio", None)
    desc = col("description", "")

    def short(s, n=240):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = ""
        s = str(s).strip().replace("\n", " ")
        return s[:n] + ("…" if len(s) > n else "")

    points = []
    for i, tid in enumerate(ids):
        g = gender[i]
        points.append({
            "i": i,
            "id": tid,
            "name": names[i],
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "x3": round(float(xyz[i, 0]), 4),
            "y3": round(float(xyz[i, 1]), 4),
            "z3": round(float(xyz[i, 2]), 4),
            "c": int(cl[i]),
            "media": dtype[i],
            "nf": int(n_film[i] or 0),
            "nt": int(n_tv[i] or 0),
            "nl": int(n_lit[i] or 0),
            "tot": int(total[i] or 0),
            "g": (None if g is None or (isinstance(g, float) and np.isnan(g)) else round(float(g), 3)),
            "d": short(desc[i]),
            "nn": neigh[i].tolist(),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"n": len(points), "model": model, "layout": layout, "clusters": k},
        "points": points,
    }
    (out / "points.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {out/'points.json'}  ({len(points)} points, layout={layout})")


if __name__ == "__main__":
    main()
