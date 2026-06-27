"""Project trope embeddings into GENRE space and prepare the viz payload.

The primary relational metric is now **genre/setting affinity**. Every trope was
embedded by gemini-embedding-2; so was a curated taxonomy of ~110 narrative
genres (see genres.py / embed_genres.py). Because both live in the same 768-dim
space, a trope's affinity to a genre is just a cosine dot product. We:

  1. score each trope against every genre  -> affinity matrix (N x G)
  2. standardize per-genre (z-score) so "how *much more* this trope is genre X
     than the average trope" drives the layout, not the genre's baseline mass
  3. UMAP that genre-affinity space -> the 2D & 3D positions (genre is what
     decides where a trope sits)
  4. dominant genre + dominant supergenre per trope (the coloring signal)
  5. per-supergenre affinity scores (remappable axes in the 3D view)
  6. KMeans over genre space (genre-coherent neighborhoods)
  7. exact k-NN in the FULL embedding space (the genuine "most similar trope"
     trace -- meaning, not just shared genre)

Usage:
  python project.py --emb out/embeddings_gemini.npz --feat out/trope_features.parquet \
      --genres out/embeddings_genres.npz --out out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genres import SUPERGENRES  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def norm_coords(a):
    """Center + scale coords into a friendly [-1, 1]-ish range."""
    a = np.asarray(a, dtype=np.float32)
    a = a - a.mean(0)
    s = np.abs(a).max() or 1.0
    return a / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--genres", required=True, help="embeddings_genres.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neighbors", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=24)
    args = ap.parse_args()

    import pandas as pd

    d = np.load(args.emb, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    X = d["vectors"].astype(np.float32)
    model = str(d["model"]) if "model" in d else "unknown"
    log(f"trope embeddings {X.shape} model={model}")

    g = np.load(args.genres, allow_pickle=True)
    gnames = [str(x) for x in g["names"]]
    gsupers = [str(x) for x in g["supers"]]
    G = g["vectors"].astype(np.float32)
    log(f"genre embeddings {G.shape} ({len(gnames)} genres)")

    feat = Path(args.feat)
    fdf = pd.read_parquet(feat) if feat.suffix == ".parquet" else pd.read_csv(feat)
    fdf["trope_id"] = fdf["trope_id"].astype(str)
    fdf = fdf.set_index("trope_id").reindex(ids)

    # ---- genre affinity (both sides L2-normalized -> cosine == dot) ----
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    A = Xn @ Gn.T  # (N, Gn)  cosine similarity, ~[-0.2, 0.6]
    # standardize each genre column: relative affinity, not baseline mass
    As = (A - A.mean(0)) / (A.std(0) + 1e-6)
    log(f"genre-affinity {A.shape}  raw[min={A.min():.3f} max={A.max():.3f}]")

    # ---- dominant genre + supergenre ----
    super_keys = list(SUPERGENRES.keys())
    super_idx = {k: i for i, k in enumerate(super_keys)}
    # member-genre column indices per supergenre
    members = {k: [j for j, s in enumerate(gsupers) if s == k] for k in super_keys}

    dom_genre = As.argmax(1)  # index into gnames
    dom_genre_name = [gnames[j] for j in dom_genre]
    # top-3 genres per trope (by standardized affinity)
    top3 = np.argsort(-As, axis=1)[:, :3]

    # per-supergenre score = mean standardized affinity of its member genres
    SA = np.zeros((X.shape[0], len(super_keys)), dtype=np.float32)
    for k, idxs in members.items():
        if idxs:
            SA[:, super_idx[k]] = As[:, idxs].mean(1)
    dom_super = SA.argmax(1)
    # min-max normalize each supergenre column to 0..1 for axis use
    SA01 = (SA - SA.min(0)) / (np.ptp(SA, 0) + 1e-6)

    # ---- UMAP over genre-affinity space (the primary layout) ----
    try:
        import umap

        red2 = umap.UMAP(n_components=2, n_neighbors=18, min_dist=0.18,
                         metric="cosine", random_state=42)
        xy = red2.fit_transform(As)
        red3 = umap.UMAP(n_components=3, n_neighbors=18, min_dist=0.18,
                         metric="cosine", random_state=42)
        xyz = red3.fit_transform(As)
        layout = "umap-genre"
    except Exception as e:  # noqa: BLE001
        log(f"UMAP unavailable ({e}); PCA fallback on genre space")
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=42).fit_transform(As)
        xyz = PCA(n_components=3, random_state=42).fit_transform(As)
        layout = "pca-genre"

    xy = norm_coords(xy)
    xyz = norm_coords(xyz)

    # ---- KMeans over genre space (genre-coherent clusters) ----
    from sklearn.cluster import KMeans

    k = min(args.clusters, X.shape[0])
    cl = KMeans(n_clusters=k, n_init=4, random_state=42).fit_predict(As)
    log(f"kmeans k={k} over genre space")

    # ---- nearest neighbors in FULL embedding space (true similarity) ----
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=args.neighbors + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    neigh = idx[:, 1:].astype(np.int32)

    # ---- assemble ----
    def col(name, default=None):
        return fdf[name].tolist() if name in fdf.columns else [default] * len(ids)

    names = col("trope")
    dtype = col("dominant_media", "unknown")
    n_film, n_tv, n_lit = col("n_film", 0), col("n_tv", 0), col("n_lit", 0)
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
        gv = gender[i]
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
            "sg": int(dom_super[i]),                 # dominant supergenre index
            "gen": dom_genre_name[i],                # dominant genre name
            "g3": [gnames[j] for j in top3[i]],      # top-3 genre names
            "sa": [round(float(v), 3) for v in SA01[i]],  # supergenre affinity 0..1
            "media": dtype[i],
            "nf": int(n_film[i] or 0),
            "nt": int(n_tv[i] or 0),
            "nl": int(n_lit[i] or 0),
            "tot": int(total[i] or 0),
            "g": (None if gv is None or (isinstance(gv, float) and np.isnan(gv)) else round(float(gv), 3)),
            "d": short(desc[i]),
            "nn": neigh[i].tolist(),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "n": len(points),
            "model": model,
            "layout": layout,
            "clusters": k,
            "supergenres": [
                {"key": kk, "label": SUPERGENRES[kk][0], "color": SUPERGENRES[kk][1]}
                for kk in super_keys
            ],
            "genres": gnames,
        },
        "points": points,
    }
    (out / "points.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {out/'points.json'}  ({len(points)} points, layout={layout})")


if __name__ == "__main__":
    main()
