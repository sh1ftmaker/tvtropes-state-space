"""Placeholder local embedder: TF-IDF + Truncated SVD (LSA).

This exists ONLY so the projection + visualization pipeline can run end-to-end
immediately, before a Gemini API key is available. It produces the same on-disk
format as embed_gemini.py (an .npz with `ids` and `vectors`) so the downstream
steps are identical. Swap in embeddings_gemini.npz when ready -- nothing else
changes.

Usage:
  python embed_local.py --in ../out/trope_features.parquet --out ../out/embeddings_local.npz --dim 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    inp = Path(args.inp)
    df = pd.read_parquet(inp) if inp.suffix == ".parquet" else pd.read_csv(inp)
    ids = df["trope_id"].astype(str).tolist()
    names = df["trope"].astype(str).tolist()
    descs = df["description"].fillna("").astype(str).tolist()
    texts = [f"{n}. {d}" for n, d in zip(names, descs)]
    log(f"loaded {len(texts)} tropes")

    vec = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=3,
        stop_words="english", sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    log(f"tfidf {X.shape}")

    dim = min(args.dim, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    Z = svd.fit_transform(X).astype(np.float32)
    Z = normalize(Z)  # unit vectors -> cosine == dot
    log(f"svd {Z.shape}  explained_var={svd.explained_variance_ratio_.sum():.3f}")

    out = Path(args.out)
    np.savez_compressed(out, ids=np.array(ids), vectors=Z, model="tfidf-svd-placeholder", dim=dim)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
