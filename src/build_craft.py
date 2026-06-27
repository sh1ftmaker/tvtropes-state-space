"""Build the payload for the crafting game (craft.html).

Infinite-Craft-style mechanic, fully offline: combining two ingredients = take
the (normalized) sum of their embedding vectors and return the TVTrope whose
vector is nearest by cosine. To do that nearest-neighbor search in the browser
we ship every trope's vector, but 30,984 x 768 float32 is 88 MB — far too big.

So we PCA-reduce the 768-dim embeddings to a compact `--dim` (default 48), fit
on the tropes and applied identically to the word + genre ingredients (one
linear map, so sums stay consistent), L2-normalize, and quantize to int8. That's
~1.5 MB of vectors, base64-packed into craft.json. The reduced cosine geometry
preserves the dominant semantic structure plenty well for a game.

Ingredients = the word taxonomy (words.py) + the genre taxonomy (genres.py).
Results are always tropes; a crafted trope's own reduced vector seeds further
combinations, so tropes + tropes -> deeper tropes.

Usage:
  python build_craft.py --emb out/embeddings_gemini.npz --points out/points.json \
      --words out/embeddings_words.npz --genres out/embeddings_genres.npz \
      --out out/craft.json --dim 48
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genres import SUPERGENRES  # noqa: E402
from words import KINDS  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def l2(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def quant(M, scale):
    """int8-quantize a float matrix already divided into [-1,1]-ish range."""
    return np.clip(np.round(M / scale * 127.0), -127, 127).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--points", required=True, help="points.json (for names + sg + popularity)")
    ap.add_argument("--words", required=True)
    ap.add_argument("--genres", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=48)
    args = ap.parse_args()

    from sklearn.decomposition import PCA

    d = np.load(args.emb, allow_pickle=True)
    X = d["vectors"].astype(np.float32)
    log(f"trope embeddings {X.shape}")

    pj = json.loads(Path(args.points).read_text(encoding="utf-8"))
    pts = pj["points"]
    assert len(pts) == X.shape[0], f"points {len(pts)} != emb {X.shape[0]}"
    # points are written in row order (i == row); keep that alignment
    pts.sort(key=lambda p: p["i"])
    names = [p["name"] for p in pts]
    sg = [int(p["sg"]) for p in pts]
    tot = [int(p.get("tot", 0)) for p in pts]
    supergenres = pj["meta"]["supergenres"]

    w = np.load(args.words, allow_pickle=True)
    wnames = [str(x) for x in w["names"]]
    wkinds = [str(x) for x in w["kinds"]]
    W = w["vectors"].astype(np.float32)

    g = np.load(args.genres, allow_pickle=True)
    gnames = [str(x) for x in g["names"]]
    gsupers = [str(x) for x in g["supers"]]
    G = g["vectors"].astype(np.float32)
    log(f"ingredients: {len(wnames)} words + {len(gnames)} genres")

    # ---- PCA fit on tropes, apply to everything (single linear map) ----
    pca = PCA(n_components=args.dim, random_state=42)
    Xr = pca.fit_transform(X)
    Wr = pca.transform(W)
    Gr = pca.transform(G)
    log(f"PCA -> {args.dim} dims, explained var = {pca.explained_variance_ratio_.sum():.3f}")

    # L2-normalize in reduced space so combination = sum then renormalize, and
    # cosine == dot for the nearest-neighbor search the browser does.
    Xr, Wr, Gr = l2(Xr), l2(Wr), l2(Gr)

    scale = float(np.abs(np.vstack([Xr, Wr, Gr])).max())
    Xq = quant(Xr, scale)
    Wq = quant(Wr, scale)
    Gq = quant(Gr, scale)

    # supergenre key -> color, and palette lookups
    super_color = {k: SUPERGENRES[k][1] for k in SUPERGENRES}
    super_key_list = list(SUPERGENRES.keys())

    elements = []
    for i, nm in enumerate(wnames):
        kind = wkinds[i]
        elements.append({
            "name": nm,
            "kind": kind,
            "color": KINDS.get(kind, ("", "#9aa0b8"))[1],
            "vec": [int(v) for v in Wq[i]],
        })
    for i, nm in enumerate(gnames):
        sk = gsupers[i]
        elements.append({
            "name": nm,
            "kind": "genre",
            "color": super_color.get(sk, "#9aa0b8"),
            "vec": [int(v) for v in Gq[i]],
        })

    payload = {
        "dim": args.dim,
        "scale": scale,
        "supergenres": supergenres,
        "kinds": [{"key": k, "label": KINDS[k][0], "color": KINDS[k][1]} for k in KINDS],
        "tropes": {
            "n": len(names),
            "names": names,
            "sg": sg,
            "tot": tot,
            "vec": base64.b64encode(Xq.tobytes()).decode("ascii"),
        },
        "elements": elements,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    mb = out.stat().st_size / 1e6
    log(f"wrote {out}  ({len(names)} tropes, {len(elements)} ingredients, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
