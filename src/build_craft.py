"""Build the payload for the crafting game (craft.html) + prove reachability.

Infinite-Craft-style mechanic, fully offline: combining two ingredients = take
the (normalized) sum of their embedding vectors and return whichever craftable
item (word / genre / trope) is nearest by cosine. The player starts with ONLY
the 4 primordial roots (Hero / World / Conflict / Wonder, see words.ROOTS) and
must be able to reach every genre and every trope from there.

Because combine() is deterministic, "everything is reachable" is not automatic —
it's a graph-reachability property we must verify. So this script:

  1. PCA-reduces the 768-d embeddings to `--dim` (default 48), fit on tropes and
     applied identically to the word/genre/root vectors (one linear map, so
     vector sums stay consistent), then int8-quantizes.
  2. Dequantizes back to float exactly as the browser will, and runs the whole
     reachability analysis on *those* vectors — so the recipes we find are valid
     in-game bit-for-bit (modulo float32 summation order).
  3. BFS from the 4 roots under the real combine rule (nearest item to the
     normalized sum, excluding the two inputs), growing a small set of reachable
     "hub" anchors as needed, until every craftable item is reached.
  4. Asserts full coverage and emits, for every item, one concrete recipe (its
     two parents) — a constructive proof, and the data behind the in-game hint.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--points", required=True, help="points.json (names + sg + popularity)")
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
    pts = sorted(pj["points"], key=lambda p: p["i"])
    assert len(pts) == X.shape[0], f"points {len(pts)} != emb {X.shape[0]}"
    tnames = [p["name"] for p in pts]
    tsg = [int(p["sg"]) for p in pts]
    ttot = [int(p.get("tot", 0)) for p in pts]
    supergenres = pj["meta"]["supergenres"]
    super_color = {sg["key"]: sg["color"] for sg in supergenres}
    super_label = {sg["key"]: sg["label"] for sg in supergenres}

    w = np.load(args.words, allow_pickle=True)
    wnames = [str(x) for x in w["names"]]
    wkinds = [str(x) for x in w["kinds"]]
    W = w["vectors"].astype(np.float32)
    is_root = np.array([k == "root" for k in wkinds])
    rnames = [n for n, r in zip(wnames, is_root) if r]
    log(f"roots: {rnames}")

    g = np.load(args.genres, allow_pickle=True)
    gnames = [str(x) for x in g["names"]]
    gsupers = [str(x) for x in g["supers"]]
    G = g["vectors"].astype(np.float32)

    # ---- PCA fit on tropes, applied to everything ----
    pca = PCA(n_components=args.dim, random_state=42)
    Xr = l2(pca.fit_transform(X))
    Wr = l2(pca.transform(W))
    Gr = l2(pca.transform(G))
    log(f"PCA -> {args.dim} dims, explained var = {pca.explained_variance_ratio_.sum():.3f}")

    # split words from roots
    Rr = Wr[is_root]
    Wr_words = Wr[~is_root]
    word_names = [n for n, r in zip(wnames, is_root) if not r]
    word_kinds = [k for k, r in zip(wkinds, is_root) if not r]

    # ---- assemble the craftable universe: words, then genres, then tropes ----
    # (roots are NOT craftable; they're the only seeds). Stored columnar to stay
    # small on mobile: kind is implied by index range, names in one array, and
    # the per-item genre signal is a single index.
    items_vec = np.vstack([Wr_words, Gr, Xr]).astype(np.float32)
    n_word, n_genre, n_trope = len(word_names), len(gnames), len(tnames)
    M = n_word + n_genre + n_trope
    super_key_idx = {sg["key"]: i for i, sg in enumerate(supergenres)}
    names_all = list(word_names) + list(gnames) + list(tnames)
    # supergenre index per genre and per trope (-1 = n/a, words use their kind)
    gsg = [super_key_idx.get(k, -1) for k in gsupers]
    tsg_idx = list(tsg)
    log(f"craftable items: {n_word} words + {n_genre} genres + {n_trope} tropes = {M}")

    # ---- quantize, then DEQUANTIZE exactly as the browser will ----
    scale = float(np.abs(np.vstack([items_vec, Rr])).max())
    itemsq = np.clip(np.round(items_vec / scale * 127.0), -127, 127).astype(np.int8)
    rootsq = np.clip(np.round(Rr / scale * 127.0), -127, 127).astype(np.int8)
    # the exact vectors the game uses for its nearest-neighbor search
    Vq = l2(itemsq.astype(np.float32) * scale / 127.0)
    Rq = l2(rootsq.astype(np.float32) * scale / 127.0)

    # ---- reachability proof under the real combine rule (cached: ~10 min) ----
    cache = Path(args.out).with_name(f"_craft_reach_{M}_{args.dim}.npz")
    if cache.exists():
        cd = np.load(cache)
        parents, n_anchor_hubs = cd["parents"].astype(np.int64), int(cd["hubs"])
        log(f"loaded cached reachability proof from {cache.name}")
    else:
        parents, n_anchor_hubs = prove_reachable(Vq, Rq, M)
        np.savez_compressed(cache, parents=parents, hubs=n_anchor_hubs)
    reached = parents[:, 0] >= 0
    cov = reached.mean()
    by = lambda lo, hi: reached[lo:hi].mean()
    log(f"REACHABILITY: {reached.sum()}/{M} = {cov:.4%}  "
        f"(words {by(0,n_word):.3f}, genres {by(n_word,n_word+n_genre):.3f}, "
        f"tropes {by(n_word+n_genre,M):.3f})  hubs+roots anchors used")
    assert reached.all(), f"NOT FULLY REACHABLE: {(~reached).sum()} items unreachable"
    log(f"PROVEN: all {M} items craftable from the 4 roots ({n_anchor_hubs} hub anchors needed)")

    payload = {
        "dim": args.dim,
        "scale": scale,
        "supergenres": supergenres,
        "kinds": [{"key": k, "label": KINDS[k][0], "color": KINDS[k][1]} for k in KINDS],
        "roots": [{"name": rnames[i], "vec": [int(v) for v in rootsq[i]]} for i in range(len(rnames))],
        "counts": {"word": n_word, "genre": n_genre, "trope": n_trope},
        "names": names_all,
        "wsub": word_kinds,                  # kind key per word (color/label lookup)
        "gsg": gsg,                          # supergenre index per genre
        "tsg": base64.b64encode(np.array(tsg_idx, dtype=np.int16).tobytes()).decode("ascii"),
        "vec": base64.b64encode(itemsq.tobytes()).decode("ascii"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # recipes shipped separately (only needed for the hint feature) so the
    # initial mobile payload stays lean. raw int32 little-endian, M*4.
    rec_out = out.with_name("craft_recipes.bin")
    rec_out.write_bytes(parents.astype("<i4").tobytes())
    mb = out.stat().st_size / 1e6
    log(f"wrote {out}  ({M} items, {len(rnames)} roots, {mb:.1f} MB) + {rec_out.name} "
        f"({rec_out.stat().st_size/1e6:.2f} MB)")


def prove_reachable(V, R, M, chunk=2048):
    """BFS from the 4 roots under combine(a,b)=argmax cosine(V, norm(a+b)),
    excluding the two inputs. Grow reachable 'hub' anchors until all M items are
    reached. Returns parents (M,4) recipe array and the hub count.

    parents[i] = [aKind, aIdx, bKind, bIdx], kind 0=root / 1=item.
    """
    Vt = V.T.copy()  # (dim, M) for fast sgemm
    nR = R.shape[0]
    parents = np.full((M, 4), -1, dtype=np.int64)

    def nn_excl(S, excl_self, excl_anchor):
        """For each row of normalized sums S (k,dim) return argmax item index,
        masking out the per-row self index (excl_self[k]) and a shared anchor
        index (excl_anchor, or -1)."""
        out = np.empty(S.shape[0], dtype=np.int64)
        for s in range(0, S.shape[0], chunk):
            e = min(s + chunk, S.shape[0])
            sims = S[s:e] @ Vt                       # (b, M)
            rows = np.arange(e - s)
            sims[rows, excl_self[s:e]] = -2.0
            if excl_anchor >= 0:
                sims[:, excl_anchor] = -2.0
            out[s:e] = sims.argmax(1)
        return out

    # ---- seeds: every root+root pair (the only moves available turn 1) ----
    frontier = []
    for i in range(nR):
        for j in range(i, nR):
            s = R[i] + R[j]
            s = s / (np.linalg.norm(s) + 1e-9)
            sims = s @ Vt
            r = int(sims.argmax())
            if parents[r, 0] < 0:
                parents[r] = (0, i, 0, j)
                frontier.append(r)
    reached = parents[:, 0] >= 0

    # anchors we can combine the frontier with: the 4 roots (kind 0) + grown hubs
    anchor_vecs = [R[i] for i in range(nR)]
    anchor_ref = [(0, i) for i in range(nR)]
    n_hubs = 0
    rng = np.random.default_rng(42)

    def expand():
        nonlocal frontier, reached
        while frontier:
            F = np.array(frontier, dtype=np.int64)
            Fv = V[F]
            newly = []
            for (akind, aidx), av in zip(anchor_ref, anchor_vecs):
                S = Fv + av
                S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
                excl_anchor = aidx if akind == 1 else -1
                res = nn_excl(S, F, excl_anchor)
                hit = ~reached[res]
                for k in np.where(hit)[0]:
                    r = int(res[k])
                    if not reached[r]:
                        reached[r] = True
                        parents[r] = (1, int(F[k]), akind, aidx)
                        newly.append(r)
            frontier = newly

    expand()

    # ---- grow hub anchors toward the unreached frontier until full ----
    while not reached.all():
        U = np.where(~reached)[0]
        Rch = np.where(reached)[0]
        sample = U if U.size <= 400 else rng.choice(U, 400, replace=False)
        # reached items nearest to the unreached sample = good pushing anchors
        sims = V[sample] @ V[Rch].T
        hubs = np.unique(Rch[sims.argmax(1)])
        added = 0
        for h in hubs:
            ref = (1, int(h))
            if ref in anchor_ref:
                continue
            anchor_ref.append(ref)
            anchor_vecs.append(V[h])
            added += 1
            n_hubs += 1
        if added == 0:  # no new anchors -> can't progress
            break
        frontier = list(np.where(reached)[0])  # re-expand everything with new anchors
        expand()

    return parents, n_hubs


if __name__ == "__main__":
    main()
