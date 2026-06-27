"""Embed TVTropes trope descriptions with Google's Gemini embedding model.

Primary embedding path for the project. Reads a feature table produced by
build_features.py and writes a dense float32 matrix aligned to trope ids.

Model: `gemini-embedding-001` (Google's current GA text embedding model, a.k.a.
"Gemini Embedding"). Configurable via --model.

Features:
  * resumable cache (per-id vectors persisted to an npz shard dir) so a rerun
    never re-pays for ids already embedded
  * batched requests with exponential backoff on 429/5xx
  * configurable output dimensionality (default 768 -- compact + great for viz;
    the model natively emits 3072 and supports MRL truncation)

Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.

Usage:
  python embed_gemini.py --in ../out/trope_features.parquet --out ../out/embeddings_gemini.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_rows(path: Path):
    """Return list of (id, text) for embedding. Accepts parquet or csv."""
    import pandas as pd

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Build the text we embed: trope name + description gives the model both the
    # label and its semantics, which keeps near-synonym tropes close together.
    ids = df["trope_id"].astype(str).tolist()
    names = df["trope"].astype(str).tolist()
    descs = df["description"].fillna("").astype(str).tolist()
    texts = []
    for n, d in zip(names, descs):
        d = d.strip()
        # Trim absurdly long descriptions to keep within token budget.
        if len(d) > 8000:
            d = d[:8000]
        texts.append(f"{n}. {d}" if d else n)
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="gemini-embedding-2")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--workers", type=int, default=24, help="concurrent single-input requests")
    ap.add_argument("--task-type", default="SEMANTIC_SIMILARITY")
    ap.add_argument("--limit", type=int, default=0, help="embed only first N (debug)")
    # auth: either an API key (AI Studio) or a Vertex AI service account
    ap.add_argument("--vertex", action="store_true", help="use Vertex AI + service account")
    ap.add_argument("--sa", default=None, help="service-account JSON path (implies --vertex)")
    ap.add_argument("--project", default="selstech")
    ap.add_argument("--location", default="global")
    args = ap.parse_args()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log("ERROR: pip install google-genai")
        sys.exit(2)

    use_vertex = args.vertex or bool(args.sa)
    if use_vertex:
        from google.oauth2 import service_account

        sa_path = args.sa or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not sa_path:
            log("ERROR: --sa <service_account.json> required for Vertex mode.")
            sys.exit(2)
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location, credentials=creds)
        log(f"auth: Vertex AI project={args.project} loc={args.location} model={args.model}")
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY), or use --sa for Vertex.")
            sys.exit(2)
        client = genai.Client(api_key=api_key)
        log(f"auth: AI Studio api-key model={args.model}")

    inp = Path(args.inp)
    out = Path(args.out)
    cache_dir = out.with_suffix("").as_posix() + "_cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    ids, texts = load_rows(inp)
    if args.limit:
        ids, texts = ids[: args.limit], texts[: args.limit]
    log(f"loaded {len(ids)} tropes")

    # Resume from cache: one .npy per id.
    cache = Path(cache_dir)
    vecs: dict[str, np.ndarray] = {}
    for f in cache.glob("*.npy"):
        vecs[f.stem] = np.load(f)
    if vecs:
        log(f"resumed {len(vecs)} cached vectors")

    todo = [(i, t) for i, t in zip(ids, texts) if i not in vecs]
    log(f"{len(todo)} to embed")

    # gemini-embedding-* processes ONE input per request, so we fan out single
    # calls across a thread pool. Each worker retries transient errors with
    # exponential backoff and persists its vector to the per-id cache.
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cfg = types.EmbedContentConfig(
        task_type=args.task_type,
        output_dimensionality=args.dim,
    )
    lock = threading.Lock()
    state = {"done": 0, "fail": 0}

    def embed_one(item):
        tid, text = item
        delay = 2.0
        for attempt in range(8):
            try:
                resp = client.models.embed_content(model=args.model, contents=text, config=cfg)
                emb = np.asarray(resp.embeddings[0].values, dtype=np.float32)
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n  # unit vectors -> cosine == dot, ideal for UMAP
                vecs[tid] = emb
                np.save(cache / f"{tid}.npy", emb)
                with lock:
                    state["done"] += 1
                    if state["done"] % 250 == 0:
                        log(f"  embedded {state['done']}/{len(todo)} (fail {state['fail']})")
                return
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                transient = any(c in msg for c in ("429", "500", "503", "deadline",
                                                   "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if attempt == 7 or not transient:
                    with lock:
                        state["fail"] += 1
                    log(f"  FAIL {tid}: {msg[:90]}")
                    return
                time.sleep(delay)
                delay = min(delay * 1.8, 45)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(embed_one, todo))
    log(f"done: {state['done']} embedded, {state['fail']} failed, {len(vecs)} total cached")

    # Assemble aligned matrix.
    keep_ids = [i for i in ids if i in vecs]
    mat = np.vstack([vecs[i] for i in keep_ids]).astype(np.float32)
    np.savez_compressed(out, ids=np.array(keep_ids), vectors=mat, model=args.model, dim=args.dim)
    log(f"wrote {out} -> {mat.shape}")
    # also a tiny manifest
    out.with_suffix(".json").write_text(json.dumps({
        "n": int(mat.shape[0]), "dim": int(mat.shape[1]), "model": args.model,
    }, indent=2))


if __name__ == "__main__":
    main()
