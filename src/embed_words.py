"""Embed the primitive narrative "words" with the SAME model as the tropes.

Same approach as embed_genres.py: ~80 base ingredients (words.py) embedded by
gemini-embedding-2 into the same 768-dim space as the tropes, so a combination's
"sum vector" can be matched against tropes by plain cosine. Cheap (~80 vectors).

Usage:
  python embed_words.py --out out/embeddings_words.npz --sa sa.json --location global
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from words import WORDS, ROOTS, word_text  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="gemini-embedding-2")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--task-type", default="SEMANTIC_SIMILARITY")
    ap.add_argument("--sa", default=None, help="service-account JSON path (Vertex)")
    ap.add_argument("--project", default="selstech")
    ap.add_argument("--location", default="global")
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    if args.sa:
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            args.sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location, credentials=creds)
        log(f"auth: Vertex AI project={args.project} loc={args.location} model={args.model}")
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log("ERROR: set GEMINI_API_KEY or pass --sa")
            sys.exit(2)
        client = genai.Client(api_key=api_key)

    cfg = types.EmbedContentConfig(task_type=args.task_type, output_dimensionality=args.dim)

    # roots embedded alongside the words (kind="root") so the game's primordial
    # 4 share the exact same model + space as everything they craft into.
    entries = [(n, "root", g) for n, g in ROOTS] + list(WORDS)

    names, kinds, vecs = [], [], []
    for name, kind, gloss in entries:
        text = word_text(name, gloss)
        delay = 2.0
        for attempt in range(8):
            try:
                resp = client.models.embed_content(model=args.model, contents=text, config=cfg)
                emb = np.asarray(resp.embeddings[0].values, dtype=np.float32)
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
                names.append(name)
                kinds.append(kind)
                vecs.append(emb)
                log(f"  [{len(names):3d}/{len(entries)}] {name}")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 7:
                    log(f"  FAIL {name}: {str(e)[:120]}")
                    raise
                time.sleep(delay)
                delay = min(delay * 1.8, 45)

    mat = np.vstack(vecs).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array(names), kinds=np.array(kinds),
                        vectors=mat, model=args.model, dim=args.dim)
    log(f"wrote {out} -> {mat.shape}")
    out.with_suffix(".json").write_text(json.dumps(
        {"n": int(mat.shape[0]), "dim": int(mat.shape[1]), "names": names}, indent=2))


if __name__ == "__main__":
    main()
