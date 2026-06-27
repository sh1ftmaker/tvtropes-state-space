# TVTropes State Space — Gemini Embedding 2

Embed all **30,984 TVTropes** by meaning with **`gemini-embedding-2`**, arrange
them by similarity, and explore the resulting "state space" in an interactive
WebGL demo.

### ▶ Live demo: https://sh1ftmaker.github.io/tvtropes-state-space/

![demo](out/gemini_sacrifice.png)

## What it does
- **Data**: a GitHub clone of the [dhruvilgala/tvtropes](https://github.com/dhruvilgala/tvtropes)
  dataset (30K tropes + descriptions, 1.9M occurrences across film / TV / literature).
- **Embeddings**: every trope's *name + description* is embedded with Google's
  `gemini-embedding-2` (768-dim, L2-normalized) via **Vertex AI**.
- **Genre dimension (primary)**: a curated taxonomy of ~110 narrative genres &
  settings (from Wikipedia's [List of writing genres](https://en.wikipedia.org/wiki/List_of_writing_genres),
  fiction branches) is embedded by the *same* model, so each trope's affinity to
  each genre is a cosine dot product — no re-embedding the 30K tropes. This
  **genre-affinity space is what lays out the map**; tropes are colored by their
  dominant supergenre (14 buckets: Fantasy, Sci-Fi, Horror, Romance, …).
- **Other signals**: a media-mix signature (film/tv/lit) + a "genderedness" score
  give alternate colorings.
- **Layout**: UMAP (2D & 3D) over the genre-affinity space for the map, KMeans for
  genre-coherent clusters, and an exact k-nearest-neighbor graph in the *full*
  embedding space (the genuine "most similar trope" trace — meaning, not just shared genre).
- **Demo**: two linked single-file WebGL views —
  - **2D map** (`index.html`): pan/zoom + touch over all 30,984 points, color by
    genre / semantic cluster / dominant medium / genderedness, size by popularity,
    full-text search, click-to-trace nearest neighbors.
  - **3D flythrough** (`3d.html`, Three.js): fly (WASD + look) or orbit through the
    cloud, render the full neighbor web, and **remap the X/Y/Z axes to any metric** —
    including a per-supergenre affinity axis each (set X=Fantasy, Y=Horror, Z=Sci-Fi
    to regroup the cloud), plus genre-map axes, film/tv/lit share, popularity,
    genderedness, cluster; set Z=flat for a 2D arrangement to compare.
  - **Trope Craft** (`craft.html`): an [Infinite Craft](https://neal.fun/infinite-craft/)-style
    game. Drag two ingredients together and the result is whichever trope sits
    nearest the *sum* of their vectors — fully offline, no LLM. Start from 76
    narrative "words" (Hero, Sword, Betrayal…) + the 107 genres, craft them into
    tropes, then combine tropes for deeper ones (Sword+Magic→SpellBlade,
    Love+Betrayal→RevengeRomance, Cyberpunk+Detective→StreetSamurai). Chips are
    colored by genre; discoveries persist in localStorage.

## Pipeline
```
data/full/TVTropesData/*.csv          # extracted dataset (659MB zip from Google Drive)
        │  build_features.py
        ▼
out/trope_features.parquet            # 30,984 rows: text + media mix + genderedness
        │  embed_gemini.py  (gemini-embedding-2, Vertex AI, 24 workers, resumable)
        ▼
out/embeddings_gemini.npz             # (30984, 768) float32, L2-normalized
        │  embed_genres.py  (genres.py taxonomy -> same model)
        ▼
out/embeddings_genres.npz             # (~110, 768) genre vectors
        │  project.py  (genre-affinity → UMAP 2D/3D + KMeans + semantic kNN)
        ▼
out/points.json                       # everything the 2D/3D demos need
        │  index.html / 3d.html  (WebGL)
        ▼
http://localhost:8731                 # interactive state space

out/embeddings_gemini.npz + embeddings_words.npz (words.py) + embeddings_genres.npz
        │  build_craft.py  (PCA->48d, int8-quantized, base64)
        ▼
out/craft.json                        # 2.9MB: trope vectors + word/genre ingredients
        │  craft.html
        ▼
Trope Craft                           # offline crafting game
```

## Reproduce
```bash
# 1. data
python src/build_features.py --data data --out out/trope_features.parquet

# 2. embeddings — Vertex AI service account (gemini-embedding-2 lives in `global`)
python src/embed_gemini.py --in out/trope_features.parquet \
    --out out/embeddings_gemini.npz --sa sa.json --location global \
    --model gemini-embedding-2 --dim 768 --workers 24
#   (or AI-Studio API key:  set GEMINI_API_KEY  and drop --sa/--location)

# 3. genre vectors (same model, ~110 narrative genres from genres.py)
python src/embed_genres.py --out out/embeddings_genres.npz --sa sa.json --location global

# 4. projection (genre affinity drives the 2D/3D layout)
python src/project.py --emb out/embeddings_gemini.npz \
    --feat out/trope_features.parquet --genres out/embeddings_genres.npz --out out

# 5. crafting game data (words taxonomy -> same model, then PCA-reduced payload)
python src/embed_words.py --out out/embeddings_words.npz --sa sa.json --location global
python src/build_craft.py --emb out/embeddings_gemini.npz --points out/points.json \
    --words out/embeddings_words.npz --genres out/embeddings_genres.npz \
    --out out/craft.json --dim 48

# 6. view
cd out && python -m http.server 8731    # open http://localhost:8731
```

## Notes
- `gemini-embedding-2` processes **one input per request**; the embedder fans
  out single calls across a thread pool and caches each vector to
  `out/embeddings_gemini_cache/` so any rerun resumes instantly.
- On Vertex AI the model is only published in the **`global`** location for this
  project; `us-central1` etc. return 404 (use `gemini-embedding-001` there).
- `embed_local.py` is a TF-IDF+SVD placeholder embedder with the same output
  format — handy for validating the viz without API access.
- **Secrets**: `sa.json` and `secret.key` are credentials — keep them out of any
  commit / share.

## Semantic sanity checks (gemini-embedding-2 nearest neighbors)
- **ChekhovsGun** → Chekhovs Armory / Gag / Skill / Gunman / News / Army / Boomerang
- **HeroicSacrifice** → Taking The Bullet, Redemption Equals Death, Someone Has To
  Die, Dying Moment Of Awesome  *(near-zero lexical overlap — pure meaning)*
