"""Probe: find a working (client config, model) for embeddings with this key."""
import os, sys, traceback
from google import genai
from google.genai import types

KEY = os.environ["GEMINI_API_KEY"]

clients = []
try:
    clients.append(("genai-api-key", genai.Client(api_key=KEY)))
except Exception as e:
    print("client genai-api-key FAILED:", e)
try:
    clients.append(("vertex-express", genai.Client(vertexai=True, api_key=KEY)))
except Exception as e:
    print("client vertex-express FAILED:", e)

models = ["gemini-embedding-002", "gemini-embedding-001",
          "text-embedding-005", "text-embedding-004", "gemini-embedding-exp-03-07"]

# first, try to list models per client
for cname, c in clients:
    print(f"\n=== list models via {cname} ===")
    try:
        got = []
        for m in c.models.list():
            nm = getattr(m, "name", "?")
            if "embed" in nm.lower():
                got.append(nm)
        print("embedding models:", got[:20] if got else "(none/iterate empty)")
    except Exception as e:
        print("  list failed:", str(e)[:160])

print("\n=== embed probes ===")
for cname, c in clients:
    for model in models:
        try:
            r = c.models.embed_content(
                model=model, contents=["a quick test trope about a hero"],
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            dim = len(r.embeddings[0].values)
            print(f"OK  client={cname:14s} model={model:28s} dim={dim}")
        except Exception as e:
            print(f"ERR client={cname:14s} model={model:28s} {str(e)[:110]}")
