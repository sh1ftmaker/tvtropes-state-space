"""Probe Vertex AI embedding access using the service-account JSON."""
import sys, traceback
from google import genai
from google.genai import types
from google.oauth2 import service_account

SA = r"C:\Users\shift\Desktop\tvtropes-embed\sa.json"
creds = service_account.Credentials.from_service_account_file(
    SA, scopes=["https://www.googleapis.com/auth/cloud-platform"])
print("loaded SA:", creds.service_account_email)

LOCATIONS = ["us-central1", "global"]
MODELS = ["gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview",
          "text-embedding-005", "text-multilingual-embedding-002"]

for loc in LOCATIONS:
    print(f"\n===== location={loc} =====")
    try:
        client = genai.Client(vertexai=True, project="selstech", location=loc, credentials=creds)
    except Exception as e:
        print("  client init failed:", str(e)[:160]); continue
    for model in MODELS:
        try:
            r = client.models.embed_content(
                model=model, contents=["a quick test trope about a hero"],
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            print(f"  OK  {model:32s} dim={len(r.embeddings[0].values)}")
        except Exception as e:
            print(f"  ERR {model:32s} {str(e)[:130]}")
