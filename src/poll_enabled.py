"""Poll until aiplatform.googleapis.com is ENABLED on selstech, then exit 0.
Exits 3 if it gives up. Also verifies a real embed call succeeds before exiting 0.
"""
import sys, time
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from google import genai
from google.genai import types

SA = r"C:\Users\shift\Desktop\tvtropes-embed\sa.json"
PROJECT, SERVICE = "selstech", "aiplatform.googleapis.com"
creds = service_account.Credentials.from_service_account_file(
    SA, scopes=["https://www.googleapis.com/auth/cloud-platform"])
sess = AuthorizedSession(creds)

DEADLINE = time.time() + 60 * 50  # watch up to 50 minutes
while time.time() < DEADLINE:
    try:
        r = sess.get(f"https://serviceusage.googleapis.com/v1/projects/{PROJECT}/services/{SERVICE}")
        state = r.json().get("state")
    except Exception as e:
        state = f"check-error: {e}"
    if state == "ENABLED":
        # confirm an actual embed works (covers billing + role propagation)
        try:
            c = genai.Client(vertexai=True, project=PROJECT, location="us-central1", credentials=creds)
            for model in ("gemini-embedding-2", "gemini-embedding-001"):
                try:
                    c.models.embed_content(model=model, contents=["ping"],
                        config=types.EmbedContentConfig(output_dimensionality=768))
                    print(f"READY model={model}")
                    sys.exit(0)
                except Exception as e:
                    print(f"enabled but {model} not callable yet: {str(e)[:90]}")
        except Exception as e:
            print("client err:", str(e)[:90])
    print(f"state={state}; waiting...", flush=True)
    time.sleep(45)

print("gave up waiting")
sys.exit(3)
