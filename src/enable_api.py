"""Enable aiplatform.googleapis.com on project selstech using the SA token."""
import json, time
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

SA = r"C:\Users\shift\Desktop\tvtropes-embed\sa.json"
PROJECT = "selstech"
SERVICE = "aiplatform.googleapis.com"

creds = service_account.Credentials.from_service_account_file(
    SA, scopes=["https://www.googleapis.com/auth/cloud-platform"])
sess = AuthorizedSession(creds)

# 1) enable the service
url = f"https://serviceusage.googleapis.com/v1/projects/{PROJECT}/services/{SERVICE}:enable"
r = sess.post(url, json={})
print("enable status:", r.status_code)
print(r.text[:800])

# 2) poll the operation if returned
try:
    op = r.json().get("name")
except Exception:
    op = None
if op and not r.json().get("done"):
    for _ in range(20):
        time.sleep(5)
        o = sess.get(f"https://serviceusage.googleapis.com/v1/{op}")
        d = o.json()
        if d.get("done"):
            print("operation done:", json.dumps(d)[:400]); break
        print("  ...waiting")

# 3) confirm state
chk = sess.get(f"https://serviceusage.googleapis.com/v1/projects/{PROJECT}/services/{SERVICE}")
try:
    print("service state:", chk.json().get("state"))
except Exception:
    print(chk.text[:400])
