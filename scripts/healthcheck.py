"""Container HEALTHCHECK: exit 0 if Streamlit's health endpoint answers."""
import os
import sys
import urllib.request

url = f"http://localhost:{os.getenv('PORT', '8080')}/_stcore/health"
try:
    sys.exit(0 if urllib.request.urlopen(url, timeout=3).status == 200 else 1)
except Exception:
    sys.exit(1)
