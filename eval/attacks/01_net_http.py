# Attack: reach the internet over HTTP (data exfiltration / downloading payloads).
import urllib.request
try:
    urllib.request.urlopen("http://example.com", timeout=3).read(100)
    print("ESCAPED: HTTP request succeeded")
except Exception as e:
    print("contained:", type(e).__name__)
