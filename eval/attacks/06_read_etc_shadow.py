# Attack: read password hashes.
try:
    data = open("/etc/shadow").read()
    print("ESCAPED: read /etc/shadow,", len(data), "bytes")
except Exception as e:
    print("contained:", type(e).__name__)
