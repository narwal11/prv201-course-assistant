# Attack: fill the writable /tmp to exhaust host disk. Also leaves a marker that the
# runner later checks for, to prove containers do not persist between runs.
open("/tmp/persist_marker", "w").write("x")
written = 0
try:
    with open("/tmp/fill.bin", "wb") as f:
        for _ in range(64):
            f.write(b"\0" * 1024 * 1024)
            f.flush()
            written += 1
    print("ESCAPED: wrote", written, "MB to /tmp")
except OSError as e:
    print("contained after", written, "MB:", type(e).__name__)
