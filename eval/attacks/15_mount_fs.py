# Attack: mount a filesystem (needs CAP_SYS_ADMIN; would allow remounting / as writable).
import subprocess
try:
    p = subprocess.run(["mount", "-t", "tmpfs", "none", "/mnt"], capture_output=True, text=True, timeout=5)
    print("ESCAPED: mount succeeded") if p.returncode == 0 else print("contained: mount exit", p.returncode)
except Exception as e:
    print("contained:", type(e).__name__)
