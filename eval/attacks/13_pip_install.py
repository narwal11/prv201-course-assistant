# Attack: install arbitrary packages at runtime.
import subprocess, sys
p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "requests-oauthlib"],
                   capture_output=True, text=True, timeout=8)
print("ESCAPED: pip install succeeded") if p.returncode == 0 else print("contained: pip exit", p.returncode)
