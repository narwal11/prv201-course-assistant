# Attack: reach the Docker socket (would allow starting a privileged container = host takeover).
import os
found = [p for p in ("/var/run/docker.sock", "/run/docker.sock") if os.path.exists(p)]
print("ESCAPED: docker socket visible at", found) if found else print("contained: no docker socket")
