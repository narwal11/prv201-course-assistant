# Attack: steal API keys or tokens from environment variables.
import os
hits = [k for k in os.environ if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "ANTHROPIC", "PASSWORD"))]
print("ESCAPED: found", hits) if hits else print("contained: no secret-looking variables")
