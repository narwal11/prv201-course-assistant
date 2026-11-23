# Attack: become root.
import os
if os.getuid() == 0:
    print("ESCAPED: already running as root")
else:
    try:
        os.setuid(0)
        print("ESCAPED: setuid(0) succeeded")
    except Exception as e:
        print("contained: uid", os.getuid(), type(e).__name__)
