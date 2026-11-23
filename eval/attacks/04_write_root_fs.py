# Attack: write a file to the container root filesystem.
try:
    with open("/pwned.txt", "w") as f:
        f.write("x")
    print("ESCAPED: wrote /pwned.txt")
except Exception as e:
    print("contained:", type(e).__name__)
