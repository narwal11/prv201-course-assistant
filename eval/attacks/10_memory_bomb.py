# Attack: allocate (and touch) 2 GB of RAM to starve the host.
try:
    blob = b"\x01" * (2 * 1024 ** 3)
    print("ESCAPED: allocated", len(blob), "bytes")
except MemoryError:
    print("contained: MemoryError")
