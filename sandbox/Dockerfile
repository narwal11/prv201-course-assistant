# Image used by the run_python tool. Build once:
#   docker build -t prv201-sandbox:latest sandbox/
# All runtime limits (no network, read-only, non-root, 512 MB, 10 s) are applied
# by `docker run` flags in src/assistant/sandbox.py, not here.
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy pandas scikit-learn matplotlib \
    && python -c "import numpy, pandas, sklearn, matplotlib"

# Run as nobody; nothing in the image is writable by that user.
USER 65534:65534
WORKDIR /tmp
