# ── Stage 1: build VAL validator ──────────────────────────────────────────────
FROM python:3.11-slim AS val-builder

# Accept proxy from build env (needed on restricted networks)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV http_proxy=$HTTP_PROXY \
    https_proxy=$HTTPS_PROXY \
    no_proxy=$NO_PROXY

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make \
    && rm -rf /var/lib/apt/lists/*

# Clear proxy so it doesn't leak into the final image
ENV http_proxy= https_proxy= no_proxy=

COPY VAL/ /build/VAL/
RUN cmake -S /build/VAL -B /build/VAL/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_DEFAULT_CMP0057=NEW 2>/dev/null \
    && cmake --build /build/VAL/build --config Release --parallel


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Accept proxy for pip downloads
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV http_proxy=$HTTP_PROXY \
    https_proxy=$HTTPS_PROXY \
    no_proxy=$NO_PROXY

# VAL binary from stage 1
COPY --from=val-builder /build/VAL/build/bin/Validate /usr/local/bin/Validate

# CPU-only torch first — prevents sentence-transformers from pulling full CUDA build (~1.5 GB)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

ENV http_proxy= https_proxy= no_proxy=

# Project source (excludes everything in .dockerignore)
COPY . .

# Results written here — mount as a volume to persist on the host
RUN mkdir -p /app/results

ENV MAP_PLANNING_VALIDATE_BIN=/usr/local/bin/Validate

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
