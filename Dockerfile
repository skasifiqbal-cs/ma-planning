# ── Stage 1: build VAL validator ──────────────────────────────────────────────
FROM python:3.11-slim AS val-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY VAL/ /build/VAL/
RUN cmake -S /build/VAL -B /build/VAL/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_DEFAULT_CMP0057=NEW 2>/dev/null \
    && cmake --build /build/VAL/build --config Release --parallel


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# VAL binary from stage 1
COPY --from=val-builder /build/VAL/build/bin/Validate /usr/local/bin/Validate

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source (excludes everything in .dockerignore)
COPY . .

# Results written here — mount as a volume to persist on the host
RUN mkdir -p /app/results

ENV MAP_PLANNING_VALIDATE_BIN=/usr/local/bin/Validate

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
