# Whisper test set: transcription page and benchmark, independent of the
# recorder and the timeline. Built and run by docker-compose.yml in this folder.
# The scripts and data are mounted from this folder, not copied in.

# Stage 1: build whisper.cpp
FROM debian:bookworm-slim AS whisper

ARG WHISPER_VERSION=v1.9.4

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${WHISPER_VERSION}" https://github.com/ggml-org/whisper.cpp /whisper
WORKDIR /whisper
# Optimised for the CPU of the machine that builds the image, so build it on
# the machine that runs the benchmark.
RUN cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j --config Release --target whisper-cli whisper-quantize

# Stage 2: Python, plus the converter for Hugging Face fine-tunes
FROM python:3.12-slim-bookworm

# WITH_HF=1 adds CPU-only PyTorch and transformers (about 1 GB), needed only to
# convert "hf:" models such as the ATC fine-tunes to ggml.
ARG WITH_HF=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libstdc++6 tzdata curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN if [ "$WITH_HF" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
     && pip install --no-cache-dir numpy transformers safetensors \
     && mkdir -p /opt/openai-whisper/whisper/assets \
     && curl -fL -o /opt/openai-whisper/whisper/assets/mel_filters.npz \
            https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz; \
    fi

COPY --from=whisper /whisper/build/bin/whisper-cli /whisper/build/bin/whisper-quantize /usr/local/bin/
COPY --from=whisper /whisper/models/convert-h5-to-ggml.py /opt/convert-h5-to-ggml.py

# HOME=/tmp: the container runs as your user, which has no home directory in the image
ENV PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /testset
