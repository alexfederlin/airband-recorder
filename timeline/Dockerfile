# Stage 1: build whisper.cpp and download the speech model
FROM debian:bookworm-slim AS whisper

ARG WHISPER_VERSION=v1.9.4
ARG WHISPER_MODEL_NAME=base

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${WHISPER_VERSION}" https://github.com/ggml-org/whisper.cpp /whisper
WORKDIR /whisper
# Static build so only the one binary needs copying. It is optimised for the
# CPU of the machine that builds the image, so build it on the machine that runs it.
RUN cmake -B build -DBUILD_SHARED_LIBS=OFF \
 && cmake --build build -j --config Release --target whisper-cli
RUN sh ./models/download-ggml-model.sh "${WHISPER_MODEL_NAME}"

# Stage 2: the timeline server
FROM python:3.12-slim-bookworm

ARG WHISPER_MODEL_NAME=base

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libstdc++6 tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY --from=whisper /whisper/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper /whisper/models/ggml-${WHISPER_MODEL_NAME}.bin /models/

ENV WHISPER_BIN=/usr/local/bin/whisper-cli \
    WHISPER_MODEL=/models/ggml-${WHISPER_MODEL_NAME}.bin \
    ARCHIVE=/data/archive \
    PORT=8090 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY server.py index.html ./

EXPOSE 8090
CMD ["python3", "server.py"]
