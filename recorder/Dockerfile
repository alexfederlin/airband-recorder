FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      alsa-utils sox libsox-fmt-all curl ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY scanner-recorder.sh /usr/local/bin/scanner-recorder.sh
RUN chmod +x /usr/local/bin/scanner-recorder.sh

CMD ["/usr/local/bin/scanner-recorder.sh"]
