FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY engine /src/engine
COPY bot /src/bot

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /src/bot/requirements.txt

RUN g++ -O3 -std=c++20 -march=native -DNDEBUG -pthread \
    -I/src/engine/include -I/src/engine \
    /src/engine/bitboard.cpp \
    /src/engine/position.cpp \
    /src/engine/movegen.cpp \
    /src/engine/evaluate.cpp \
    /src/engine/src/eval/evaluator.cpp \
    /src/engine/src/search/search.cpp \
    /src/engine/src/uci/uci.cpp \
    /src/engine/src/main.cpp \
    -o /opt/mythos

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/bot/src:/app"

RUN groupadd --system mythos \
    && useradd --system --gid mythos --create-home --home-dir /home/mythos mythos

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/mythos /app/engine/mythos
COPY bot /app/bot

RUN chown -R mythos:mythos /app /opt/venv \
    && chmod 0755 /app/engine/mythos

USER mythos

EXPOSE 8080

CMD ["python", "-m", "mythos_bot"]
