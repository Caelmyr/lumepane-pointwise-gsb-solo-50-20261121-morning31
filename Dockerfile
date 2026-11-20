ARG LUMEPANE_DEPS_IMAGE=python:3.11-slim
FROM ${LUMEPANE_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/lumepane
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 LUMEPANE_DATA=/var/lib/lumepane
WORKDIR /opt/lumepane
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY lumepane ./lumepane
COPY static ./static
RUN useradd -U -u 11503 lumepane && install -d -o lumepane -g lumepane /var/lib/lumepane
USER lumepane
EXPOSE 8000
CMD ["python", "-m", "lumepane"]
