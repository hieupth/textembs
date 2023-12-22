# Build stage: environment
FROM continuumio/miniconda3 AS ENVIRONMENT
# Build environment
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda install -y -c conda-forge conda-pack && \
    conda-pack -n textembserve -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
# Unpack environment.
RUN /venv/bin/conda-unpack

# Runtime stage:
FROM debian:buster AS RUNTIME
# Copy /venv from the previous stage:
COPY --from=ENVIRONMENT /venv /venv
COPY app.py .
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 80