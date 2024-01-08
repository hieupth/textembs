# Build stage:
FROM hieupth/mamba AS build

ADD . .
RUN apt-get update && \
    apt-get install -y build-essential pkg-config libssl-dev && \
    mamba install -c conda-forge conda-pack && \
    mamba env create -f environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "textembserve", "/bin/bash", "-c"]
# 
RUN pip install .
#
RUN conda-pack -n textembserve -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
#
RUN /venv/bin/conda-unpack

# Runtime stage:
FROM debian:buster AS runtime
# Copy /venv from the previous stage:
COPY --from=build /venv /venv
#
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 80