FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get -y update \
    && apt-get -y install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update \
    && apt-get -y --no-install-recommends install \
        bash \
        clang \
        curl \
        git \
        iproute2 \
        libgl1 \
        libglib2.0-0 \
        procps \
        protobuf-compiler \
        strace \
        sudo \
        tmux \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
    && apt-get -y upgrade \
    && apt-get -y dist-upgrade \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -d /home/exo exo \
    && usermod -aG adm,audio,video exo \
    && echo "exo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudo \
    && chmod 0440 /etc/sudoers.d/sudo

RUN mkdir -p /home/exo/src && chown -R exo:exo /home/exo/src

COPY --chown=exo:exo . /home/exo/src/

# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/
COPY <<EOF /entrypoint.sh
#!/bin/bash

source .venv/bin/activate

if [ -n "\${NODE_NAME}" ]; then
  exo --node-id=\${NODE_NAME} "\$@"
else
  exo "\$@"
fi
EOF

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

USER exo
WORKDIR /home/exo/src

RUN bash -c "\
    python3.12 -m venv .venv && \
    source .venv/bin/activate && \
        pip install --upgrade pip && \
        pip install -e . && \
        cd exo/networking/grpc && \
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. node_service.proto && \
        sed -i 's/import\ node_service_pb2/from . &/' node_service_pb2_grpc.py && \
    deactivate \
"

RUN bash -c "\
    source .venv/bin/activate && \
        pip install --no-cache-dir \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
        pip install --no-cache-dir flax llvmlite capstone || true && \
    deactivate \
"

EXPOSE 52415/tcp
