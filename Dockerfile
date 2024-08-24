# Nvidia image for GPU support
FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
ENV WORKING_PORT=8080
ENV DEBUG=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV PATH=/usr/local/python3.12/bin:$PATH
ENV NODE_ID=bytebolt-exo

# Install python3.12
RUN apt-get update && \
    apt-get install --no-install-recommends -y git software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.12 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies, configure python3.12 as "python"
RUN pip3 install --no-cache-dir --upgrade requests
RUN ln -fs /usr/bin/python3.12 /usr/bin/python

COPY setup.py .

# Install exo
RUN pip3 install --no-cache-dir .
RUN pip3 install --no-cache-dir tensorflow
RUN pip3 cache purge

COPY . .

# Run command
ENTRYPOINT ["/usr/bin/python"]
CMD ["main.py", "--disable-tui", "--node-id", "$NODE_ID"]
EXPOSE $WORKING_PORT
