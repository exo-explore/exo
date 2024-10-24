# nvidia: --build-arg BASE_IMAGE=nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
ARG BASE_IMAGE=ubuntu:jammy-20240911.1

# Base image
FROM $BASE_IMAGE

# Set environment variables
ENV WORKING_PORT=8080
ENV DEBUG=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH=/usr/local/python3.12/bin:$PATH

# Set pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install dependencies and setup python3.12
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y git gnupg build-essential software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get remove -y python3 python3-dev && \
    apt-get install --no-install-recommends -y python3.12 python3.12-dev python3.12-distutils && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Link python3.12 to python3 and pip3
RUN ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.12 /usr/bin/pip3

# Copy installation files
COPY setup.py .

# Install exo
RUN pip3 install --no-cache-dir . && \
    pip3 cache purge

# Copy source code
# TODO: Change this to copy only the necessary files
COPY . .

# either use ENV NODE_ID or generate a random node id
RUN if [ -z "$NODE_ID" ]; then export NODE_ID=$(uuidgen); fi

# Run command
CMD ["python3", "main.py", "--disable-tui", "--node-id", "$NODE_ID"]

# Expose port
EXPOSE $WORKING_PORT
