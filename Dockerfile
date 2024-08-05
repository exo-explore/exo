FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

ENV WORKING_PORT=8080
ENV DEBUG=1

WORKDIR /app

ENV PATH /usr/local/python3.12/bin:$PATH
ENV NODE_ID=bytebolt-exo

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.12 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN apt-get install git -y

RUN apt clean &&  \
    rm -rf /var/lib/apt/lists \
    && \
    rm -rf /var/log/apt/*

RUN curl -sSL https://install.python-poetry.org | python3.10 - --preview
RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.12 /usr/bin/python


COPY . .

RUN pip install --no-cache-dir .
RUN pip cache purge

ENTRYPOINT ["/usr/bin/python"]
CMD ["main.py", "--disable-tui", "--node-id", "$NODE_ID"]
EXPOSE $WORKING_PORT
