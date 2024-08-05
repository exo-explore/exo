FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

ENV WORKING_PORT=8080
ENV DEBUG_LEVEL=1

WORKDIR /app

ENV PATH /usr/local/python3.12/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN apt-get install git -y

RUN apt clean &&  \
    rm -rf /var/lib/apt/lists \
    && \
    rm -rf /var/log/apt/*

RUN curl -sSL https://install.python-poetry.org | python3.10 - --preview
RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.10 /usr/bin/python


COPY . .

RUN pip install --no-cache-dir .
RUN pip cache purge

CMD ["DEBUG=$DEBUG_LEVEL", "python3.12", "main.py"]
EXPOSE $WORKING_PORT
