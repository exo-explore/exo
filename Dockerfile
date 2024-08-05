FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

ENV WORKING_PORT=8080
ENV DEBUG=1

WORKDIR /app

ENV PATH /usr/local/python3.12/bin:$PATH
ENV NODE_ID=bytebolt-exo

RUN apt update
RUN apt install -y wget libffi-dev gcc build-essential  \
    curl tcl-dev tk-dev uuid-dev lzma-dev liblzma-dev libssl-dev libsqlite3-dev \
    git


RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.12.0.tgz
RUN tar -zxvf Python-3.12.0.tgz
RUN cd Python-3.12.0 && ./configure --prefix=/opt/python3.12 && make && make install

# Delete the python source code and temp files
RUN rm Python-3.12.0.tgz
RUN rm -r Python-3.12.0/

# Now link it so that $python works
RUN ln -s /opt/python3.12/python3.12 /usr/bin/python



RUN apt clean &&  \
    rm -rf /var/lib/apt/lists \
    && \
    rm -rf /var/log/apt/*

RUN curl -sSL https://install.python-poetry.org | python3.12 - --preview
RUN pip3 install --upgrade requests


COPY . .

RUN pip install --no-cache-dir .
RUN pip cache purge

ENTRYPOINT ["/usr/bin/python"]
CMD ["main.py", "--disable-tui", "--node-id", "$NODE_ID"]
EXPOSE $WORKING_PORT
