FROM nvcr.io/nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV WORKING_PORT=8080
ENV DEBUG_LEVEL=1

WORKDIR /app

RUN apt-get update && apt-get install -y python3.12

ENV PATH /usr/local/python3.12/bin:$PATH

RUN apt clean && \\
    rm -rf /var/lib/apt/lists \\
    && \\
    rm -rf /var/log/apt/*

RUN pip cache purge

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["DEBUG=$DEBUG_LEVEL", "python3.12", "main.py"]
EXPOSE $WORKING_PORT
