FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    pkg-config \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    liblapacke-dev \
    python3-pip \
    curl \
    git

RUN git clone https://github.com/ml-explore/mlx.git && cd mlx && mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_PREFIX_PATH="/usr/lib/aarch64-linux-gnu" \
      -DLAPACK_LIBRARIES="/usr/lib/aarch64-linux-gnu/liblapack.so" \
      -DBLAS_LIBRARIES="/usr/lib/aarch64-linux-gnu/libopenblas.so" \
      -DLAPACK_INCLUDE_DIRS="/usr/include" && \
    sed -i 's/option(MLX_BUILD_METAL "Build metal backend" ON)/option(MLX_BUILD_METAL "Build metal backend" OFF)/' ../CMakeLists.txt && \
    make -j && \
    make install && \
    cd .. && \
    pip install --no-cache-dir .

COPY setup.py .
COPY exo ./exo

RUN sed -i '/mlx==/d' setup.py && \
    pip install --no-cache-dir .

RUN pip install --no-cache-dir --no-deps mlx-lm==0.18.2

CMD ["exo", "--inference-engine", "mlx"]
