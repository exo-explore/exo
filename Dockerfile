FROM node:latest
EXPOSE 52415
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH
WORKDIR /home/exo
COPY --from=rust:latest /usr/local/cargo /usr/local/cargo
COPY --from=rust:latest /usr/local/rustup /usr/local/rustup
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY . . 
RUN rustup toolchain install nightly && uv sync
WORKDIR /home/exo/dashboard
RUN npm install && npm run build
WORKDIR /home/exo
CMD [ "uv", "run", "exo" ]