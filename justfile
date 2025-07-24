regenerate-protobufs:
    #!/usr/bin/env bash
    if [ -f shared/protobufs/schemas/*.proto ]; then
        protoc --proto_path=shared/protobufs/schemas --python_out=shared/protobufs/types --pyi_out=shared/protobufs/types shared/protobufs/schemas/*.proto
        uv run ruff format ./shared/protobufs/types
    else
        echo "No .proto files found in shared/protobufs/schemas/"
    fi

fmt:
    uv run ruff format master worker shared engines/*

lint:
    uv run ruff check --fix master worker shared engines/*

lint-check:
    uv run ruff check master worker shared engines/*

test:
    uv run pytest master worker shared engines/*

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall

protobufs:
    just regenerate-protobufs

build: regenerate-protobufs
    uv build --all-packages

# Build the Go forwarder binary
build-forwarder:
    cd networking/forwarder && go build -buildvcs=false -o ../../build/forwarder .

# Run forwarder tests
test-forwarder:
    cd networking/forwarder && go test ./src/...

# Build all components (Python packages and Go forwarder)
build-all: build build-forwarder