default:
    @just --list

fmt:
    uv run ruff format master worker shared engines/*

lint:
    uv run ruff check --fix master worker shared engines/*

lint-check:
    uv run ruff check master worker shared engines/*

test:
    uv run pytest master worker shared engines/*

test-fast:
    uv run pytest master shared engines/*

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

build:
    uv build --all-packages

# Build the Go forwarder binary
build-forwarder:
    HASH=$(uv run scripts/hashdir.py) && go build -C networking/forwarder -buildvcs=false -o $GO_BUILD_DIR/forwarder -ldflags "-X 'main.SourceHash=${HASH}'" 
    chmod 0755 $GO_BUILD_DIR/forwarder

# Run forwarder tests
test-forwarder:
    cd networking/forwarder && go test ./src/...

# Build all components (Python packages and Go forwarder)
build-all: build build-forwarder

run n="1" clean="false":
    @echo "→ Spinning up {{n}} node(s) (clean={{clean}})"
    if [ "{{clean}}" = "true" ]; then ./run.sh -c; else ./run.sh; fi
    if [ "{{n}}" -gt 1 ]; then \
        for i in $(seq 2 "{{n}}"); do \
            if [ "{{clean}}" = "true" ]; then ./run.sh -rc; else ./run.sh -r; fi; \
        done; \
    fi
