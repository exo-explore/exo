regenerate-protobufs:
    protoc --proto_path=shared/protobufs/schemas --python_out=shared/protobufs/types --pyi_out=shared/protobufs/types shared/protobufs/schemas/*.proto
    uv run ruff format ./shared/protobufs/types

fmt:
    uv run ruff format master worker shared engines/*

lint:
    uv run ruff check --fix master worker shared engines/*

test:
    uv run pytest master worker shared engines/*

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

protobufs:
    just regenerate-protobufs

build: regenerate-protobufs
    uv build --all-packages