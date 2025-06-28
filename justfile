regenerate-protobufs:
    protoc --proto_path=shared/protobufs/schemas --python_out=shared/protobufs/types shared/protobufs/schemas/*.proto
    uv run ruff format ./shared/protobufs/types