fmt:
    uv run ruff format src

lint:
    uv run ruff check --fix src

test:
    uv run pytest src

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cd rust && cargo run --bin stub_gen
    just sync-clean

clean:
    rm -rf **/__pycache__
    rm -rf rust/target
    rm -rf .venv
