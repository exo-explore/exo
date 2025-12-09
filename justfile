fmt:
    nix fmt

lint:
    uv run ruff check --fix

test:
    uv run pytest src

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cargo run --bin stub_gen
    just sync-clean

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
