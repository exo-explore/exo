export NIX_CONFIG := "extra-experimental-features = nix-command flakes"

default: lint fmt
all: lint fmt check

fmt:
    treefmt || nix fmt

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
    uv sync --reinstall-package exo_pyo3_bindings

build-dashboard:
    #!/usr/bin/env bash
    pushd dashboard
    npm install
    npm run build
    popd

package: build-dashboard
    uv run pyinstaller packaging/pyinstaller/exo.spec

build-app: package
    xcodebuild build -project app/EXO/EXO.xcodeproj -scheme EXO -configuration Debug -derivedDataPath app/EXO/build
    @echo "\nBuild complete. Run with:\n  open {{justfile_directory()}}/app/EXO/build/Build/Products/Debug/EXO.app"

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard/node_modules
    rm -rf dashboard/.svelte-kit
    rm -rf dashboard/build
