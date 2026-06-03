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
    #!/usr/bin/env bash
    export UV_NO_SYNC=1
    # build stubs
    PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" cargo run --bin stub_gen
    # install with maturin -> incremental builds
    cd rust/exo_rs
    maturin develop --uv
    cd ../..
    # symlink exo_rs to .nix-devel site packages
    dst=".nix-devel/site-packages"
    src="$(uv run python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    mkdir -p "$dst"
    rm -rf "$dst/exo_rs" "$dst"/exo_rs-*.dist-info
    ln -s "$src/exo_rs" "$dst/exo_rs"
    for dist_info in "$src"/exo_rs-*.dist-info; do
        ln -s "$dist_info" "$dst/$(basename "$dist_info")"
    done

build-dashboard:
    #!/usr/bin/env bash
    pushd dashboard
    npm install
    npm run build
    popd

package: build-dashboard
    uv run pyinstaller packaging/pyinstaller/exo.spec
    rm -rf build

build-app: rust-rebuild sync-clean package
    env -u LD xcodebuild build -project app/EXO/EXO.xcodeproj -scheme EXO -configuration Debug -derivedDataPath app/EXO/build
    @echo "\nBuild complete. Run with:\n  open {{justfile_directory()}}/app/EXO/build/Build/Products/Debug/EXO.app"

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard/node_modules
    rm -rf dashboard/.svelte-kit
    rm -rf dashboard/build
