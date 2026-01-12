{ inputs, ... }:
{
  perSystem =
    { config, self', inputs', pkgs, lib, ... }:
    let
      # Fenix nightly toolchain with all components
      fenixPkgs = inputs'.fenix.packages;
      rustToolchain = fenixPkgs.complete.withComponents [
        "cargo"
        "rustc"
        "clippy"
        "rustfmt"
        "rust-src"
        "rust-analyzer"
      ];

      # Crane with fenix toolchain
      craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;

      # Source filtering - only include rust/ directory and root Cargo files
      # This ensures changes to Python/docs/etc don't trigger Rust rebuilds
      src = lib.cleanSourceWith {
        src = inputs.self;
        filter =
          path: type:
          let
            baseName = builtins.baseNameOf path;
            parentDir = builtins.dirOf path;
            inRustDir =
              (lib.hasInfix "/rust/" path)
              || (lib.hasSuffix "/rust" parentDir)
              || (baseName == "rust" && type == "directory");
            isRootCargoFile =
              (baseName == "Cargo.toml" || baseName == "Cargo.lock")
              && (builtins.dirOf path == toString inputs.self);
          in
          isRootCargoFile
          || (inRustDir && (craneLib.filterCargoSources path type || lib.hasSuffix ".toml" path || lib.hasSuffix ".md" path));
      };

      # Common arguments for all Rust builds
      commonArgs = {
        inherit src;
        pname = "exo-rust";
        version = "0.0.1";
        strictDeps = true;

        nativeBuildInputs = [
          pkgs.pkg-config
          pkgs.python313 # Required for pyo3-build-config
        ];

        buildInputs = [
          pkgs.openssl
          pkgs.python313 # Required for pyo3 tests
        ];

        OPENSSL_NO_VENDOR = "1";

        # Required for pyo3 tests to find libpython
        LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.python313 ];
      };

      # Build dependencies once for caching
      cargoArtifacts = craneLib.buildDepsOnly (
        commonArgs
        // {
          cargoExtraArgs = "--workspace";
        }
      );
    in
    {
      # Export toolchain for use in treefmt and devShell
      options.rust = {
        toolchain = lib.mkOption {
          type = lib.types.package;
          default = rustToolchain;
          description = "The Rust toolchain to use";
        };
      };

      config = {
        packages = {
          # The system_custodian binary
          system_custodian = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "-p system_custodian";

              meta = {
                description = "System custodian daemon for exo";
                mainProgram = "system_custodian";
              };
            }
          );

          # Python bindings wheel via maturin
          exo_pyo3_bindings = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts;
              pname = "exo_pyo3_bindings";

              nativeBuildInputs = commonArgs.nativeBuildInputs ++ [
                pkgs.maturin
              ];

              buildPhaseCargoCommand = ''
                maturin build \
                  --release \
                  --manylinux off \
                  --manifest-path rust/exo_pyo3_bindings/Cargo.toml \
                  --features "pyo3/extension-module,pyo3/experimental-async" \
                  --interpreter ${pkgs.python313}/bin/python \
                  --out dist
              '';

              # Don't use crane's default install behavior
              doNotPostBuildInstallCargoBinaries = true;

              installPhaseCommand = ''
                mkdir -p $out
                cp dist/*.whl $out/
              '';
            }
          );
        };

        checks = {
          # Full workspace build (all crates)
          cargo-build = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "--workspace";
            }
          );
          # Run tests with nextest
          cargo-nextest = craneLib.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "--workspace";
            }
          );

          # Build documentation
          cargo-doc = craneLib.cargoDoc (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "--workspace";
            }
          );
        };
      };
    };
}
