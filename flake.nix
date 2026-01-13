{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = "exo.cachix.org-1:okq7hl624TBeAR3kV+g39dUFSiaZgLRkLsFBCuJ2NZI=";
    extra-substituters = "https://exo.cachix.org";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule
      ];

      perSystem =
        { config, inputs', pkgs, lib, ... }:
        let
          fenixToolchain = inputs'.fenix.packages.complete;
        in
        {
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixpkgs-fmt.enable = true;
              ruff-format = {
                enable = true;
                excludes = [ "rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi" ];
              };
              rustfmt = {
                enable = true;
                package = fenixToolchain.rustfmt;
              };
              prettier = {
                enable = true;
                includes = [ "*.ts" ];
              };
              swift-format.enable = true;
            };
          };

          checks.lint = pkgs.runCommand "lint-check" { } ''
            export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
            ${pkgs.ruff}/bin/ruff check ${inputs.self}/
            touch $out
          '';

          devShells.default = with pkgs; pkgs.mkShell {
            packages =
              [
                # FORMATTING
                config.treefmt.build.wrapper

                # PYTHON
                python313
                uv
                ruff
                basedpyright

                # RUST
                (fenixToolchain.withComponents [
                  "cargo"
                  "rustc"
                  "clippy"
                  "rustfmt"
                  "rust-src"
                ])
                rustup # Just here to make RustRover happy

                # NIX
                nixpkgs-fmt

                # SVELTE
                nodejs

                # MISC
                just
                jq
              ]
              ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [
                # IFCONFIG
                unixtools.ifconfig

                # Build dependencies for Linux
                pkg-config
                openssl
              ])
              ++ (pkgs.lib.optionals pkgs.stdenv.isDarwin [
                # MACMON
                macmon
              ]);

            shellHook = ''
              # PYTHON
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.python313}/lib"
              ${lib.optionalString pkgs.stdenv.isLinux ''
                # Build environment for Linux
                export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
                export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
              ''}
              echo
              echo "🍎🍎 Run 'just <recipe>' to get started"
              just --list
            '';
          };
        };
    };
}
