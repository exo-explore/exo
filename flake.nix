{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Provides Rust dev-env integration:
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Provides formatting infrastructure:
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # TODO: figure out caching story
  # nixConfig = {
  #   # nix community cachix
  #   extra-trusted-public-keys = "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs=";
  #   extra-substituters = "https://nix-community.cachix.org";
  # };

  outputs =
    inputs:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];
      fenixToolchain = system: inputs.fenix.packages.${system}.complete;
    in
    inputs.flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.fenix.overlays.default ];
        };
        treefmtEval = inputs.treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs = {
            nixpkgs-fmt.enable = true;
            ruff-format = {
              enable = true;
              excludes = [ "rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi" ];
            };
            rustfmt = {
              enable = true;
              package = (fenixToolchain system).rustfmt;
            };
            prettier = {
              enable = true;
              includes = [ "*.ts" ];
            };
            swift-format.enable = true;
          };
        };
      in
      {
        formatter = treefmtEval.config.build.wrapper;
        checks.formatting = treefmtEval.config.build.check inputs.self;
        checks.lint = pkgs.runCommand "lint-check" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}/
          touch $out
        '';

        devShells.default = pkgs.mkShell {
          packages =
            with pkgs;
            [
              # PYTHON
              python313
              uv
              ruff
              basedpyright

              # RUST
              ((fenixToolchain system).withComponents [
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
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              # Build environment for Linux
              export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
            ''}
            echo
            echo "🍎🍎 Run 'just <recipe>' to get started"
            just --list
          '';

        };
      }
    );
}
