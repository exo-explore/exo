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
      ];
    in
    inputs.flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.fenix.overlays.default ];
        };
      in
      {
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
              (fenix.complete.withComponents [
                "cargo"
                "rustc"
                "clippy"
                "rustfmt"
                "rust-src"
              ])
              rustup # Just here to make RustRover happy

              # NIX
              nixpkgs-fmt

              # JUST
              just
            ]
            ++ (pkgs.lib.optionals pkgs.stdenv.isDarwin [
              # MACMON
              macmon
            ]);

          shellHook = ''
            # PYTHON
            export DASHBOARD_DIR=$(git rev-parse --show-toplevel)/dashboard;
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.python313}/lib
            echo
            echo "🍎🍎 Run 'just <recipe>' to get started"
            just --list
          '';

        };
      }
    );
}
