{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Use flake-parts for modular configs
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    # Flake-parts wrapper for mkShell
    make-shell.url = "github:nicknovitski/make-shell";

    # Provides path to project root with:
    #   1. ${lib.getExe config.flake-root.package}
    #   2. $FLAKE_ROOT environment-varible
    flake-root.url = "github:srid/flake-root";

    # Provides flake integration with [Just](https://just.systems/man/en/)
    just-flake.url = "github:juspay/just-flake";

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
    inputs@{
      flake-parts,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { flake-parts-lib, self, ... }:
      {
        imports = [
          inputs.make-shell.flakeModules.default

          ./nix/modules/pkgs-init.nix # nixpkgs overlays manager
          ./nix/modules/flake-root.nix
          ./nix/modules/just-flake.nix
          ./nix/modules/macmon.nix
          ./nix/modules/python.nix
          ./nix/modules/rust.nix
          ./nix/modules/go-forwarder.nix
        ];
        systems = [
          "x86_64-linux"
          "aarch64-darwin"
        ];
        perSystem =
          {
            config,
            self',
            inputs',
            pkgs,
            system,
            ...
          }:
          {
            # Per-system attributes can be defined here. The self' and inputs'
            # module parameters provide easy access to attributes of the same
            # system.
            # NOTE: pkgs is equivalent to inputs'.nixpkgs.legacyPackages.hello;
            apps = { };

            make-shells.default = {
              packages = [
                pkgs.protobuf
              ];

              nativeBuildInputs = with pkgs; [
                nixpkgs-fmt
              ];

              shellHook = ''
                export GO_BUILD_DIR=$(git rev-parse --show-toplevel)/build;
                export DASHBOARD_DIR=$(git rev-parse --show-toplevel)/dashboard;
              '';

              # Arguments which are intended to be environment variables in the shell environment
              # should be changed to attributes of the `env` option
              env = { };

              # Arbitrary mkDerivation arguments should be changed to be attributes of the `additionalArguments` option
              additionalArguments = { };
            };
          };
        flake = {
          # The usual flake attributes can be defined here, including system-
          # agnostic ones like nixosModule and system-enumerating ones, although
          # those are more easily expressed in perSystem.

        };
      }
    );
}
