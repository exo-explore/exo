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
  };

  outputs =
    inputs@{
      flake-parts,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } (
      {
        flake-parts-lib,
        self,
        ...
      }:
      let
        nixpkgs-lib = inputs.nixpkgs.lib;

        # A wraper around importApply that supplies default parameters
        importApply' =
          path: extraParams:
          (flake-parts-lib.importApply path (
            nixpkgs-lib.recursiveUpdate {
              localSelf = self;
              inherit flake-parts-lib;
              inherit nixpkgs-lib;
            } extraParams
          ));

        # instantiate all the flake modules, passing custom arguments to them as needed
        flakeModules = {
          flakeRoot = importApply' ./.flake-modules/flake-root.nix { inherit (inputs) flake-root; };
          goForwarder = importApply' ./.flake-modules/go-forwarder.nix { };
        };
      in
      {
        imports = [
          inputs.make-shell.flakeModules.default
          flakeModules.flakeRoot
          flakeModules.goForwarder
          ./.flake-modules/macmon.nix
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
          let
            buildInputs = with pkgs; [
            ];
            nativeBuildInputs = with pkgs; [
            ];
          in
          {
            # Per-system attributes can be defined here. The self' and inputs'
            # module parameters provide easy access to attributes of the same
            # system.
            # NOTE: pkgs is equivalent to inputs'.nixpkgs.legacyPackages.hello;
            apps = {
              python-lsp = {
                type = "app";
                program = "${pkgs.basedpyright}/bin/basedpyright-langserver";
              };
              default = self'.apps.forwarder;
            };

            make-shells.default = {
              packages = [
                pkgs.python313
                pkgs.uv
                pkgs.protobuf
                pkgs.basedpyright
                pkgs.ruff
              ];

              nativeBuildInputs =
                with pkgs;
                [
                  nixpkgs-fmt
                  cmake
                ]
                ++ buildInputs
                ++ nativeBuildInputs;

              # Arguments which are intended to be environment variables in the shell environment
              # should be changed to attributes of the `env` option
              env = {
                # fixes libstdc++.so issues and libgl.so issues
                LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
              };

              shellHook = ''
                export GO_BUILD_DIR=$(git rev-parse --show-toplevel)/build;
                export DASHBOARD_DIR=$(git rev-parse --show-toplevel)/dashboard;
              '';

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
