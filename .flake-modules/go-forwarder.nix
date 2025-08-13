# Configures the Golang support and builds the forwarder
# TODO: split this up in the future as this is unrelated tasks??

# Top-level parameters that are bound to the provider flake
# These are passed from `flake.nix` using importApply
{
  localSelf,
  flake-parts-lib,
  nixpkgs-lib,
  ...
}:

# These values would bind to the consumer flake when this flake module is imported:
{
  config,
  self,
  inputs,
  getSystem,
  moduleWithSystem,
  withSystem,
  ...
}:

# The actual flake-parts module configuration
{
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
      flakeRoot = nixpkgs-lib.getExe config.flake-root.package;

      # Build the networking/forwarder Go utility.
      forwarder = pkgs.buildGoModule {
        pname = "exo-forwarder";
        version = "0.1.0";
        src = "${flakeRoot}/networking/forwarder";

        vendorHash = "sha256-BXIGg2QYqHDz2TNe8hLAGC6jVlffp9766H+WdkkuVgA=";

        # Only the main package at the repository root needs building.
        subPackages = [ "." ];
      };
    in
    {
      packages = {
        inherit forwarder;
      };

      apps = {
        forwarder = {
          type = "app";
          program = "${forwarder}/bin/forwarder";
        };
      };

      make-shells.default = {
        # Go 1.24 compiler – align with go.mod
        packages = [ pkgs.go_1_24 ];

        # TODO: change this into exported env via nix directly???
        shellHook = ''
          export GOPATH=$(mktemp -d)
        '';
      };
    };
}
