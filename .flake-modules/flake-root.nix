# Provides path to project root with:
#   1. ${lib.getExe config.flake-root.package}
#   2. $FLAKE_ROOT environment-varible

# Top-level parameters that are bound to the provider flake
# These are passed from `flake.nix` using importApply
{
  localSelf,
  flake-parts-lib,
  nixpkgs-lib,
  flake-root,
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
  imports = [ flake-root.flakeModule ];
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
      flake-root.projectRootFile = "flake.nix"; # Not necessary, as flake.nix is the default

      make-shells.default = {
        inputsFrom = [ config.flake-root.devShell ]; # Adds $FLAKE_ROOT to environment
      };
    };
}
