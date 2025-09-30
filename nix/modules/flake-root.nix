# Provides path to project root with:
#   1. ${lib.getExe config.flake-root.package}
#   2. $FLAKE_ROOT environment-varible

# These values would bind to the consumer flake when this flake module is imported:
{ inputs, ... }:

# The actual flake-parts module configuration
{
  imports = [ inputs.flake-root.flakeModule ];
  perSystem =
    { config, ... }:
    {
      flake-root.projectRootFile = "flake.nix"; # Not necessary, as flake.nix is the default

      make-shells.default = {
        inputsFrom = [ config.flake-root.devShell ]; # Adds $FLAKE_ROOT to environment
      };
    };
}
