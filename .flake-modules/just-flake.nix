# Provides pretty banner & command index for this flake

# Top-level parameters that are bound to the provider flake
# These are passed from `flake.nix` using importApply
{
  localSelf,
  flake-parts-lib,
  nixpkgs-lib,
  just-flake,
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
  imports = [ just-flake.flakeModule ];
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
      just-flake.features = {
        # treefmt.enable = true;
        # rust.enable = true;
        # convco.enable = true;
        # hello = {
        #   enable = true;
        #   justfile = ''
        #     hello:
        #       echo Hello World
        #   '';
        # };
      };

      make-shells.default = {
        inputsFrom = [ config.just-flake.outputs.devShell ];
      };
    };
}
