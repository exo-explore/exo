# Provides macmon binary for the worker.

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
    {
      make-shells.default = {
        packages = if (system == "aarch64-darwin") then ([ pkgs.macmon ]) else ([]);
      };
    };
}
