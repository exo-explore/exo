# Provides pretty banner & command index for this flake

{ inputs, ... }:
{
  imports = [ inputs.just-flake.flakeModule ];
  perSystem =
    { config, ... }:
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
