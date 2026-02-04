{ inputs, ... }:
{
  perSystem =
    { pkgs, lib, ... }:
    let
      # Filter source to only include resources directory
      resourcesSrc = lib.cleanSourceWith {
        src = inputs.self + "/resources";
      };
    in
    {
      packages.resources = pkgs.runCommand "exo-resources" { } ''
        cp -r ${resourcesSrc} $out
      '';
    };
}

