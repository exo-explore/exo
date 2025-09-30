# Single module responsible for collecting all overlays and instantiating in one go

{
  flake-parts-lib,
  inputs,
  self,
  specialArgs,
  ...
}:
let
  inherit (flake-parts-lib) mkPerSystemOption;
in
{
  options.perSystem = mkPerSystemOption (
    {
      system,
      config,
      lib,
      options,
      pkgs,
      self',
      ...
    }@args:
    let
      inherit (lib.types)
        attrsOf
        listOf
        submoduleWith
        raw
        ;
    in
    {
      options.pkgs-init.overlays = lib.mkOption {
        description = ''
          List of nixpkgs overlays (functions of the form: final: prev: { ... }).
          Any module can append. Order matters.
        '';
        default = [ ];
        example = [
          (final: prev: {
            my-hello = prev.hello;
          })
        ];
        type = lib.types.listOf lib.types.unspecified;
      };
      options.pkgs-init.importArgs = lib.mkOption {
        description = "Extra arguments merged into the nixpkgs import call.";
        default = { };
        type = lib.types.attrs;
      };
      config = {
        _module.args.pkgs = import inputs.nixpkgs (
          {
            inherit system;
            overlays = config.pkgs-init.overlays;
          }
          // config.pkgs-init.importArgs
        );
      };
    }
  );
}
