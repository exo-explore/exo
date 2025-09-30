{
  perSystem =
    {
      config,
      pkgs,
      lib,
      ...
    }:
    {
      make-shells.default = {
        # Go 1.24 compiler – align with go.mod
        packages = [ pkgs.go_1_24 ];
        shellHook = ''
          GOPATH="''$(${lib.getExe config.flake-root.package})"/.go_cache
          export GOPATH
        '';
      };
    };
}
