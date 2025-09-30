# Configures Python shell

{
  perSystem =
    { pkgs, ... }:
    {
      make-shells.default = {
        packages = [
          pkgs.python313
          pkgs.uv
          pkgs.ruff
          pkgs.basedpyright
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.python313}/lib
        '';
      };
    };
}
