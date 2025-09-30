{
  perSystem =
    { lib, pkgs, ... }:
    lib.mkMerge [
      (lib.mkIf pkgs.stdenv.isDarwin {
        make-shells.default = {
          packages = [ pkgs.macmon ];
        };
      })
    ];

}
