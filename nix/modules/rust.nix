# Configures Rust shell

{ inputs, ... }:
{
  perSystem =
    { pkgs, ... }:
    {
      pkgs-init.overlays = [
        inputs.fenix.overlays.default
      ];

      make-shells.default = {
        packages = [
          (pkgs.fenix.complete.withComponents [
            "cargo"
            "rustc"
            "clippy"
            "rustfmt"
            "rust-src"
          ])
          pkgs.rustup # literally only added to make RustRover happy (otherwise useless)
        ];
      };
    };
}
