{ inputs, ... }:
{
  perSystem = { system, ... }:
    let
      root = inputs.self + /.;

      # common settings for all nixpkgs instantiation
      common = {
        allowUnfreePredicate = pkg:
          # Allow unfree for metal-toolchain (needed for Darwin Metal packages)
          (pkg.pname or "") == "metal-toolchain";

        overlays = [
          # SEE: at some point, figure out a better story for overlays...?
          #      maybe its own folder or something...? 
          (import "${root}/nix/apple-sdk-overlay.nix")
          (final: prev: {
            macmon = prev.macmon.overrideAttrs (_: {
              version = "git";
              src = final.fetchFromGitHub {
                owner = "swiftraccoon";
                repo = "macmon";
                rev = "9154d234f763fbeffdcb4135d0bbbaf80609699b";
                hash = "sha256-CwhilKNbs5XL9/tF5DMwyPBlE/hpmjGNTuxQ36sM50M=";
              };
            });
          })
        ];
      };

      # Creates pkgs instantiation with common settings, and optionally 
      # configurable knobs to tweak overlays and allowed packages and so on
      mkPkgs =
        { nixpkgs
        , allowUnfreePredicate ? p: true
        , overlays ? [ ]
        }: import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg:
            (common.allowUnfreePredicate pkg) && (allowUnfreePredicate pkg);
          overlays = common.overlays ++ overlays;
        };

      # latest stable and unstable nixpkgs
      pkgs = mkPkgs { nixpkgs = inputs.nixpkgs; };
      pkgs-unstable = mkPkgs { nixpkgs = inputs.nixpkgs-unstable; };

      # add specific versions of nixpkgs if needed
      #pkgs-23_11 = mkPkgs { nixpkgs = inputs.nixpkgs-23_11; };
    in
    {
      _module.args = {
        # combine into single object for nice autocomplete
        pkgs = pkgs // {
          unstable = pkgs-unstable;
        };
      };
    };
}
