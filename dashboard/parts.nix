{ inputs, ... }:
{
  perSystem =
    { pkgs, lib, ... }:
    let
      # Filter source to only include dashboard directory
      src = lib.cleanSourceWith {
        src = inputs.self;
        filter =
          path: type:
          let
            baseName = builtins.baseNameOf path;
            inDashboardDir =
              (lib.hasInfix "/dashboard/" path)
              || (lib.hasSuffix "/dashboard" (builtins.dirOf path))
              || (baseName == "dashboard" && type == "directory");
          in
          inDashboardDir;
      };

      # Build the dashboard with dream2nix (includes node_modules in output)
      dashboardFull = inputs.dream2nix.lib.evalModules {
        packageSets.nixpkgs = pkgs;
        modules = [
          ./dashboard.nix
          {
            paths.projectRoot = inputs.self;
            paths.projectRootFile = "flake.nix";
            paths.package = inputs.self + "/dashboard";
          }
          # Inject the filtered source
          {
            deps.dashboardSrc = lib.mkForce "${src}/dashboard";
          }
        ];
      };
    in
    {
      # Extract just the static site from the full build
      packages.dashboard = pkgs.runCommand "exo-dashboard" { } ''
        cp -r ${dashboardFull}/build $out
      '';
    };
}
