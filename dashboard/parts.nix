{ inputs, ... }:
{
  perSystem =
    { pkgs, lib, ... }:
    let
      # Filter source to ONLY include package.json and package-lock.json
      # This ensures prettier-svelte only rebuilds when lockfiles change
      dashboardLockfileSrc = lib.cleanSourceWith {
        src = inputs.self;
        filter =
          path: type:
          let
            baseName = builtins.baseNameOf path;
            isDashboardDir = baseName == "dashboard" && type == "directory";
            isPackageFile =
              (lib.hasInfix "/dashboard/" path || lib.hasSuffix "/dashboard" (builtins.dirOf path))
              && (baseName == "package.json" || baseName == "package-lock.json");
          in
          isDashboardDir || isPackageFile;
      };

      # Stub source with lockfiles and minimal files for build to succeed
      # This allows prettier-svelte to avoid rebuilding when dashboard source changes
      dashboardStubSrc = pkgs.runCommand "dashboard-stub-src" { } ''
        mkdir -p $out
        cp ${dashboardLockfileSrc}/dashboard/package.json $out/
        cp ${dashboardLockfileSrc}/dashboard/package-lock.json $out/
        # Minimal files so vite build succeeds (produces empty output)
        echo '<!DOCTYPE html><html><head></head><body></body></html>' > $out/index.html
        mkdir -p $out/src
        touch $out/src/app.html
      '';

      # Deps-only build using stub source (for prettier-svelte)
      # Only rebuilds when package.json or package-lock.json change
      dashboardDeps = inputs.dream2nix.lib.evalModules {
        packageSets.nixpkgs = pkgs;
        modules = [
          ./dashboard.nix
          {
            paths.projectRoot = inputs.self;
            paths.projectRootFile = "flake.nix";
            paths.package = inputs.self + "/dashboard";
          }
          {
            deps.dashboardSrc = lib.mkForce dashboardStubSrc;
          }
          # Override build phases to skip the actual build - just need node_modules
          {
            mkDerivation = {
              buildPhase = lib.mkForce "true";
              installPhase = lib.mkForce ''
                runHook preInstall
                runHook postInstall
              '';
            };
          }
        ];
      };

      # Filter source to only include dashboard directory
      dashboardSrc = lib.cleanSourceWith {
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
            deps.dashboardSrc = lib.mkForce "${dashboardSrc}/dashboard";
          }
        ];
      };
    in
    {
      # Extract just the static site from the full build
      packages.dashboard = pkgs.runCommand "exo-dashboard" { } ''
        cp -r ${dashboardFull}/build $out
      '';

      # Prettier with svelte plugin for treefmt
      # Uses dashboardDeps instead of dashboardFull to avoid rebuilding on source changes
      packages.prettier-svelte = pkgs.writeShellScriptBin "prettier-svelte" ''
        export NODE_PATH="${dashboardDeps}/lib/node_modules/exo-dashboard/node_modules"
        exec ${pkgs.nodejs}/bin/node \
          ${dashboardDeps}/lib/node_modules/exo-dashboard/node_modules/prettier/bin/prettier.cjs \
          --plugin "${dashboardDeps}/lib/node_modules/exo-dashboard/node_modules/prettier-plugin-svelte/plugin.js" \
          "$@"
      '';
    };
}
