{ lib
, config
, dream2nix
, ...
}:
let
  # Read and parse the lock file
  rawLockFile = builtins.fromJSON (builtins.readFile "${config.deps.dashboardSrc}/package-lock.json");

  # For packages with bundleDependencies, filter out deps that are bundled
  # (bundled deps are inside the tarball, not separate lockfile entries)
  fixedPackages = lib.mapAttrs
    (path: entry:
      if entry ? bundleDependencies && entry.bundleDependencies != [ ]
      then entry // {
        dependencies = lib.filterAttrs
          (name: _: !(lib.elem name entry.bundleDependencies))
          (entry.dependencies or { });
      }
      else entry
    )
    (rawLockFile.packages or { });

  fixedLockFile = rawLockFile // { packages = fixedPackages; };
in
{
  imports = [
    dream2nix.modules.dream2nix.nodejs-package-lock-v3
    dream2nix.modules.dream2nix.nodejs-granular-v3
  ];

  name = "exo-dashboard";
  version = "1.0.0";

  mkDerivation = {
    src = config.deps.dashboardSrc;

    buildPhase = ''
      runHook preBuild
      npm run build
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      cp -r build $out/build
      runHook postInstall
    '';
  };

  deps = { nixpkgs, ... }: {
    inherit (nixpkgs) stdenv;
    dashboardSrc = null; # Injected by parts.nix
  };

  nodejs-package-lock-v3 = {
    # Don't use packageLockFile - provide the fixed lock content directly
    packageLock = fixedLockFile;
  };
}
