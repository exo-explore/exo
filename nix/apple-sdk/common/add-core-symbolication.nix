{
  lib,
  fetchFromGitHub,
  stdenvNoCC,
}:

let
  CoreSymbolication = stdenvNoCC.mkDerivation (finalAttrs: {
    pname = "CoreSymbolication";
    version = "0-unstable-2018-06-17";

    src = fetchFromGitHub {
      repo = "CoreSymbolication";
      owner = "matthewbauer";
      rev = "24c87c23664b3ee05dc7a5a87d647ae476a680e4";
      hash = "sha256-PzvLq94eNhP0+rLwGMKcMzxuD6MlrNI7iT/eV0obtSE=";
    };

    patches = [
      # Add missing symbol definitions needed to build `zlog` in system_cmds.
      # https://github.com/matthewbauer/CoreSymbolication/pull/2
      ../patches/0001-Add-function-definitions-needed-to-build-zlog-in-sys.patch
      ../patches/0002-Add-CF_EXPORT-To-const-symbols.patch
    ];

    dontBuild = true;

    installPhase = ''
      mkdir -p "$out/include"
      cp *.h "$out/include"
    '';

    meta = {
      description = "Reverse engineered headers for Apple's CoreSymbolication framework";
      homepage = "https://github.com/matthewbauer/CoreSymbolication";
      license = lib.licenses.mit;
      teams = [ lib.teams.darwin ];
      platforms = lib.platforms.darwin;
    };
  });
in
self: super: {
  buildPhase = super.buildPhase or "" + ''
    mkdir -p System/Library/PrivateFrameworks/CoreSymbolication.framework/Versions/A/Headers
    ln -s Versions/Current/Headers System/Library/PrivateFrameworks/CoreSymbolication.framework/Headers
    cp '${CoreSymbolication}/include/'*.h System/Library/PrivateFrameworks/CoreSymbolication.framework/Versions/A/Headers
  '';
}
