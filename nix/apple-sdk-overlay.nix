# Overlay that builds apple-sdk with a custom versions.json (for SDK 26.2).
# The upstream nixpkgs package reads versions.json at eval time via a relative
# path, so we patch the upstream package.nix text instead of building a patched
# source tree. That keeps the upstream implementation while avoiding foreign-
# system derivations during evaluation.
final: prev:
let
  upstreamSrc = final.path + "/pkgs/by-name/ap/apple-sdk";
  patchedPackage = builtins.toFile "apple-sdk-package.nix" (
    builtins.replaceStrings
      [
        "./metadata/versions.json"
        "./common/"
        "./setup-hooks/"
      ]
      [
        "${./apple-sdk/metadata/versions.json}"
        "${upstreamSrc}/common/"
        "${upstreamSrc}/setup-hooks/"
      ]
      (builtins.readFile (upstreamSrc + "/package.nix"))
  );
in
if prev.stdenv.hostPlatform.isDarwin then {
  apple-sdk_26 = final.callPackage patchedPackage {
    darwinSdkMajorVersion = "26";
  };
} else { }
