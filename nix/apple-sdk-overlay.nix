# Overlay that builds apple-sdk with a custom versions.json (for SDK 26.2).
# The upstream nixpkgs package reads versions.json at eval time via a relative
# path, so we can't override it through callPackage args. Instead, we copy
# the upstream source and patch the one file.
final: _prev:
let
  upstreamSrc = final.path + "/pkgs/by-name/ap/apple-sdk";
  patchedSrc = final.runCommandLocal "apple-sdk-src-patched" { } ''
    cp -r ${upstreamSrc} $out
    chmod -R u+w $out
    cp ${./apple-sdk/metadata/versions.json} $out/metadata/versions.json
  '';
in
{
  apple-sdk_26 = final.callPackage (patchedSrc + "/package.nix") {
    darwinSdkMajorVersion = "26";
  };
}
