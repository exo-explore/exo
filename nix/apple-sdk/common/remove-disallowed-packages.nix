let
  disallowedPackages = builtins.fromJSON (builtins.readFile ../metadata/disallowed-packages.json);
in

{
  lib,
  jq,
  stdenv,
}:

self: super: {
  # Remove headers and stubs for packages that are available in nixpkgs.
  buildPhase = super.buildPhase or "" + ''
    ${lib.concatMapStringsSep "\n" (
      pkg:
      lib.concatLines (
        [ ''echo "Removing headers and libraries from ${pkg.package}"'' ]
        ++ (map (header: "rm -rf -- usr/include/${header}") pkg.headers or [ ])
        ++ (map (framework: "rm -rf -- System/Library/Frameworks/${framework}") pkg.frameworks or [ ])
        ++ (map (library: "rm -rf -- usr/lib/${library}") pkg.libraries or [ ])
      )
    ) disallowedPackages}
  '';
}
