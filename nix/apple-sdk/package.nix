let
  sdkVersions = builtins.fromJSON (builtins.readFile ./metadata/versions.json);
in

{
  lib,
  stdenv,
  stdenvNoCC,
  substitute,

  # Specifies the major version used for the SDK. Uses `hostPlatform.darwinSdkVersion` by default.
  darwinSdkMajorVersion ? lib.versions.major stdenv.hostPlatform.darwinSdkVersion,

  # Enabling bootstrap disables propagation. Defaults to `false` (meaning to propagate certain packages and `xcrun`)
  # except in stage0 of the Darwin stdenv bootstrap.
  enableBootstrap ? stdenv.name == "bootstrap-stage0-stdenv-darwin",

  # Required by various phases
  callPackage,
}:

let
  sdkInfo =
    sdkVersions.${darwinSdkMajorVersion}
      or (lib.throw "Unsupported SDK major version: ${darwinSdkMajorVersion}");
  sdkVersion = sdkInfo.version;

  fetchSDK = callPackage ./common/fetch-sdk.nix { };

  phases = lib.composeManyExtensions (
    [
      (callPackage ./common/add-core-symbolication.nix { })
      (callPackage ./common/derivation-options.nix { })
      (callPackage ./common/passthru-private-frameworks.nix { inherit sdkVersion; })
      (callPackage ./common/passthru-source-release-files.nix { inherit sdkVersion; })
      (callPackage ./common/remove-disallowed-packages.nix { })
      (callPackage ./common/process-stubs.nix { })
    ]
    # Avoid infinite recursions by not propagating certain packages, so they can themselves build with the SDK.
    ++ lib.optionals (!enableBootstrap) [
      (callPackage ./common/propagate-inputs.nix { })
      (callPackage ./common/propagate-xcrun.nix { inherit sdkVersion; })
    ]
    # This has to happen last.
    ++ [
      (callPackage ./common/run-build-phase-hooks.nix { })
    ]
  );
in
stdenvNoCC.mkDerivation (
  lib.extends phases (finalAttrs: {
    pname = "apple-sdk";
    inherit (sdkInfo) version;

    src = fetchSDK sdkInfo;

    dontConfigure = true;

    strictDeps = true;

    setupHooks = [
      # `role.bash` is copied from `../build-support/setup-hooks/role.bash` due to the requirements not to reference
      # paths outside the package when it is in `by-name`.  It needs to be kept in sync, but it fortunately does not
      # change often. Once `build-support` is available as a package (or some other mechanism), it should be changed
      # to whatever that replacement is.
      ./setup-hooks/role.bash
      (substitute {
        src = ./setup-hooks/sdk-hook.sh;
        substitutions = [
          "--subst-var-by"
          "sdkVersion"
          (lib.escapeShellArgs (lib.splitVersion sdkVersion))
        ];
      })
    ];

    installPhase =
      let
        sdkName = "MacOSX${lib.versions.majorMinor sdkVersion}.sdk";
        sdkMajor = lib.versions.major sdkVersion;
      in
      ''
        runHook preInstall

        mkdir -p "$sdkpath"

        cp -rd . "$sdkpath/${sdkName}"
        ln -s "${sdkName}" "$sdkpath/MacOSX${sdkMajor}.sdk"
        ln -s "${sdkName}" "$sdkpath/MacOSX.sdk"

        # Swift adds these locations to its search paths. Avoid spurious warnings by making sure they exist.
        mkdir -p "$platformPath/Developer/Library/Frameworks"
        mkdir -p "$platformPath/Developer/Library/PrivateFrameworks"
        mkdir -p "$platformPath/Developer/usr/lib"

        runHook postInstall
      '';

    passthru = {
      sdkroot = finalAttrs.finalPackage + "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk";
    };

    __structuredAttrs = true;

    meta = {
      description = "Frameworks and libraries required for building packages on Darwin";
      homepage = "https://developer.apple.com";
      teams = [ lib.teams.darwin ];
      platforms = lib.platforms.darwin;
      badPlatforms = [ lib.systems.inspect.patterns.is32bit ];
    };
  })
)
