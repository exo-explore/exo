{ lib, stdenvNoCC, requireFile, nix }:

let
  narFile = requireFile {
    name = "metal-toolchain-17C48.nar";
    message = ''
      The Metal Toolchain NAR must be available.

      If you have cachix configured for exo.cachix.org, this should be automatic.

      Otherwise:
        1. Install Xcode 26+ from the App Store
        2. Run: xcodebuild -downloadComponent MetalToolchain
        3. Export the toolchain:
           hdiutil attach "$(find /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain -name '*.dmg' | head -1)" -mountpoint /tmp/metal-dmg
           cp -R /tmp/metal-dmg/Metal.xctoolchain /tmp/metal-export
           hdiutil detach /tmp/metal-dmg
        4. Create NAR and add to store:
           nix nar pack /tmp/metal-export > /tmp/metal-toolchain-17C48.nar
           nix store add --mode flat /tmp/metal-toolchain-17C48.nar
    '';
    hash = "sha256-ayR5mXN4sZAddwKEG2OszGRF93k9ZFc7H0yi2xbylQw=";
  };
in
stdenvNoCC.mkDerivation {
  pname = "metal-toolchain";
  version = "17C48";

  dontUnpack = true;
  dontBuild = true;
  dontFixup = true;

  nativeBuildInputs = [ nix ];

  installPhase = ''
    runHook preInstall

    nix-store --restore $out < ${narFile}

    # Create bin directory with symlinks for PATH
    mkdir -p $out/bin
    ln -s $out/usr/bin/metal $out/bin/metal
    ln -s $out/usr/bin/metallib $out/bin/metallib

    runHook postInstall
  '';

  # Metal language version for CMake (from: echo __METAL_VERSION__ | metal -E -x metal -P -)
  passthru.metalVersion = "400";

  meta = {
    description = "Apple Metal compiler toolchain";
    platforms = [ "aarch64-darwin" ];
    license = lib.licenses.unfree;
  };
}
