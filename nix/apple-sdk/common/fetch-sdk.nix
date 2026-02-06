{
  lib,
  fetchurl,
  cpio,
  pbzx,
}:

{
  urls,
  version,
  hash,
}:

fetchurl {
  pname = "macOS-SDK";
  inherit version urls hash;

  recursiveHash = true;

  nativeBuildInputs = [
    cpio
    pbzx
  ];

  postFetch = ''
    renamed=$(mktemp -d)/sdk.xar
    mv "$downloadedFile" "$renamed"
    pbzx "$renamed" | cpio -idm

    src=Library/Developer/CommandLineTools/SDKs/MacOSX${lib.versions.majorMinor version}.sdk

    # Remove unwanted binaries, man pages, and folders from the SDK.
    rm -rf $src/usr/bin $src/usr/share $src/System/Library/Perl

    mkdir -p "$out"
    cp -rd $src/* "$out"
  '';
}
