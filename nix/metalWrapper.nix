{ stdenvNoCC
, metalVersion
}:
assert stdenvNoCC.isDarwin;
stdenvNoCC.mkDerivation {
  pname = "metal-wrapper-impure";
  version = metalVersion;

  __noChroot = true;
  buildCommand = ''
    mkdir -p $out/bin && cd $out/bin

    METALLIB_PATH=''${GH_OVERRIDE_METALLIB:-$(/usr/bin/xcrun --sdk macosx -f metallib)}
    METAL_PATH=''${GH_OVERRIDE_METAL:-"$(dirname "$METALLIB_PATH")/metal"}
    echo "$METAL_PATH"
    echo "$METALLIB_PATH"

    ln -sf "$METAL_PATH" metal
    ln -sf "$METALLIB_PATH" metallib

    [[ -e $out/bin/metal ]] && [[ -e $out/bin/metallib ]] || { echo ":(" && exit 1; }
    METAL_VERSION=$(echo __METAL_VERSION__ | "$METAL_PATH" -E -x metal -P - | tail -1 | tr -d '\n')
    [[ "$METAL_VERSION" == "${metalVersion}" ]] || { echo "Metal version $METAL_VERSION is not ${metalVersion}" && exit 1; }
  '';
}
