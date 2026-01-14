{ stdenv
, metalVersion
, xcodeBaseDir ? "/Applications/Xcode.app"
}:
assert stdenv.isDarwin;
stdenv.mkDerivation {
  pname = "metal-wrapper-impure";
  version = metalVersion;

  __noChroot = true;
  buildCommand = ''
    DEVELOPER_DIR=${xcodeBaseDir}/Contents/Developer
    [[ -x "$DEVELOPER_DIR/usr/bin/xcodebuild" ]] || (echo "Missing xcodebuild at $DEVELOPER_DIR/usr/bin/xcodebuild" && exit 1)
    SDKROOT=${xcodeBaseDir}/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    [[ -d "$SDKROOT" ]] || (echo "Missing SDKROOT at $SDKROOT" && exit 1)
    export DEVELOPER_DIR SDKROOT
    mkdir -p $out/bin && cd $out/bin
    ln -s $(/usr/bin/xcrun --sdk macosx -f metal)
    ln -s $(/usr/bin/xcrun --sdk macosx -f metallib)
    [[ -f $out/bin/metal ]] && [[ -f $out/bin/metallib ]] || exit 1
  '';
}
