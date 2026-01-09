{ stdenv
, xcodeBaseDir ? "/Applications/Xcode.app"
, xcodeVersion ? "26.2"
}:
assert stdenv.hostPlatform.isDarwin;
stdenv.mkDerivation {
  pname = "xcode-wrapper-impure";
  version = xcodeVersion;

  __noChroot = true;
  buildCommand = ''
    DEVELOPER_DIR=${xcodeBaseDir}/Contents/Developer/
    SDKROOT=${xcodeBaseDir}/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    export DEVELOPER_DIR SDKROOT
    mkdir -p $out/bin && cd $out/bin
    ln -s $(/usr/bin/xcrun -f metal) $out/bin/metal
    ln -s $(/usr/bin/xcrun -f metallib) $out/bin/metallib
    cd ..
    ln -s ${xcodeBaseDir}/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk $out/MacOSX.sdk
    ln -s ${xcodeBaseDir}/Contents/Developer/ $out/Developer
  '';
}
