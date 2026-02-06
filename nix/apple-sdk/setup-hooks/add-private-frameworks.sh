function enablePrivateFrameworks() {
    export NIX_CFLAGS_COMPILE+=" -iframework $DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/PrivateFrameworks"
    export NIX_LDFLAGS+=" -F$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/PrivateFrameworks"
}

preConfigureHooks+=(enablePrivateFrameworks)
