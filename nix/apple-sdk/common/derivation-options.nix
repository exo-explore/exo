{ lib, config }:

self: super: {
  preBuild = super.preBuild or "" + ''
    platformPath=$out/Platforms/MacOSX.platform
    sdkpath=$platformPath/Developer/SDKs
  '';

  preInstall = super.preInstall or "" + ''
    platformPath=$out/Platforms/MacOSX.platform
    sdkpath=$platformPath/Developer/SDKs
  '';
}
