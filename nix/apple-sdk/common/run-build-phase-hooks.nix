{ }:

self: super: {
  buildPhase = ''
    runHook preBuild
    ${super.buildPhase or ""}
    runHook postBuild
  '';
}
