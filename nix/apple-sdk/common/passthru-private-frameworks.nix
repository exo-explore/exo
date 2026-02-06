{ makeSetupHook, sdkVersion }:

self: super: {
  passthru = super.passthru or { } // {
    privateFrameworksHook = makeSetupHook {
      name = "apple-sdk-private-frameworks-hook";
    } ../setup-hooks/add-private-frameworks.sh;
  };
}
