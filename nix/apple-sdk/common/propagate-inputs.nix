{
  lib,
  cups,
  darwin,
  db,
  libiconv,
  ncurses,
  stdenv,
  stdenvNoCC,
  xcbuild,
}:

let
  # CUPS has too many dependencies to build as part of the Darwin bootstrap. It’s also typically taken as an explicit
  # dependency by other packages, so building only the headers (to satisfy other SDK headers) should be okay.
  cupsHeaders = darwin.bootstrapStdenv.mkDerivation {
    pname = "${lib.getName cups}-headers";
    version = lib.getVersion cups;

    inherit (cups) src;

    patches = cups.patches or [ ];

    strictDeps = true;

    dontBuild = true;

    buildInputs = [ darwin.libresolv ]; # The `configure` script requires libresolv headers.

    # CUPS’s configure script fails to find `ar` when cross-compiling.
    configureFlags = [ "ac_cv_path_AR=${stdenv.cc.targetPrefix}ar" ];

    installTargets = [ "install-headers" ];

    __structuredAttrs = true;

    meta = {
      inherit (cups.meta)
        homepage
        description
        license
        maintainers
        platforms
        ;
    };
  };
in
self: super: {
  # These packages are propagated only because other platforms include them in their libc (or otherwise by default).
  # Reducing the number of special cases required to support Darwin makes supporting it easier for package authors.
  propagatedBuildInputs =
    super.propagatedBuildInputs or [ ]
    ++ [
      libiconv
      darwin.libresolv
      darwin.libsbuf
      # Shipped with the SDK only as a library with no headers
      (lib.getLib darwin.libutil)
    ]
    # x86_64-darwin links the object files from Csu when targeting very old releases
    ++ lib.optionals stdenvNoCC.hostPlatform.isx86_64 [ darwin.Csu ];

  # The Darwin module for Swift requires certain headers to be included in the SDK (and not just be propagated).
  buildPhase = super.buildPhase or "" + ''
    for header in '${lib.getDev libiconv}/include/'* '${lib.getDev ncurses}/include/'* '${cupsHeaders}/include/'*; do
      ln -s "$header" "usr/include/$(basename "$header")"
    done
  '';

  # Exported to allow the headers to pass the requisites check in the stdenv bootstrap.
  passthru = (super.passthru or { }) // {
    cups-headers = cupsHeaders;
  };
}
