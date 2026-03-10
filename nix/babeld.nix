{ stdenv
, lib
, fetchurl
}:
stdenv.mkDerivation rec {
  pname = "babeld";
  version = "1.13.1";

  src = fetchurl {
    url = "https://www.irif.fr/~jch/software/files/${pname}-${version}.tar.gz";
    hash = "sha256-FfJNJtoMz8Bzq83vAwnygeRoTyqnESb4JlcsTIRejdk=";
  };

  outputs = [
    "out"
    "man"
  ];

  makeFlags = [
    "PREFIX=${placeholder "out"}"
    "ETCDIR=${placeholder "out"}/etc"
  ]
  ++ lib.optional stdenv.isDarwin "LDLIBS=''";
}

