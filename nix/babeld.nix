{ stdenv
, lib
, fetchgit
}:
stdenv.mkDerivation {
  pname = "babeld";
  version = "1.13.2-rc";

  # TODO: pin to specific version/revision, or better yet, use a patch file
  src = fetchgit {
    url = "https://github.com/AndreiCravtov/babeld.git";
    fetchSubmodules = true;
    sha256 = "sha256-Z4fZNh9ZdWRaTrUxgbXZnDCqvG6m4F/CND3ApyavbLw=";
  };

  outputs = [
    "out"
    "man"
  ];

  makeFlags = [
    "PREFIX=${placeholder "out"}"
    "ETCDIR=${placeholder "out"}/etc"
  ];
}

