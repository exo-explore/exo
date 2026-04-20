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
    sha256 = "sha256-/qsoMSRhtwa/2hvACtFwbl+563o+TKxWMS684D+g8mk=";
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

