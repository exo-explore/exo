{ stdenv
, lib
, fetchFromGitHub
, replaceVars
, fetchzip
, cmake
, nlohmann_json
, apple-sdk_26
, metal-toolchain
, runCommand
, fmt
, python313Packages
, uvLockMlxVersion
}:

assert stdenv.isDarwin;

let
  python = python313Packages.python;

  # Static dependencies included directly during compilation
  gguf-tools = fetchFromGitHub {
    owner = "antirez";
    repo = "gguf-tools";
    rev = "8fa6eb65236618e28fd7710a0fba565f7faa1848";
    hash = "sha256-15FvyPOFqTOr5vdWQoPnZz+mYH919++EtghjozDlnSA=";
  };

  metal_cpp = fetchzip {
    url = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip";
    hash = "sha256-7n2eI2lw/S+Us6l7YPAATKwcIbRRpaQ8VmES7S8ZjY8=";
  };

  nanobind = fetchFromGitHub {
    owner = "wjakob";
    repo = "nanobind";
    rev = "v2.10.2";
    hash = "sha256-io44YhN+VpfHFWyvvLWSanRgbzA0whK8WlDNRi3hahU=";
    fetchSubmodules = true;
  };

  mlx = stdenv.mkDerivation rec {
    pname = "mlx";
    version = let v = "0.30.7.dev20260224+e862b122"; in
      assert v == uvLockMlxVersion || throw "MLX version mismatch: nix/mlx.nix has ${v} but uv.lock has ${uvLockMlxVersion}. Update both the version and hash in nix/mlx.nix.";
      v;
    pyproject = true;

    src = fetchFromGitHub {
      owner = "rltakashige";
      repo = "mlx-jaccl-fix-small-recv";
      rev = "e862b1223a2310d4cc8df1135aed42f5246bc50a";
      hash = "sha256-GosFIWxIB48Egb1MqJrR3xhsUsQeWdRk5rV93USY6wQ=";
    };

    patches = [
      (replaceVars ./darwin-build-fixes.patch {
        sdkVersion = apple-sdk_26.version;
        metalVersion = metal-toolchain.metalVersion;
      })
    ];

    postPatch = ''
      substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
        --replace-fail "g++" "$CXX"
    '';

    dontUseCmakeConfigure = true;

    enableParallelBuilding = true;

    # Allows multiple cores to be used in Python builds.
    postUnpack = ''
      export MAKEFLAGS+="''${enableParallelBuilding:+-j$NIX_BUILD_CORES}"
    '';

    # Updates the wrong fetcher rev attribute
    passthru.skipBulkUpdate = true;

    env = {
      DEV_RELEASE = 1;
      CMAKE_ARGS = toString [
        (lib.cmakeBool "USE_SYSTEM_FMT" true)
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_JSON" "${nlohmann_json.src}")
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_NANOBIND" "${nanobind}")
        (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
        (lib.cmakeBool "MLX_BUILD_CPU" true)
        (lib.cmakeBool "MLX_BUILD_METAL" true)
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal_cpp}")
        (lib.cmakeOptionType "string" "CMAKE_OSX_DEPLOYMENT_TARGET" "${apple-sdk_26.version}")
        (lib.cmakeOptionType "filepath" "CMAKE_OSX_SYSROOT" "${apple-sdk_26.passthru.sdkroot}")
      ];
      SDKROOT = apple-sdk_26.passthru.sdkroot;
      MACOSX_DEPLOYMENT_TARGET = apple-sdk_26.version;
    };

    build-system = [
      python313Packages.setuptools
    ];

    nativeBuildInputs = [
      cmake
      metal-toolchain
      python313Packages.pypaBuildHook
      python313Packages.pypaInstallHook
      python313Packages.setuptools
      python313Packages.typing-extensions
      python313Packages.wheel
      python313Packages.cmake
      python313Packages.ninja
    ];

    buildInputs = [
      fmt
      gguf-tools
      python313Packages.nanobind
      python313Packages.pybind11
      apple-sdk_26
    ];

    # Tests require Metal GPU access which isn't available in the Nix sandbox.
    # To run tests, build with: nix build --option sandbox false .#mlx.passthru.tests.mlxTest
    doCheck = false;

    pythonImportsCheck = [ "mlx" ];

    passthru.tests = {
      # Runs example scripts to verify MLX works. Requires --option sandbox false
      # since Metal GPU access is needed.
      mlxTest =
        runCommand "run-mlx-examples"
          {
            buildInputs = [ mlx ];
            nativeBuildInputs = [ python ];
          }
          ''
            cp ${src}/examples/python/logistic_regression.py .
            ${python.interpreter} logistic_regression.py
            rm logistic_regression.py

            cp ${src}/examples/python/linear_regression.py .
            ${python.interpreter} linear_regression.py
            rm linear_regression.py

            touch $out
          '';
    };

    meta = {
      homepage = "https://github.com/ml-explore/mlx";
      description = "Array framework for Apple silicon";
      changelog = "https://github.com/ml-explore/mlx/releases/tag/${src.tag}";
      license = lib.licenses.mit;
      platforms = [ "aarch64-darwin" ];
    };
  };
in
mlx
