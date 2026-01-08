{ stdenv
, lib
, buildPythonPackage
, fetchFromGitHub
, replaceVars
, fetchzip
, setuptools
, cmake
, nanobind
, pybind11
, nlohmann_json
, apple-sdk_26
, metal
, numpy
, pytestCheckHook
, python
, runCommand
, fmt
}:
assert stdenv.isDarwin;
let
  # static dependencies included directly during compilation
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

  mlx = buildPythonPackage rec {
    pname = "mlx";
    version = "0.30.1";
    pyproject = true;

    src = fetchFromGitHub {
      owner = "ml-explore";
      repo = "mlx";
      tag = "v${version}";
      hash = "sha256-Vt0RH+70VBwUjXSfPTsNdRS3g0ookJHhzf2kvgEtgH8=";
    };

    patches = [
      (replaceVars ./darwin-build-fixes.patch {
        sdkVersion = apple-sdk_26.version;
        metalVersion = metal.version;
      })
    ];

    postPatch = ''
      substituteInPlace pyproject.toml \
        --replace-fail "nanobind==2.10.2" "nanobind"

      substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
        --replace-fail "g++" "$CXX"
    '';

    dontUseCmakeConfigure = true;

    enableParallelBuilding = true;

    # Allows multiple cores to be used in Python builds.
    postUnpack = ''
      export MAKEFLAGS+="''${enableParallelBuilding:+-j$NIX_BUILD_CORES}"
    '';

    # updates the wrong fetcher rev attribute
    passthru.skipBulkUpdate = true;

    env = {
      DEV_RELEASE = 1;
      # NOTE The `metal` command-line utility used to build the Metal kernels is not open-source.
      # this is what the xcode wrapper is for - it patches in the system metal cli
      CMAKE_ARGS = toString [
        (lib.cmakeBool "USE_SYSTEM_FMT" true)
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_JSON" "${nlohmann_json.src}")
        (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
        (lib.cmakeBool "MLX_BUILD_METAL" true)
        (lib.cmakeOptionType "filepath" "METAL_LIB"
          "${metal}/Metal.framework")
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal_cpp}")
        (lib.cmakeOptionType "string" "CMAKE_OSX_DEPLOYMENT_TARGET" "${apple-sdk_26.version}")
        (lib.cmakeOptionType "filepath" "CMAKE_OSX_SYSROOT" "${apple-sdk_26.passthru.sdkroot}")
      ];
      SDKROOT = apple-sdk_26.passthru.sdkroot;
      MACOSX_DEPLOYMENT_TARGET = apple-sdk_26.version;
    };

    build-system = [
      setuptools
    ];

    nativeBuildInputs = [
      cmake
      metal
    ];

    buildInputs = [
      fmt
      gguf-tools
      nanobind
      pybind11
      apple-sdk_26
    ];

    pythonImportsCheck = [ "mlx" ];

    # Run the mlx Python test suite.
    nativeCheckInputs = [
      numpy
      pytestCheckHook
    ];

    enabledTestPaths = [
      "python/tests/"
    ];

    # Additional testing by executing the example Python scripts supplied with mlx
    # using the version of the library we've built.
    passthru.tests = {
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
      platforms = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
    };
  };
in
mlx
