{ stdenv
, lib
, buildPythonPackage
, fetchFromGitHub
, replaceVars
, fetchzip
  # build-system
, setuptools
  # nativeBuildInputs
, cmake
, patchelf
, zsh
, # buildInputs
  nanobind
, pybind11
, nlohmann_json
  # buildInputs (darwin)
, xcode
  # buildInputs (linux)
, openblas
  # tests
, numpy
, pytestCheckHook
, python
, runCommand
, fmt
}:
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

  macosSdk = "${xcode}/MacOSX.sdk";

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
        sdkVersion = if stdenv.isDarwin then xcode.version else "nah";
        metalVersion = if stdenv.isDarwin then 230 else "nah";
      })
    ];

    postPatch = ''
      substituteInPlace pyproject.toml \
        --replace-fail "nanobind==2.10.2" "nanobind"

      substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
        --replace-fail "g++" "$CXX"
    '';

    preFixup = ''
      patchelf --set-rpath '$ORIGIN/lib64:$ORIGIN/lib:${lib.makeLibraryPath ([ stdenv.cc.cc.lib ] ++ lib.optionals stdenv.isLinux [ openblas ])}' $out/lib/python*/site-packages/mlx/core*.so
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
      CMAKE_ARGS = toString ([
        (lib.cmakeBool "USE_SYSTEM_FMT" true)
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_JSON" "${nlohmann_json.src}")
        (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
      ] ++ lib.optionals stdenv.isDarwin [
        (lib.cmakeBool "MLX_BUILD_METAL" true)
        (lib.cmakeOptionType "filepath" "METAL_LIB"
          "${macosSdk}/System/Library/Frameworks/Metal.framework")
        (lib.cmakeOptionType "filepath" "FOUNDATION_LIB"
          "${macosSdk}/System/Library/Frameworks/Foundation.framework")
        (lib.cmakeOptionType "filepath" "QUARTZ_LIB"
          "${macosSdk}/System/Library/Frameworks/QuartzCore.framework")
        (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal_cpp}")
        (lib.cmakeOptionType "string" "CMAKE_OSX_DEPLOYMENT_TARGET" "${xcode.version}")
        # (lib.cmakeOptionType "filepath" "CMAKE_OSX_SYSROOT" "${macosSdk}")
        (lib.cmakeOptionType "filepath" "CMAKE_FRAMEWORK_PATH" "${macosSdk}/System/Library/Frameworks")
        (lib.cmakeOptionType "string" "CMAKE_C_FLAGS"
           "-Wno-error=elaborated-enum-base -Wno-error=nullability-completeness")
        (lib.cmakeOptionType "string" "CMAKE_CXX_FLAGS"
          "-Wno-error=elaborated-enum-base -Wno-error=nullability-completeness")
      ] ++ lib.optionals stdenv.isLinux [

        (lib.cmakeBool "MLX_BUILD_METAL" false)
        (lib.cmakeBool "MLX_BUILD_CUDA" true)
      ]);
    } // lib.optionalAttrs stdenv.isDarwin {
      # DEVELOPER_DIR = "${xcode}/Developer";
      SDKROOT = "${macosSdk}";
      MACOSX_DEPLOYMENT_TARGET = xcode.version;
    };

    build-system = [
      setuptools
    ];

    nativeBuildInputs = [
      cmake
      patchelf
      zsh
    ] ++ lib.optionals stdenv.isDarwin [ xcode ];

    buildInputs = [
      fmt
      gguf-tools
      nanobind
      pybind11
    ] ++ lib.optionals stdenv.isLinux [ openblas ];

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
