{ inputs, ... }:
{
  perSystem =
    { self', pkgs, lib, ... }:
    let
      inherit (pkgs.stdenv.hostPlatform) isDarwin isx86_64;
      inherit (pkgs.config) cudaSupport;
      inherit (pkgs) cudaPackages;
      cudaLibs = with cudaPackages; [
        cuda_cudart
        cuda_cccl
        cuda_cupti
        cuda_nvrtc
        cuda_nvtx
        cudnn
        libcufile
        libcublas
        libcufft
        libcurand
        libcusolver
        libcusparse
        libcusparse_lt
        libnvjitlink
        libnvshmem
        nccl
      ];
      cuda_cccl_compat = pkgs.runCommand "cuda-cccl-compat" { } ''
        mkdir -p $out/include
        ln -s ${cudaPackages.cuda_cccl}/include $out/include/cccl
      '';

      cudaRoot = pkgs.symlinkJoin {
        name = "cuda-merged-exo";
        paths = builtins.concatMap (p: [ (lib.getBin p) (lib.getLib p) (lib.getDev p) ]) (cudaLibs ++ [ cudaPackages.cuda_nvcc cuda_cccl_compat ]);
      };



      # Load workspace from uv.lock
      workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = inputs.self;
      };

      # Create overlay from workspace
      # Use wheels from PyPI for most packages; we override mlx with our pure Nix Metal build
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      # Override overlay to inject Nix-built components
      exoOverlay = final: prev: {
        # Replace workspace exo_pyo3_bindings with Nix-built wheel.
        # Preserve passthru so mkVirtualEnv can resolve dependency groups.
        # Copy .pyi stub + py.typed marker so basedpyright can find the types.
        exo-pyo3-bindings = pkgs.stdenv.mkDerivation {
          pname = "exo-pyo3-bindings";
          version = "0.1.0";
          src = self'.packages.exo_pyo3_bindings;
          # Install from pre-built wheel
          nativeBuildInputs = [ final.pyprojectWheelHook ];
          dontStrip = true;
          passthru = prev.exo-pyo3-bindings.passthru or { };
          postInstall = ''
            local siteDir=$out/${final.python.sitePackages}/exo_pyo3_bindings
            cp ${inputs.self}/rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi $siteDir/
            touch $siteDir/py.typed
          '';
        };
      };

      python = pkgs.python313;

      # Overlay to provide build systems and custom packages
      buildSystemsOverlay = _final: prev: {
        mlx = prev.mlx.overrideAttrs (old:
          let
            # Static dependencies included directly during compilation
            gguf-tools = pkgs.fetchFromGitHub {
              owner = "antirez";
              repo = "gguf-tools";
              rev = "8fa6eb65236618e28fd7710a0fba565f7faa1848";
              hash = "sha256-15FvyPOFqTOr5vdWQoPnZz+mYH919++EtghjozDlnSA=";
            };

            metal_cpp = pkgs.fetchzip {
              url = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip";
              hash = "sha256-7n2eI2lw/S+Us6l7YPAATKwcIbRRpaQ8VmES7S8ZjY8=";
            };

            nanobind = pkgs.fetchFromGitHub {
              owner = "wjakob";
              repo = "nanobind";
              rev = "v2.10.2";
              hash = "sha256-io44YhN+VpfHFWyvvLWSanRgbzA0whK8WlDNRi3hahU=";
              fetchSubmodules = true;
            };

            nvtx = pkgs.fetchFromGitHub {
              name = "nvtx3";
              owner = "NVIDIA";
              repo = "NVTX";
              rev = "v3.1.1";
              hash = "sha256-sx72N+Gskg9Vtqc3sXsWoE/2PHFI2Hq08lEaw0sll5Y=";
            };
            cudnn = pkgs.fetchFromGitHub {
              name = "cudnn_frontend";
              owner = "NVIDIA";
              repo = "cudnn-frontend";
              rev = "v1.16.0";
              hash = "sha256-+8aBl9dKd2Uz50XoOr91NRyJ4OGJtzfDNNNYGQJ9b94=";
            };
            mlx_cuda_cccl_compat = pkgs.runCommand "cuda-cccl-compat" { } ''
              mkdir -p $out/include
              exit 1
              ln -s ${cudaPackages.cuda_cccl}/include/cuda $out/include/cuda
              ln -s ${cudaPackages.cuda_cccl}/include/nv $out/include/nv
            '';
          in
          {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.cmake ] ++ lib.optionals isDarwin [ self'.packages.metal-toolchain ] ++ lib.optionals cudaSupport [
              cudaPackages.cuda_nvcc
              pkgs.autoAddDriverRunpath
              pkgs.autoPatchelfHook
            ];
            # TODO: non-sdk_26 support
            buildInputs = (old.buildInputs or [ ])
            ++ [ gguf-tools pkgs.fmt pkgs.nlohmann_json pkgs.openblas ]
            ++ lib.optionals isDarwin [ pkgs.apple-sdk_26 ]
            ++ lib.optionals cudaSupport (cudaLibs ++ [ cudaPackages.cudnn ]);
            patches = (old.patches or [ ])
            ++ lib.optionals cudaSupport [ ../nix/mlx_patch_fmod.patch ]
            ++ lib.optionals isDarwin [
              (pkgs.replaceVars ../nix/darwin-build-fixes.patch {
                sdkVersion = pkgs.apple-sdk_26.version;
                inherit (self'.packages.metal-toolchain) metalVersion;
              })
            ];
            postPatch = ''
              substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
                --replace-fail "g++" "${lib.getExe' pkgs.stdenv.cc "c++"}"
            '';
            autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ lib.optionals cudaSupport [ "libcuda.so.1" ];

            DEV_RELEASE = 1;
            CMAKE_ARGS = toString ([
              (lib.cmakeBool "USE_SYSTEM_FMT" true)
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_JSON" "${pkgs.nlohmann_json.src}")
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_NANOBIND" "${nanobind}")
              (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
              (lib.cmakeBool "MLX_BUILD_CPU" true)
              (lib.cmakeBool "MLX_BUILD_METAL" isDarwin)
              (lib.cmakeBool "MLX_BUILD_CUDA" false)
              (lib.cmakeOptionType "string" "CMAKE_INSTALL_LIBDIR" "lib")
            ] ++ lib.optionals cudaSupport [
              (lib.cmakeOptionType "filepath" "CUDAToolkit_ROOT" "${cudaRoot}")
              (lib.cmakeOptionType "string" "MLX_CUDA_ARCHITECTURES" "121")
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_CUTLASS" "${cudaPackages.cutlass}")
              # TODO: replace with cudaPackages.cudnn
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_CUDNN" "${cudnn}")
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_CCCL" "${cudaPackages.cuda_cccl}")
              # TODO: replace with cudaPackages.nvtx
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_NVTX3" "${nvtx}")
            ] ++ lib.optionals isDarwin [
              (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal_cpp}")
              (lib.cmakeOptionType "string" "CMAKE_OSX_DEPLOYMENT_TARGET" "${pkgs.apple-sdk_26.version}")
              (lib.cmakeOptionType "filepath" "CMAKE_OSX_SYSROOT" "${pkgs.apple-sdk_26.passthru.sdkroot}")
            ] ++ lib.optionals (isDarwin && isx86_64) [
              (lib.cmakeBool "MLX_ENABLE_X64_MAC" true)
            ]);
          } // lib.optionalAttrs isDarwin {
            SDKROOT = pkgs.apple-sdk_26.passthru.sdkroot;
            MACOSX_DEPLOYMENT_TARGET = pkgs.apple-sdk_26.version;
          });

      };

      # Additional overlay for Linux-specific fixes (type checking env).
      # Native wheels have shared lib dependencies we don't need at type-check time.
      pythonSet = (pkgs.callPackage inputs.pyproject-nix.build.packages {
        inherit python;
      }).overrideScope (
        lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.default
          overlay
          exoOverlay
          buildSystemsOverlay
        ]
      );

      exoVenv = pythonSet.mkVirtualEnv "exo-env" { exo = [ ]; };

      # Virtual environment with dev dependencies for testing
      testVenv = pythonSet.mkVirtualEnv "exo-test-env" {
        exo = [ "dev" ]; # Include pytest, pytest-asyncio, pytest-env
      };

      mkPythonScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ exoVenv ];
        runtimeEnv = {
          EXO_DASHBOARD_DIR = self'.packages.dashboard;
          EXO_RESOURCES_DIR = inputs.self + /resources;
        };
        text = ''exec python ${path} "$@"'';
      };

      benchVenv = pythonSet.mkVirtualEnv "exo-bench-env" {
        exo-bench = [ ];
      };

      mkBenchScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ benchVenv ];
        text = ''exec python ${path} "$@"'';
      };

      mkSimplePythonScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ pkgs.python313 ];
        text = ''exec python ${path} "$@"'';
      };

      exo = pkgs.runCommand "exo"
        {
          nativeBuildInputs = [ pkgs.makeWrapper ];
        }
        ''
          mkdir -p $out/bin

          # Create wrapper script
          makeWrapper ${exoVenv}/bin/exo $out/bin/exo \
            --set EXO_DASHBOARD_DIR ${self'.packages.dashboard} \
            --set EXO_RESOURCES_DIR ${inputs.self + /resources} \
            ${lib.optionalString isDarwin "--prefix PATH : ${pkgs.macmon}/bin"}
        '';
    in
    {
      packages = {
        inherit python exo;
        # for devShell
        exo-venv = exoVenv;
        # for running tests in ci
        exo-test-env = testVenv;
        exo-bench = mkBenchScript "exo-bench" (inputs.self + /bench/exo_bench.py);
        exo-eval = mkBenchScript "exo-eval" (inputs.self + /bench/exo_eval.py);
        exo-eval-tool-calls = mkBenchScript "exo-eval-tool-calls" (inputs.self + /bench/eval_tool_calls.py);
        # used by ./tests/run_exo_on.sh
        exo-get-all-models-on-cluster = mkSimplePythonScript "exo-get-all-models-on-cluster" (inputs.self + /tests/get_all_models_on_cluster.py);
      };

      checks = {
        lint = pkgs.runCommand "ruff-lint" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}
          touch $out
        '';

        typecheck = pkgs.runCommand "typecheck"
          {
            nativeBuildInputs = [
              testVenv
              pkgs.basedpyright
            ];
          }
          ''
            cd ${inputs.self}
            export HOME=$TMPDIR
            basedpyright --pythonpath ${testVenv}/bin/python --project ${inputs.self}/pyproject.toml
            touch $out
          '';
      };
    };
}
