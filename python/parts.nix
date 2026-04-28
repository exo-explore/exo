{ inputs, ... }:
let
  # Load workspace from uv.lock
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../.;
  };

  mkPythonSet = { pkgs, lib, self', members }:
    let
      inherit (pkgs.stdenv.hostPlatform) isLinux isDarwin isx86_64;
      inherit (pkgs.config) cudaSupport;
      inherit (pkgs) cudaPackages;
      libmlx_source =
        if (builtins.elem "mlx-cuda13" members.exo or [ ]) then "mlx-cuda-13"
        else if (builtins.elem "mlx-cuda12" members.exo or [ ]) then "mlx-cuda-12"
        else "mlx-cpu";
      python = pkgs.python313;
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
      buildSystemsOverlay = final: prev:
        lib.optionalAttrs isDarwin
          {
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
              in
              {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.cmake self'.packages.metal-toolchain ];
                # TODO: non-sdk_26 support
                buildInputs = (old.buildInputs or [ ])
                ++ [ gguf-tools pkgs.fmt pkgs.nlohmann_json pkgs.apple-sdk_26 ];
                patches = [
                  (pkgs.replaceVars ../nix/darwin-build-fixes.patch {
                    sdkVersion = pkgs.apple-sdk_26.version;
                    inherit (self'.packages.metal-toolchain) metalVersion;
                  })
                ];
                postPatch = ''
                  substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
                    --replace-fail "g++" "${lib.getExe' pkgs.stdenv.cc "c++"}"
                '';

                DEV_RELEASE = 1;
                CMAKE_ARGS = toString ([
                  (lib.cmakeBool "USE_SYSTEM_FMT" true)
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_JSON" "${pkgs.nlohmann_json.src}")
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_NANOBIND" "${nanobind}")
                  (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
                  (lib.cmakeBool "MLX_BUILD_CPU" true)
                  (lib.cmakeBool "MLX_BUILD_METAL" true)
                  (lib.cmakeOptionType "string" "CMAKE_INSTALL_LIBDIR" "lib")
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal_cpp}")
                  (lib.cmakeOptionType "string" "CMAKE_OSX_DEPLOYMENT_TARGET" "${pkgs.apple-sdk_26.version}")
                  (lib.cmakeOptionType "filepath" "CMAKE_OSX_SYSROOT" "${pkgs.apple-sdk_26.passthru.sdkroot}")
                ] ++ lib.optionals (isDarwin && isx86_64) [
                  (lib.cmakeBool "MLX_ENABLE_X64_MAC" true)
                ]);
                SDKROOT = pkgs.apple-sdk_26.passthru.sdkroot;
                MACOSX_DEPLOYMENT_TARGET = pkgs.apple-sdk_26.version;
              });
          } // lib.optionalAttrs isLinux {
          mlx = prev.mlx.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ lib.optionals cudaSupport [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ lib.optionals cudaSupport cudaLibs;
            postInstall = ''
              cp -r "${final.${libmlx_source}}/${final.python.sitePackages}/mlx" "$out/${final.python.sitePackages}/mlx/"
            '';
            autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
          });
        } // lib.optionalAttrs cudaSupport {
          "${libmlx_source}" = prev."${libmlx_source}".overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ cudaLibs;
            autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
          });
          nvidia-cufile = prev.nvidia-cufile.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ [ pkgs.rdma-core ];
          });
          nvidia-cusolver = prev.nvidia-cusolver.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ cudaLibs;
          });
          nvidia-nvshmem-cu13 = prev.nvidia-nvshmem-cu13.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ [ pkgs.rdma-core pkgs.pmix pkgs.libfabric pkgs.ucx pkgs.openmpi ];
          });
          nvidia-cusparse = prev.nvidia-cusparse.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ cudaLibs;
          });
          torch = prev.torch.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ cudaLibs;
            autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
          });
          torchaudio = prev.torchaudio.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = old.buildInputs ++ [ cudaPackages.cuda_cudart ];
            preFixup = "addAutoPatchelfSearchPath '${final.torch}'";
          });
          torchvision = prev.torchvision.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.autoAddDriverRunpath ];
            preFixup = "addAutoPatchelfSearchPath '${final.torch}'";
          });

          torch-c-dlpack-ext = prev.torch-c-dlpack-ext.overrideAttrs (old: {
            buildInputs = old.buildInputs ++ cudaLibs;
            autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            preFixup = "addAutoPatchelfSearchPath '${final.torch}'";
          });
          # Currently treating vllm as a cuda dep. it obviously exists as a non cuda dep
          vllm = prev.vllm.overrideAttrs (old:
            let
              cuda_cccl_compat = pkgs.runCommand "cuda-cccl-compat" { } ''
                mkdir -p $out/include
                ln -s ${cudaPackages.cuda_cccl}/include $out/include/cccl
              '';

              cudaRoot = pkgs.symlinkJoin {
                name = "cuda-merged-exo";
                paths = builtins.concatMap (p: [ (lib.getBin p) (lib.getLib p) (lib.getDev p) ]) (cudaLibs ++ [ cudaPackages.cuda_nvcc cuda_cccl_compat ]);
              };

              cutlass = pkgs.fetchFromGitHub {
                name = "cutlass-source";
                owner = "NVIDIA";
                repo = "cutlass";
                tag = "v4.2.1";
                hash = "sha256-iP560D5Vwuj6wX1otJhwbvqe/X4mYVeKTpK533Wr5gY=";
              };
              triton-kernels = pkgs.fetchFromGitHub {
                owner = "triton-lang";
                repo = "triton";
                tag = "v3.6.0";
                hash = "sha256-JFSpQn+WsNnh7CAPlcpOcUp0nyKXNbJEANdXqmkt4Tc=";
              };

              cutlass-flashmla = pkgs.fetchFromGitHub {
                owner = "NVIDIA";
                repo = "cutlass";
                rev = "147f5673d0c1c3dcf66f78d677fd647e4a020219";
                hash = "sha256-dHQto08IwTDOIuFUp9jwm1MWkFi8v2YJ/UESrLuG71g=";
              };

              flashmla = pkgs.stdenv.mkDerivation {
                pname = "flashmla";
                version = "1.0.0";

                src = pkgs.fetchFromGitHub {
                  name = "FlashMLA-source";
                  owner = "vllm-project";
                  repo = "FlashMLA";
                  rev = "c2afa9cb93e674d5a9120a170a6da57b89267208";
                  hash = "sha256-pKlwxV6G9iHag/jbu3bAyvYvnu5TbrQwUMFV0AlGC3s=";
                };

                dontConfigure = true;

                buildPhase = ''
                  rm -rf csrc/cutlass
                  ln -sf ${cutlass-flashmla} csrc/cutlass
                '';

                installPhase = ''
                  cp -rva . $out
                '';
              };
              qutlass = pkgs.fetchFromGitHub {
                name = "qutlass-source";
                owner = "IST-DASLab";
                repo = "qutlass";
                rev = "830d2c4537c7396e14a02a46fbddd18b5d107c65";
                hash = "sha256-aG4qd0vlwP+8gudfvHwhtXCFmBOJKQQTvcwahpEqC84=";
              };
              vllm-flash-attn = pkgs.stdenv.mkDerivation {
                pname = "vllm-flash-attn";
                version = "2.7.2.post1";

                src = pkgs.fetchFromGitHub {
                  name = "flash-attention-source";
                  owner = "vllm-project";
                  repo = "flash-attention";
                  rev = "188be16520ceefdc625fdf71365585d2ee348fe2";
                  hash = "sha256-Osec+/IF3+UDtbIhDMBXzUeWJ7hDJNb5FpaVaziPSgM=";
                };

                patches = [
                  (pkgs.fetchpatch {
                    url = "https://github.com/Dao-AILab/flash-attention/commit/dad67c88d4b6122c69d0bed1cebded0cded71cea.patch";
                    hash = "sha256-JSgXWItOp5KRpFbTQj/cZk+Tqez+4mEz5kmH5EUeQN4=";
                  })
                  (pkgs.fetchpatch {
                    url = "https://github.com/Dao-AILab/flash-attention/commit/e26dd28e487117ee3e6bc4908682f41f31e6f83a.patch";
                    hash = "sha256-NkCEowXSi+tiWu74Qt+VPKKavx0H9JeteovSJKToK9A=";
                  })
                ];

                dontConfigure = true;

                buildPhase = ''
                  rm -rf csrc/cutlass
                  ln -sf ${cutlass} csrc/cutlass
                '';

                installPhase = ''
                  cp -rva . $out
                '';
              };
            in
            {
              patches = (old.patches or [ ]) ++ [ ../nix/vllm-setuppy-cmake.patch ];
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                pkgs.cmake
                pkgs.ninja
                pkgs.autoAddDriverRunpath
              ] ++ lib.optionals cudaSupport [
                cudaPackages.cuda_nvcc
              ];
              # TODO: vllm rocm/cpu
              VLLM_TARGET_DEVICE = "empty";
              preConfigure = ''
                export MAX_JOBS="$NIX_BUILD_CORES"
              '';

              # TODO: vllm non cuda13 support, more arch's, etc.
            } // lib.optionalAttrs cudaSupport {
              buildInputs = cudaLibs ++ [ cudaRoot ];

              VLLM_CUDA_VERSION = cudaPackages.cudaMajorMinorVersion;
              CUDA_HOME = "${cudaRoot}";
              CUDAToolkit_ROOT = "${cudaRoot}";
              CUDACXX = "${cudaRoot}/bin/nvcc";
              VLLM_CUTLASS_SRC_DIR = "${lib.getDev cutlass}";
              VLLM_TARGET_DEVICE = "cuda";
              TORCH_CUDA_ARCH_LIST = "12.0;12.1";
              TRITON_KERNELS_SRC_DIR = "${lib.getDev triton-kernels}/python/triton_kernels/triton_kernels";
              FLASH_MLA_SRC_DIR = "${lib.getDev flashmla}";
              QUTLASS_SRC_DIR = "${lib.getDev qutlass}";
              VLLM_FLASH_ATTN_SRC_DIR = "${lib.getDev vllm-flash-attn}";
              CAFFE2_USE_CUDNN = "ON";
              CAFFE2_USE_CUFILE = "ON";
              CUTLASS_ENABLE_CUBLAS = "ON";
              CUTLASS_NVCC_ARCHS_ENABLED = "12.0;12.1";

              cmakeFlags = [
                (lib.cmakeBool "CMAKE_SKIP_INSTALL_RPATH" true)
                (lib.cmakeBool "CMAKE_BUILD_WITH_INSTALL_RPATH" true)
                (lib.cmakeFeature "CUDA_HOME" "${cudaRoot}")
                (lib.cmakeFeature "CUDAToolkit_ROOT" "${cudaRoot}")
                (lib.cmakeFeature "CMAKE_CUDA_COMPILER" "${cudaRoot}/bin/nvcc")
                (lib.cmakeFeature "CMAKE_PREFIX_PATH" "${cudaRoot}")
                (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_CUTLASS" "${lib.getDev cutlass}")
                (lib.cmakeFeature "FLASH_MLA_SRC_DIR" "${lib.getDev flashmla}")
                (lib.cmakeFeature "VLLM_FLASH_ATTN_SRC_DIR" "${lib.getDev vllm-flash-attn}")
                (lib.cmakeFeature "QUTLASS_SRC_DIR" "${lib.getDev qutlass}")
                (lib.cmakeFeature "TORCH_CUDA_ARCH_LIST" "12.0;12.1")
                (lib.cmakeFeature "CUTLASS_NVCC_ARCHS_ENABLED" "${cudaPackages.flags.cmakeCudaArchitecturesString}")
                (lib.cmakeFeature "CUDA_TOOLKIT_ROOT_DIR" "${cudaRoot}")
                (lib.cmakeFeature "CAFFE2_USE_CUDNN" "ON")
                (lib.cmakeFeature "CAFFE2_USE_CUFILE" "ON")
                (lib.cmakeFeature "CUTLASS_ENABLE_CUBLAS" "ON")
              ];
            });

        } // lib.optionalAttrs (cudaSupport && isx86_64) {
          numba = prev.numba.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb ];
          });
        };
      pyprojectOverlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
        dependencies = members;
      };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        # Use environment variable pointing to editable root directory
        root = "$REPO_ROOT";
        members = [ "exo" "exo-bench" ];
      };
      pythonSet = (pkgs.callPackage inputs.pyproject-nix.build.packages {
        inherit python;
      }).overrideScope (
        lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.default
          pyprojectOverlay
          exoOverlay
          buildSystemsOverlay
        ]
      );
      # mlx and mlx-cuda ship clashing cmake files - we dont need them at runtime anyway
      venv = name: (pythonSet.mkVirtualEnv "${name}-venv" members).overrideAttrs (_: { venvSkip = [ "lib/python${python.pythonVersion}/site-packages/mlx/share/cmake/*" "lib/python${python.pythonVersion}/site-packages/build_backend.py" ]; });
      mkApp = text: name: pkgs.writeShellApplication {
        inherit name;
        text = "exec " + lib.optionalString cudaSupport "nixglhost " + text;
        runtimeEnv = {
          EXO_DASHBOARD_DIR = self'.packages.dashboard;
          EXO_RESOURCES_DIR = inputs.self + /resources;
        };
        runtimeInputs = [
          (venv name)
          pkgs.nix-gl-host
        ]
        ++ lib.optionals isDarwin [ pkgs.macmon ];
        passthru = {
          venv = venv name;
          evenv = ((pythonSet.overrideScope editableOverlay).mkVirtualEnv "${name}-evenv" (members // { exo = (members.exo or [ ]) ++ [ "dev" ]; })).overrideAttrs (_: { venvSkip = [ "lib/python${python.pythonVersion}/site-packages/mlx/share/cmake/*" "lib/python${python.pythonVersion}/site-packages/build_backend.py" ]; });
        };
      };
    in
    {
      inherit venv;
      mkPythonScript = path: mkApp ''python ${path} "$@"'';
      mkExo = mkApp ''exo "$@"'';
    };
in
{
  perSystem =
    { self', pkgs, unfreePkgs, lib, ... }:
    let
      inherit (pkgs.stdenv.hostPlatform) isLinux;
      inherit (mkPythonSet { inherit self' pkgs lib; members = { exo = [ "mlx-cpu" "vllm-none" ]; }; }) mkExo;

      # Virtual environment with dev dependencies for testing
      testVenv = (mkPythonSet {
        inherit self' pkgs lib; members = {
        exo = [ "dev" "mlx-cpu" "vllm-none" ]; # Include pytest, pytest-asyncio, pytest-env
      };
      }).venv "exo-test";

      mkBenchScript = (mkPythonSet {
        inherit self' pkgs lib; members = {
        exo = [ "mlx-cpu" "vllm-none" ];
        exo-bench = [ ]; # Include pytest, pytest-asyncio, pytest-env
      };
      }).mkPythonScript;

      mkSimplePythonScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ pkgs.python313 ];
        text = ''exec python ${path} "$@"'';
      };
      cuda12Set = mkPythonSet { inherit self' lib; inherit (unfreePkgs.pkgsCuda.cudaPackages_12) pkgs; members = { exo = [ "mlx-cuda12" "vllm-none" ]; }; };
      cuda13Set = mkPythonSet { inherit self' lib; inherit (unfreePkgs.pkgsCuda.cudaPackages_13) pkgs; members = { exo = [ "mlx-cpu" "vllm-cuda13" ]; }; };
    in
    {
      packages = {
        exo = mkExo "exo";
        # for running tests in ci
        exo-test-env = testVenv;
        exo-bench = mkBenchScript "exo-bench" (inputs.self + /bench/exo_bench.py);
        exo-eval = mkBenchScript "exo-eval" (inputs.self + /bench/exo_eval.py);
        exo-eval-tool-calls = mkBenchScript "exo-eval-tool-calls" (inputs.self + /bench/eval_tool_calls.py);
        # used by ./tests/run_exo_on.sh
        exo-get-all-models-on-cluster = mkSimplePythonScript "exo-get-all-models-on-cluster" (inputs.self + /tests/get_all_models_on_cluster.py);
      } // lib.optionalAttrs isLinux {
        exo-cuda-12 = cuda12Set.mkExo "exo-cuda-12";
        exo-cuda-13 = cuda13Set.mkExo "exo-cuda-13";
      };

      checks = {
        lint = pkgs.runCommand "ruff-lint" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}
          touch $out
        '';

        typecheck = pkgs.runCommand "typecheck" { nativeBuildInputs = [ testVenv ]; } ''
          cd ${inputs.self}
          basedpyright
          touch $out
        '';
      };
    };
}
