{ inputs, ... }:
let
  mkPythonSet = { self', pkgs, lib, apple-sdk, editable ? false }:
    let
      inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux isx86_64;
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
      exoOverlay = final: prev:
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
              ++ lib.optionals isDarwin [ apple-sdk ]
              ++ lib.optionals cudaSupport (cudaLibs ++ [ cudaPackages.cudnn ]);
              patches = (old.patches or [ ])
              ++ lib.optionals cudaSupport [ ../nix/mlx_patch_fmod.patch ]
              ++ lib.optionals isDarwin [
                (pkgs.replaceVars ../nix/darwin-build-fixes.patch {
                  sdkVersion = apple-sdk.version;
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
                (lib.cmakeOptionType "string" "CMAKE_OSX_DEPLOYMENT_TARGET" "${apple-sdk.version}")
                (lib.cmakeOptionType "filepath" "CMAKE_OSX_SYSROOT" "${apple-sdk.passthru.sdkroot}")
              ] ++ lib.optionals (isDarwin && isx86_64) [
                (lib.cmakeBool "MLX_ENABLE_X64_MAC" true)
              ]);
            } // lib.optionalAttrs isDarwin {
              SDKROOT = apple-sdk.passthru.sdkroot;
              MACOSX_DEPLOYMENT_TARGET = apple-sdk.version;
            });
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
          torch = prev.torch.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = (old.buildInputs or [ ]) ++ lib.optionals cudaSupport cudaLibs;
            autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ lib.optionals cudaSupport [ "libcuda.so.1" ];
          });
          torchaudio = prev.torchaudio.overrideAttrs (old:
            {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
              buildInputs = (old.buildInputs or [ ]) ++ lib.optionals cudaSupport [
                cudaPackages.cuda_cudart
              ];
              preFixup = (old.preFixup or "") + ''
                addAutoPatchelfSearchPath "${final.torch}"
              '';
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ lib.optionals cudaSupport [ "libcuda.so.1" ];
            });
          torchvision = prev.torchvision.overrideAttrs (old:
            {
              nativebuildInputs = (old.nativebuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
              preFixup = (old.preFixup or "") + ''
                addAutoPatchelfSearchPath "${final.torch}"
              '';
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ lib.optionals cudaSupport [ "libcuda.so.1" ];
            });
          torch-c-dlpack-ext = prev.torch-c-dlpack-ext.overrideAttrs (old:
            {
              nativebuildInputs = (old.nativebuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
              preFixup = (old.preFixup or "") + ''
                addAutoPatchelfSearchPath "${final.torch}"
	      '';
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ lib.optionals cudaSupport [ "libcuda.so.1" ];
            });
          xgrammar = prev.xgrammar.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.cmake pkgs.autoPatchelfHook ];
          });
          # Currently treating vllm as a cuda dep. it obviously exists as a non cuda dep
          vllm = prev.vllm.overrideAttrs (old:
            let
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
              patches = (old.patches or [ ]) ++ [ ../nix/vllm_uv2nix_cmake.patch ];
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                pkgs.cmake
                pkgs.ninja
                pkgs.autoAddDriverRunpath
              ] ++ lib.optionals cudaSupport [
                cudaPackages.cuda_nvcc
              ];
              # TODO: vllm rocm/cpu
              VLLM_TARGET_DEVICE = "empty";
              # TODO: vllm non cuda13 support, more arch's, etc.
            } // lib.optionalAttrs cudaSupport {
              buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;

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

              UV2NIX_CMAKE_FLAGS_JSON = builtins.toJSON [
                "-DCUDAToolkit_ROOT=${cudaRoot}"
                "-DCMAKE_CUDA_COMPILER=${cudaRoot}/bin/nvcc"
                "-DCMAKE_PREFIX_PATH=${cudaRoot}"
                "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=${lib.getDev cutlass}"
                "-DFLASH_MLA_SRC_DIR=${lib.getDev flashmla}"
                "-DVLLM_FLASH_ATTN_SRC_DIR=${lib.getDev vllm-flash-attn}"
                "-DQUTLASS_SRC_DIR=${lib.getDev qutlass}"
                "-DTORCH_CUDA_ARCH_LIST=12.0;12.1"
                "-DCUTLASS_NVCC_ARCHS_ENABLED=${cudaPackages.flags.cmakeCudaArchitecturesString}"
                "-DCAFFE2_USE_CUDNN=ON"
                "-DCAFFE2_USE_CUFILE=ON"
                "-DCUTLASS_ENABLE_CUBLAS=ON"
              ];
            });
        } // lib.optionalAttrs cudaSupport {
          nvidia-cufile = prev.nvidia-cufile.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.rdma-core ];
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pkgs.util-linux ];
          });
          nvidia-cusolver = prev.nvidia-cusolver.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = (old.buildInputs or [ ]) ++ (with cudaPackages; [ libnvjitlink libcublas libcusparse ]);
          });
          nvidia-cusparse = prev.nvidia-cusparse.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = (old.buildInputs or [ ]) ++ [ cudaPackages.libnvjitlink ];
          });
          nvidia-nvshmem-cu13 = prev.nvidia-nvshmem-cu13.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.rdma-core pkgs.pmix pkgs.libfabric pkgs.ucx pkgs.openmpi ];
          });
          nvidia-cutlass-dsl-libs-base = prev.nvidia-cutlass-dsl-libs-base.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
            autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
          });
        } // lib.optionalAttrs (cudaSupport && isx86_64) {
          numba = prev.numba.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb ];
          });
          intel-openmp = prev.intel-openmp.overrideAttrs (_old: {
            postFixup = ''
              rm -f $out/lib/libarcher.so
              rm -f $out/lib/libomptarget.so
              rm -f $out/lib/libomptarget.rtl.*.so*
              rm -f $out/lib/libomptarget.sycl.wrap.so
            '';
          });
        };

      # Load workspace from uv.lock
      workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = ../.;
      };

      # Create overlay from workspace
      # Use wheels from PyPI for most packages; we override mlx with our pure Nix Metal build
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
    in
    (pkgs.callPackage inputs.pyproject-nix.build.packages {
      python = pkgs.python313;
    }).overrideScope (
      lib.composeManyExtensions ([
        inputs.pyproject-build-systems.overlays.default
        overlay
        exoOverlay
      ] ++ lib.optionals editable [
        (workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; members = [ "exo" "bench" ]; })
      ])
    );

  mkExo = args@{ self', pkgs, lib, ... }:
    let
      venv = ((mkPythonSet args).mkVirtualEnv "exo-env" {
        exo = lib.optionals pkgs.config.cudaSupport [ "cuda" ];
      }).overrideAttrs {
        venvSkip = [ "lib/python3.13/site-packages/build_backend.py" ];
      };
    in
    pkgs.runCommand "exo"
      {
        nativeBuildInputs = [ pkgs.makeWrapper ];
      }
      ''
        mkdir -p $out/bin

        # Create wrapper script
        makeWrapper ${venv}/bin/exo $out/bin/exo \
          --set EXO_DASHBOARD_DIR ${self'.packages.dashboard} \
          --set EXO_RESOURCES_DIR ${inputs.self + /resources} \
          ${lib.optionalString pkgs.stdenv.hostPlatform.isDarwin "--prefix PATH : ${pkgs.macmon}/bin"}
      '';
in
{
  perSystem =
    { self', pkgs, cudaPkgs, lib, ... }:
    let
      inherit (pkgs.stdenv.hostPlatform) isDarwin;
      pythonSet = mkPythonSet { inherit self' pkgs lib; apple-sdk = pkgs.apple-sdk_26; };
      # taking cudaPkgs.cudaPackages_13.pkgs creates a new nixpkgs that defaults to cuda 13
      cudaPythonSet = mkPythonSet { inherit self' lib; inherit (cudaPkgs.cudaPackages_13) pkgs; apple-sdk = pkgs.apple-sdk_26; };

      editablePythonSet = mkPythonSet { inherit self' lib pkgs; apple-sdk = pkgs.apple-sdk_26; editable = true; };
      evenv = (editablePythonSet.mkVirtualEnv "exo-dev-env"
        {
          exo = [ "dev" ];
          exo-pyo3-bindings = [ ];
          exo-bench = [ ];
        }).overrideAttrs {
        venvSkip = [ "lib/python3.13/site-packages/build_backend.py" ];
      };
      exoCudaVenv = (cudaPythonSet.mkVirtualEnv "exo-env" {
        exo = [ "cuda" ];
        exo-pyo3-bindings = [ ];
      }).overrideAttrs {
        venvSkip = [ "lib/python3.13/site-packages/build_backend.py" ];
      };

      # Virtual environment with dev dependencies for testing
      testVenv = (pythonSet.mkVirtualEnv "exo-test-env"
        {
          exo = [ "dev" ]; # Include pytest, pytest-asyncio, pytest-env
          exo-pyo3-bindings = [ ];
        }
      ).overrideAttrs {
        # venvIgnoreCollisions = venvCollisionPaths;
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

      exoCudaPackage = cudaPkgs.runCommand "exo"
        {
          nativeBuildInputs = [ cudaPkgs.makeWrapper ];
        }
        ''
          mkdir -p $out/bin

          # Create wrapper script
          makeWrapper ${exoCudaVenv}/bin/exo $out/bin/exo \
            --set EXO_DASHBOARD_DIR ${self'.packages.dashboard} \
            --set EXO_RESOURCES_DIR ${inputs.self + /resources} \
            --prefix LD_LIBRARY_PATH : /run/opengl-driver/lib:${lib.getLib pkgs.util-linux}/lib:${lib.getLib pkgs.systemd}/lib:${lib.getLib pkgs.numactl}/lib:${lib.getLib pkgs.stdenv.cc.cc.lib}/lib \
            ${lib.optionalString isDarwin "--prefix PATH : ${pkgs.macmon}/bin"}
        '';
    in
    {
      packages = {
        exo = mkExo { inherit self' lib pkgs; apple-sdk = pkgs.apple-sdk_26; };
        exo-cuda = exoCudaPackage;
        exo-bench = mkBenchScript "exo-bench" (inputs.self + /bench/exo_bench.py);
        exo-eval = mkBenchScript "exo-eval" (inputs.self + /bench/exo_eval.py);
        exo-eval-tool-calls = mkBenchScript "exo-eval-tool-calls" (inputs.self + /bench/eval_tool_calls.py);
        exo-get-all-models-on-cluster = mkSimplePythonScript "exo-get-all-models-on-cluster" (inputs.self + /tests/get_all_models_on_cluster.py);
        editable-venv = evenv;
      } // lib.optionalAttrs isDarwin {
        # Test environment for running pytest outside of Nix sandbox (needs GPU access)
        exo-test-env = testVenv;
        exo-osx14 = mkExo { inherit self' lib pkgs; apple-sdk = pkgs.apple-sdk_14; };
      };

      checks = {
        # Ruff linting (works on all platforms)
        lint = pkgs.runCommand "ruff-lint" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}
          touch $out
        '';

        # Hermetic basedpyright type checking
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
            basedpyright --pythonpath ${testVenv}/bin/python
            touch $out
          '';
      };
    };
}
