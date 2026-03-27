{ inputs, ... }:
{
  perSystem =
    { self', pkgs, cudaPkgs, lib, ... }:
    let
      # Load workspace from uv.lock
      workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = ../.;
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

      inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux;

      # Overlay to provide build systems and custom packages
      cudaOverlay = final: prev:
        let
          torchLibs = [
            final.nvidia-cuda-runtime
            final.nvidia-cuda-nvrtc
            final.nvidia-cuda-cupti
            final.nvidia-nvjitlink
            final.nvidia-cudnn-cu13
            final.nvidia-cusparse
            final.nvidia-cusparselt-cu13
            final.nvidia-cufile
            final.nvidia-nvshmem-cu13
            final.nvidia-nccl-cu13
            final.nvidia-cublas
            final.nvidia-cufft
            final.nvidia-curand
            final.nvidia-cusolver
          ];
          cutlass = cudaPkgs.fetchFromGitHub {
            name = "cutlass-source";
            owner = "NVIDIA";
            repo = "cutlass";
            tag = "v4.2.1";
            hash = "sha256-iP560D5Vwuj6wX1otJhwbvqe/X4mYVeKTpK533Wr5gY=";
          };
          triton-kernels = cudaPkgs.fetchFromGitHub {
            owner = "triton-lang";
            repo = "triton";
            tag = "v3.6.0";
            hash = "sha256-JFSpQn+WsNnh7CAPlcpOcUp0nyKXNbJEANdXqmkt4Tc=";
          };

          cutlass-flashmla = cudaPkgs.fetchFromGitHub {
            owner = "NVIDIA";
            repo = "cutlass";
            rev = "147f5673d0c1c3dcf66f78d677fd647e4a020219";
            hash = "sha256-dHQto08IwTDOIuFUp9jwm1MWkFi8v2YJ/UESrLuG71g=";
          };

          flashmla = cudaPkgs.stdenv.mkDerivation {
            pname = "flashmla";
            version = "1.0.0";

            src = cudaPkgs.fetchFromGitHub {
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
          qutlass = cudaPkgs.fetchFromGitHub {
            name = "qutlass-source";
            owner = "IST-DASLab";
            repo = "qutlass";
            rev = "830d2c4537c7396e14a02a46fbddd18b5d107c65";
            hash = "sha256-aG4qd0vlwP+8gudfvHwhtXCFmBOJKQQTvcwahpEqC84=";
          };
          vllm-flash-attn = cudaPkgs.stdenv.mkDerivation {
            pname = "vllm-flash-attn";
            version = "2.7.2.post1";

            src = cudaPkgs.fetchFromGitHub {
              name = "flash-attention-source";
              owner = "vllm-project";
              repo = "flash-attention";
              rev = "188be16520ceefdc625fdf71365585d2ee348fe2";
              hash = "sha256-Osec+/IF3+UDtbIhDMBXzUeWJ7hDJNb5FpaVaziPSgM=";
            };

            patches = [
              (cudaPkgs.fetchpatch {
                url = "https://github.com/Dao-AILab/flash-attention/commit/dad67c88d4b6122c69d0bed1cebded0cded71cea.patch";
                hash = "sha256-JSgXWItOp5KRpFbTQj/cZk+Tqez+4mEz5kmH5EUeQN4=";
              })
              (cudaPkgs.fetchpatch {
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

          cudaLibs = with cudaPkgs.cudaPackages_13; [
            cuda_cudart
            cuda_cccl
            cuda_cupti
            cuda_nvrtc
            cuda_nvtx
            cudnn
            libcufile
            libcublas
            libcublas
            libcufft
            libcurand
            libcusolver
            libcusparse
            libcusparse_lt
            libcufile
            libnvjitlink
            libnvshmem
            nccl
          ];

          cuda_cccl_compat = cudaPkgs.runCommand "cuda-cccl-compat" { } ''
            mkdir -p $out/include
            ln -s ${cudaPkgs.cudaPackages_13.cuda_cccl}/include $out/include/cccl
          '';
        in
        {
	  nvidia-cufile = prev.nvidia-cufile.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ 
                cudaPkgs.rdma-core
                cudaPkgs.autoAddDriverRunpath
            ];
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ cudaPkgs.util-linux ];
	  });
          nvidia-cusolver = prev.nvidia-cusolver.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ (with cudaPkgs.cudaPackages_13; [ libnvjitlink libcublas libcusparse cudaPkgs.autoAddDriverRunpath]);
	  });
          nvidia-cusparse = prev.nvidia-cusparse.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ cudaPkgs.cudaPackages_13.libnvjitlink cudaPkgs.autoAddDriverRunpath];
          });
          nvidia-nvshmem-cu13 = prev.nvidia-nvshmem-cu13.overrideAttrs (old: {
	    buildInputs = (old.buildInputs or [ ]) ++ [ cudaPkgs.rdma-core cudaPkgs.pmix cudaPkgs.libfabric cudaPkgs.ucx cudaPkgs.openmpi cudaPkgs.autoAddDriverRunpath];
	  });
 
          torch = prev.torch.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs ++ [ cudaPkgs.autoAddDriverRunpath ];
            autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
          });
          torchaudio = prev.torchaudio.overrideAttrs (old:
            {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                cudaPkgs.cudaPackages_13.cuda_cudart
                cudaPkgs.autoAddDriverRunpath
              ];
              preFixup = (old.preFixup or "") + ''
                addAutoPatchelfSearchPath "${final.torch}"
              '';
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
            });
          torchvision = prev.torchvision.overrideAttrs (old:
            {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                cudaPkgs.autoAddDriverRunpath
              ];
              preFixup = (old.preFixup or "") + ''
                addAutoPatchelfSearchPath "${final.torch}"
              '';
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
            });
          xgrammar = prev.xgrammar.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ cudaPkgs.cmake ];
            #patches = (old.patches or [ ]) ++ [ ../nix/xgrammar_cmake.patch ];
          });
          vllm = prev.vllm.overrideAttrs (old: {
            patches = (old.patches or [ ]) ++ [ ../nix/vllm_uv2nix_cmake.patch ];
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              cudaPkgs.cmake
              cudaPkgs.ninja
              cudaPkgs.autoAddDriverRunpath
              cudaPkgs.cudaPackages_13.cuda_nvcc
            ];
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ torchLibs ++ [ final.torch ];

            CUDA_HOME = "${cudaPkgs.symlinkJoin {
              name = "cuda-merged-exo";
              paths = builtins.concatMap (p: [ (lib.getBin p) (lib.getLib p) (lib.getDev p) ]) (cudaLibs ++ [ cudaPkgs.cudaPackages_13.cuda_nvcc cuda_cccl_compat ]);
            }}";
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
              "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=${lib.getDev cutlass}"
              "-DFLASH_MLA_SRC_DIR=${lib.getDev flashmla}"
              "-DVLLM_FLASH_ATTN_SRC_DIR=${lib.getDev vllm-flash-attn}"
              "-DQUTLASS_SRC_DIR=${lib.getDev qutlass}"
              "-DTORCH_CUDA_ARCH_LIST=12.0;12.1"
              "-DCUTLASS_NVCC_ARCHS_ENABLED=${cudaPkgs.cudaPackages_13.flags.cmakeCudaArchitecturesString}"
              "-DCAFFE2_USE_CUDNN=ON"
              "-DCAFFE2_USE_CUFILE=ON"
              "-DCUTLASS_ENABLE_CUBLAS=ON"
            ];
          });
        };

      buildSystemsOverlay = final: prev:
        {
          vllm = prev.vllm.overrideAttrs (old: {
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ final.torch ];
            VLLM_TARGET_DEVICE = "empty";
          });
        } // lib.optionalAttrs isDarwin {
          # Use our pure Nix-built MLX with Metal support (macOS only)
          mlx = self'.packages.mlx;
        };

      # Additional overlay for Linux-specific fixes (type checking env).
      # Native wheels have shared lib dependencies we don't need at type-check time.
      linuxOverlay = final: prev:
        let
          ignoreMissing = drv: drv.overrideAttrs { autoPatchelfIgnoreMissingDeps = [ "*" ]; };
          nvidiaPackages = lib.filterAttrs (name: _: lib.hasPrefix "nvidia-" name) prev;

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
            mlx = prev.mlx.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ gguf-tools pkgs.openblas pkgs.fmt ];
              postPatch = ''
                substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
                  --replace-fail "g++" "$CXX"
              '';

              env = {
                DEV_RELEASE = 1;
                CMAKE_ARGS = toString ([
                  (lib.cmakeBool "USE_SYSTEM_FMT" true)
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_JSON" "${pkgs.nlohmann_json.src}")
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_NANOBIND" "${nanobind}")
                  (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
                  (lib.cmakeBool "MLX_BUILD_CPU" true)
                  (lib.cmakeOptionType "string" "CMAKE_INSTALL_LIBDIR" "lib")
                ] ++ lib.optionals isDarwin [
                  (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal_cpp}")
                  (lib.cmakeBool "MLX_BUILD_METAL" true)
                ] ++ lib.optionals pkgs.config.cudaSupport [
                  (lib.cmakeBool "MLX_BUILD_CUDA" true)
                ]);

              };
            });
          };


      python = pkgs.python313;

      pythonSet = (pkgs.callPackage inputs.pyproject-nix.build.packages {
        inherit python;
      }).overrideScope (
        lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.default
          overlay
          exoOverlay
          buildSystemsOverlay
          linuxOverlay
        ]
      );
      cudaPythonSet = (cudaPkgs.callPackage inputs.pyproject-nix.build.packages {
        python = cudaPkgs.python313;
      }).overrideScope (
        lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.default
          overlay
          exoOverlay
          buildSystemsOverlay
          linuxOverlay
          cudaOverlay
        ]
      );
      # mlx-cpu and mlx-cuda-13 both ship mlx/ site-packages files; keep first.
      # mlx-cpu/mlx-cuda-13 and nvidia-cudnn-cu12/cu13 ship overlapping files.
      venvCollisionPaths = lib.optionals isLinux [
        "lib/python3.13/site-packages/cv2*"
        "lib/python3.13/site-packages/mlx*"
        "lib/python3.13/site-packages/nvidia*"
      ];

      editablePythonSet = cudaPythonSet.overrideScope (
        workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; members = [ "exo" "python/*" "bench" ]; }
      );
      evenv = (editablePythonSet.mkVirtualEnv "exo-dev-env"
        {
          exo = [ "mlx" "dev" "cuda" ];
          exo-pyo3-bindings = [ ];
          exo-bench = [ ];
        vllm-engine = [ ];
        }).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
      };
      exoVenv = (pythonSet.mkVirtualEnv "exo-env" {
        exo = [ "mlx" ];
        mlx-engine = [ ];
        exo-pyo3-bindings = [ ];
      }).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
      };
      exoCudaVenv = (cudaPythonSet.mkVirtualEnv "exo-env" {
        exo = lib.optionals cudaPkgs.config.cudaSupport [ "cuda" ];
        exo-pyo3-bindings = [ ];
        vllm-engine = [ ];
      }).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
      };



      # Virtual environment with dev dependencies for testing
      testVenv = (pythonSet.mkVirtualEnv "exo-test-env"
        {
          exo = [ "dev" ]; # Include pytest, pytest-asyncio, pytest-env
          exo-pyo3-bindings = [ ];
        }
      ).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
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

      exoPackage = pkgs.runCommand "exo"
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

      exoCudaPackage = cudaPkgs.runCommand "exo"
        {
          nativeBuildInputs = [ pkgs.makeWrapper ];
        }
        ''
          mkdir -p $out/bin

          # Create wrapper script
          makeWrapper ${exoCudaVenv}/bin/exo $out/bin/exo \
            --set EXO_DASHBOARD_DIR ${self'.packages.dashboard} \
            --set EXO_RESOURCES_DIR ${inputs.self + /resources} \
            ${lib.optionalString isDarwin "--prefix PATH : ${pkgs.macmon}/bin"}
        '';
    in
    {
      # Python package only available on macOS (requires MLX/Metal)
      packages = (lib.optionalAttrs isDarwin
        {
          # Test environment for running pytest outside of Nix sandbox (needs GPU access)
          exo-test-env = testVenv;
        }) // {
        exo = exoPackage;
        exo-cuda = exoCudaPackage;
        exo-bench = mkBenchScript "exo-bench" (inputs.self + /bench/exo_bench.py);
        exo-eval = mkBenchScript "exo-eval" (inputs.self + /bench/exo_eval.py);
        exo-eval-tool-calls = mkBenchScript "exo-eval-tool-calls" (inputs.self + /bench/eval_tool_calls.py);
        exo-get-all-models-on-cluster = mkSimplePythonScript "exo-get-all-models-on-cluster" (inputs.self + /tests/get_all_models_on_cluster.py);
        editable-venv = evenv;
        inherit python;
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
