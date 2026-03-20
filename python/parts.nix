{ inputs, ... }:
{
  perSystem =
    { config, self', pkgs, lib, system, ... }:
    let
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

      inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux;

      python = pkgs.python313;

      # Overlay to provide build systems and custom packages
      buildSystemsOverlay = final: prev:
        let
          addSetupTools = (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              final.setuptools
            ];
          });
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
            tag = "v3.5.0";
            hash = "sha256-F6T0n37Lbs+B7UHNYzoIQHjNNv3TcMtoXjNrT8ZUlxY=";
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

          mergedCudaLibraries = with pkgs.cudaPackages_13; [
            cuda_cudart # cuda_runtime.h, -lcudart
            cuda_cccl
            libcurand # curand_kernel.h
            libcusparse # cusparse.h
            libcusolver # cusolverDn.h
            cuda_nvtx
            cuda_nvrtc
            # cusparselt # cusparseLt.h
            libcublas
          ];

	  cuda_cccl_compat = pkgs.runCommand "cuda-cccl-compat" {} ''
	    mkdir -p $out/include
	    ln -s ${pkgs.cudaPackages_13.cuda_cccl}/include $out/include/cccl
	  '';
          cudaToolkitRoot = pkgs.symlinkJoin {
              name = "cuda-merged-exo";
              paths = builtins.concatMap (p: [ (lib.getBin p) (lib.getLib p) (lib.getDev p) ]) (mergedCudaLibraries ++ [ pkgs.cudaPackages_13.cuda_nvcc  cuda_cccl_compat ]);
            };

        in

        {
          # mlx-lm is a git dependency that needs setuptools
          mlx-lm = prev.mlx-lm.overrideAttrs addSetupTools;
          # rouge-score and sacrebleu don't declare setuptools as a build dependency
          rouge-score = prev.rouge-score.overrideAttrs addSetupTools;
          sacrebleu = prev.sacrebleu.overrideAttrs addSetupTools;
          sqlitedict = prev.sqlitedict.overrideAttrs addSetupTools;
          word2number = prev.word2number.overrideAttrs addSetupTools;
          fastsafetensors = prev.fastsafetensors.overrideAttrs (old: { nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools final.pybind11 ]; });
          torch = prev.torch.overrideAttrs (old: {
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ torchLibs ++ [ final.typing-extensions final.numpy ];
            autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
          });
          torchaudio = prev.torchaudio.overrideAttrs (old:
            {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
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
              ];
              preFixup = (old.preFixup or "") + ''
                addAutoPatchelfSearchPath "${final.torch}"
              '';
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
            });
          xgrammar = prev.xgrammar.overrideAttrs (old: { nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools final.scikit-build-core final.packaging final.pathspec pkgs.cmake final.nanobind ]; 

  prePatch = ''
cat cpp/nanobind/CMakeLists.txt
'';
  patches = (old.patches or [ ]) ++ [ ./nanobind_cmake.patch ];
});
          vllm = prev.vllm.overrideAttrs (old: {
            patches = (old.patches or [ ]) ++ [ ./vllm_uv2nix_cmake.patch ];
            nativeBuildInputs = with pkgs.cudaPackages_13; (old.nativeBuildInputs or [ ]) ++ [
              final.setuptools
              final.setuptools-scm
              final.scikit-build-core
              pkgs.cmake
              cuda_nvcc
              final.jinja2
              final.wheel
              final.markupsafe
              pkgs.ninja
              pkgs.autoAddDriverRunpath
            ];
            buildInputs = with pkgs.cudaPackages_13; [
              libcufile
              cudnn
              nccl
            ] ++ mergedCudaLibraries;
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ torchLibs ++ [ final.torch ];

            CUDA_HOME = "${cudaToolkitRoot}";
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
            CUTLASS_NVCC_ARCHS_ENABLED = "12.1;12.1";

            UV2NIX_CMAKE_FLAGS_JSON = builtins.toJSON [
              "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=${lib.getDev cutlass}"
              "-DFLASH_MLA_SRC_DIR=${lib.getDev flashmla}"
              "-DVLLM_FLASH_ATTN_SRC_DIR=${lib.getDev vllm-flash-attn}"
              "-DQUTLASS_SRC_DIR=${lib.getDev qutlass}"
              "-DTORCH_CUDA_ARCH_LIST=12.0;12.1"
              "-DCUTLASS_NVCC_ARCHS_ENABLED=${pkgs.cudaPackages_13.flags.cmakeCudaArchitecturesString}"
              "-DCUDA_HOME=${cudaToolkitRoot}"
              "-DCAFFE2_USE_CUDNN=ON"
              "-DCAFFE2_USE_CUFILE=ON"
              "-DCUTLASS_ENABLE_CUBLAS=ON"
            ];


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
        in
        lib.optionalAttrs isLinux (
          (lib.mapAttrs (_: ignoreMissing) nvidiaPackages) // {
            mlx = ignoreMissing prev.mlx;
            mlx-cuda-13 = prev.mlx-cuda-13.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.nvidia-cublas
                final.nvidia-cuda-nvrtc
                final.nvidia-cudnn-cu13
                final.nvidia-nccl-cu13
              ];
              preFixup = ''
                addAutoPatchelfSearchPath ${final.nvidia-cublas}
                addAutoPatchelfSearchPath ${final.nvidia-cuda-nvrtc}
                addAutoPatchelfSearchPath ${final.nvidia-cudnn-cu13}
                addAutoPatchelfSearchPath ${final.nvidia-nccl-cu13}
              '';
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            torch = ignoreMissing prev.torch;
            triton = ignoreMissing prev.triton;
          }
        );



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
      # mlx-cpu and mlx-cuda-13 both ship mlx/ site-packages files; keep first.
      # mlx-cpu/mlx-cuda-13 and nvidia-cudnn-cu12/cu13 ship overlapping files.
      venvCollisionPaths = lib.optionals isLinux [
        "lib/python3.13/site-packages/mlx*"
        "lib/python3.13/site-packages/nvidia*"
      ];

      exoVenv = (pythonSet.mkVirtualEnv "exo-env" {
        exo = lib.optionals isDarwin [ "mlx" ];
        exo-pyo3-bindings = [ ];
      }).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
      };
      exoCudaVenv = (pythonSet.mkVirtualEnv "exo-env" {
        exo = [ "cuda" ];
        exo-pyo3-bindings = [ ];
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

      exoCudaPackage = pkgs.runCommand "exo"
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
