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

      python = pkgs.python313;

      # Overlay to provide build systems and custom packages
      buildSystemsOverlay = final: prev: {
        # mlx-lm is a git dependency that needs setuptools
        mlx-lm = prev.mlx-lm.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });
        # rouge-score and sacrebleu don't declare setuptools as a build dependency
        rouge-score = prev.rouge-score.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });
        sacrebleu = prev.sacrebleu.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });
        sqlitedict = prev.sqlitedict.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });
        word2number = prev.word2number.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });
      } // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
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
        lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux (
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
        ]);

      # mlx-cpu and mlx-cuda-13 both ship mlx/ site-packages files; keep first.
      # mlx-cpu/mlx-cuda-13 and nvidia-cudnn-cu12/cu13 ship overlapping files.
      venvCollisionPaths = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
        "lib/python3.13/site-packages/mlx*"
        "lib/python3.13/site-packages/nvidia*"
      ];

      exoVenv = (pythonSet.mkVirtualEnv "exo-env" {
          exo_pyo3_bindings = [ ];
          exo = [ ];
        }).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
      };

      # Virtual environment with dev dependencies for testing
      testVenv = (pythonSet.mkVirtualEnv "exo-test-env" workspace.deps.default).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
      };

      benchVenv = pythonSet.mkVirtualEnv "exo-bench-env" {
        exo-bench = [ ];
      };

      exoCudaVenv = (pythonSet.mkVirtualEnv "exo-cuda-env" {
          exo_pyo3_bindings = [ ];
          exo = [ "cuda" ];
      }).overrideAttrs {
        venvIgnoreCollisions = venvCollisionPaths;
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
            ${lib.optionalString pkgs.stdenv.hostPlatform.isDarwin "--prefix PATH : ${pkgs.macmon}/bin"}
        '';


      exoCudaPackage = pkgs.runCommand "exo-cuda"
        {
          nativeBuildInputs = [ pkgs.makeWrapper ];
        }
        ''
          for dir in /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib; do
            if [ -e "$dir/libcuda.so.1" ]; then
              NVIDIA_LIBS="$dir/libcuda.so.1"
              for lib in libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
                [ -e "$dir/$lib" ] && NVIDIA_LIBS="$NVIDIA_LIBS:$dir/$lib"
              done
              export LD_PRELOAD="$NVIDIA_LIBS''${LD_PRELOAD:+:$LD_PRELOAD}"
              break
            fi
          done
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.libnvjitlink}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          mkdir -p $out/bin
          makeWrapper ${exoCudaVenv}/bin/exo $out/bin/exo-cuda \
            --set EXO_DASHBOARD_DIR ${self'.packages.dashboard} \
            --set EXO_RESOURCES_DIR ${inputs.self + /resources} \
        '';
    in
    {
      # Python package only available on macOS (requires MLX/Metal)
      packages = lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin
        {
          exo = exoPackage;
          # Test environment for running pytest outside of Nix sandbox (needs GPU access)
          exo-test-env = testVenv;
      } // {
        exo-cuda-unwrapped = exoCudaPackage;
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
