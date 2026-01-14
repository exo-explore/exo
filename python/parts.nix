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
      overlay = workspace.mkPyprojectOverlay { };

      # Override overlay to inject Nix-built components
      exoOverlay = final: prev: {
        # Replace workspace exo_pyo3_bindings with Nix-built wheel
        exo-pyo3-bindings = pkgs.stdenv.mkDerivation {
          pname = "exo-pyo3-bindings";
          version = "0.1.0";
          src = self'.packages.exo_pyo3_bindings;
          # Install from pre-built wheel
          nativeBuildInputs = [ final.pyprojectWheelHook ];
          dontStrip = true;
        };
      };

      python = pkgs.python313;

      # Overlay to provide build systems for source builds
      buildSystemsOverlay = final: prev: {
        # mlx-lm is a git dependency that needs setuptools
        mlx-lm = prev.mlx-lm.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });

        # Build MLX from source with proper dependencies
        mlx = prev.mlx.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            pkgs.cmake
            pkgs.ninja
            final.nanobind
          ];
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.darwin.apple_sdk.frameworks.Accelerate
            pkgs.darwin.apple_sdk.frameworks.Metal
            pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders
            pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShadersGraph
          ];
          dontUseCmakeConfigure = true;
        });
      };

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
      exoVenv = pythonSet.mkVirtualEnv "exo-env" workspace.deps.default;

      # Virtual environment with dev dependencies for testing
      testVenv = pythonSet.mkVirtualEnv "exo-test-env" (
        workspace.deps.default // {
          exo = [ "dev" ]; # Include pytest, pytest-asyncio, pytest-env
        }
      );

      exoPackage = pkgs.runCommand "exo"
        {
          nativeBuildInputs = [ pkgs.makeWrapper ];
        }
        ''
          mkdir -p $out/bin

          # Create wrapper scripts
          for script in exo exo-master exo-worker; do
            makeWrapper ${exoVenv}/bin/$script $out/bin/$script \
              --set DASHBOARD_DIR ${self'.packages.dashboard}
          done
        '';

      pyinstallerPackage =
        let
          venv = pythonSet.mkVirtualEnv "exo-pyinstaller-env" (
            workspace.deps.default
            // {
              # Include pyinstaller in the environment
              exo = [ "dev" ];
            }
          );
        in
        pkgs.stdenv.mkDerivation {
          pname = "exo-pyinstaller";
          version = "0.3.0";

          src = inputs.self;

          nativeBuildInputs = [ venv pkgs.makeWrapper pkgs.macmon pkgs.darwin.system_cmds ];

          buildPhase = ''
            # macmon must be in PATH for PyInstaller to bundle it
            export PATH="${pkgs.macmon}/bin:$PATH"
            # HOME must be writable for PyInstaller's cache
            export HOME="$TMPDIR"

            # Copy dashboard to expected location
            mkdir -p dashboard/build
            cp -r ${self'.packages.dashboard}/* dashboard/build/

            # Run PyInstaller
            ${venv}/bin/python -m PyInstaller packaging/pyinstaller/exo.spec
          '';

          installPhase = ''
            cp -r dist/exo $out
          '';
        };
    in
    {
      # Python package only available on macOS for now due to the dependency on
      # mlx/mlx-cpu being tricky to build on Linux. We can either remove this
      # dependency in the PyProject or build it with Nix.
      packages = lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
        exo = exoPackage;
        exo-pyinstaller = pyinstallerPackage;
      };

      checks = {
        # Ruff linting (works on all platforms)
        lint = pkgs.runCommand "ruff-lint" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}/
          touch $out
        '';
      }
      # Pytest only on macOS (requires MLX)
      // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
        pytest = pkgs.runCommand "pytest"
          {
            nativeBuildInputs = [ testVenv ];
          } ''
          export HOME="$TMPDIR"
          export EXO_TESTS=1
          cd ${inputs.self}
          ${testVenv}/bin/python -m pytest src -m "not slow" --import-mode=importlib
          touch $out
        '';
      };
    };
}
