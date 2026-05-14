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

      cuda_cccl_compat = pkgs.runCommand "cuda-cccl-compat" { } ''
        mkdir -p $out/include
        ln -s ${cudaPackages.cuda_cccl}/include $out/include/cccl
      '';
      cudaLibs = with cudaPackages; [
        cuda_crt
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
      cudaRoot = pkgs.symlinkJoin {
        name = "cuda-merged-exo";
        paths = builtins.concatMap (p: [ (lib.getBin p) (lib.getLib p) (lib.getDev p) ]) (cudaLibs ++ [ cudaPackages.cuda_nvcc cuda_cccl_compat ]);
      };
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
      mkApp =
        let
          libPath = lib.makeLibraryPath (
            [ pkgs.stdenv.cc.cc.lib ] ++ lib.optionals cudaSupport [ cudaRoot ]
          );
        in
        text: name: pkgs.writeShellApplication {
          inherit name;
          text = ''
            LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:}''${LD_LIBRARY_PATH:-}" exec \
              ${lib.optionalString cudaSupport "nixglhost "} ${text}
          '';
          runtimeEnv = {
            EXO_DASHBOARD_DIR = self'.packages.dashboard;
            EXO_RESOURCES_DIR = inputs.self + /resources;
          };
          runtimeInputs = [
            (venv name)
          ] ++ lib.optionals cudaSupport [ pkgs.nix-gl-host ]
          ++ lib.optionals isDarwin [ pkgs.macmon ];
          passthru = {
            venv = venv name;
            evenv = ((pythonSet.overrideScope editableOverlay).mkVirtualEnv "${name}-evenv" (members // { exo = (members.exo or [ ]) ++ [ "dev" ]; })).overrideAttrs (_: {
              venvSkip = [ "lib/python${python.pythonVersion}/site-packages/mlx/share/cmake/*" "lib/python${python.pythonVersion}/site-packages/build_backend.py" ];
            });
          } // lib.optionalAttrs cudaSupport {
            inherit cudaRoot;
          };
        };

    in
    {
      inherit venv;
      mkPythonScript = path: mkApp ''python ${path} "$@"'';
      exo = mkApp ''exo "$@"'' "exo";
    };
in
{
  perSystem =
    { self', pkgs, unfreePkgs, lib, ... }:
    let
      inherit (pkgs.stdenv.hostPlatform) isLinux;
      inherit (mkPythonSet { inherit self' pkgs lib; members = { exo = [ "mlx-cpu" ]; }; }) exo;

      # Virtual environment with dev dependencies for testing
      testVenv = (mkPythonSet {
        inherit self' pkgs lib; members = {
        exo = [ "dev" "mlx-cpu" ]; # Include pytest, pytest-asyncio, pytest-env
      };
      }).venv "exo-test";

      mkBenchScript = (mkPythonSet {
        inherit self' pkgs lib; members = {
        exo = [ "mlx-cpu" ];
        exo-bench = [ ]; # Include pytest, pytest-asyncio, pytest-env
      };
      }).mkPythonScript;

      mkSimplePythonScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ pkgs.python313 ];
        text = ''exec python ${path} "$@"'';
      };
      # if someone is particularly interested in cuda12 support in nix, please open an issue.
      # until then, it's more hassle than its worth
      #cuda12Set = mkPythonSet { inherit self' lib; inherit (unfreePkgs.pkgsCuda.cudaPackages_12) pkgs; members = { exo = [ "mlx-cuda12" ]; }; };
      cuda13Set = mkPythonSet { inherit self' lib; inherit (unfreePkgs.pkgsCuda.cudaPackages_13) pkgs; members = { exo = [ "mlx-cuda13" ]; }; };
    in
    {
      packages = {
        inherit exo;
        # for running tests in ci
        exo-test-env = testVenv;
        exo-bench = mkBenchScript "exo-bench" (inputs.self + /bench/exo_bench.py);
        exo-eval = mkBenchScript "exo-eval" (inputs.self + /bench/exo_eval.py);
        exo-eval-tool-calls = mkBenchScript "exo-eval-tool-calls" (inputs.self + /bench/eval_tool_calls.py);
        # used by ./tests/run_exo_on.sh
        exo-get-all-models-on-cluster = mkSimplePythonScript "exo-get-all-models-on-cluster" (inputs.self + /tests/get_all_models_on_cluster.py);
      } // lib.optionalAttrs isLinux {
        #exo-cuda-12 = cuda12Set.exo;
        exo-cuda-13 = cuda13Set.exo;
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
