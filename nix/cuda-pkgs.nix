{ nixpkgs, system }:
let
  pkgs = import nixpkgs { inherit system; };
in
if system == "aarch64-linux" then
  import nixpkgs
  {
    inherit system;
    config = {
      allowUnfree = true;
      allowBroken = true;
      allowUnsupportedSystem = true;
      cudaSupport = true;
      cudaCapabilities = [ "12.1" ];
    };
    overlays = [
      (final: prev:
        let
          cudaCompatStub = cfinal: cprev: {
            cuda_compat = prev.runCommand "cuda13.0-cuda_compat-stub" { } "mkdir -p $out";
          };
        in
        {
          cudaPackages = prev.cudaPackages_13.overrideScope cudaCompatStub // {
            override = args:
              (prev.cudaPackages_13.override args).overrideScope cudaCompatStub;
          };

          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (pyFinal: pyPrev: {
              fastsafetensors = pyFinal.buildPythonPackage {
                pname = "fastsafetensors";
                version = "0.2.2";
                src = prev.fetchFromGitHub {
                  owner = "foundation-model-stack";
                  repo = "fastsafetensors";
                  rev = "v0.2.2";
                  hash = "";
                };
                pyproject = true;
                build-system = [
                  pyFinal.setuptools
                  pyFinal.pybind11
                ];
                buildInputs = [
                  final.cudaPackages.cuda_cudart
                  final.cudaPackages.cuda_nvml_dev
                ];
                nativeBuildInputs = [
                  final.cudaPackages.cuda_nvcc
                ];
                dependencies = [
                  pyFinal.typer
                ];
                env.CUDA_HOME = "${final.cudaPackages.cuda_nvcc}";
                pythonImportsCheck = [ "fastsafetensors" ];
              };
              cupy = pyPrev.cupy.override {
                cudaPackages = final.cudaPackages;
              };

              bitsandbytes = pyPrev.bitsandbytes.overrideAttrs (old: {
                preConfigure = (old.preConfigure or "") + ''
                  export CXXFLAGS="''${CXXFLAGS:-} -I${final.cudaPackages.cuda_crt}/include"
                  export CUDAFLAGS="''${CUDAFLAGS:-} -I${final.cudaPackages.cuda_crt}/include"
                  export CMAKE_CUDA_FLAGS="''${CMAKE_CUDA_FLAGS:-} -I${final.cudaPackages.cuda_crt}/include"
                '';
                buildInputs = (old.buildInputs or [ ]) ++ [ final.cudaPackages.cuda_crt ];
              });

              vllm = pyPrev.vllm.overrideAttrs (old: {
                buildInputs = (old.buildInputs or [ ]) ++ [
                  final.cuda_cccl_with_prefix
                  final.cudaPackages.cuda_crt
                ];
                preConfigure = (old.preConfigure or "") + ''
                  export CXXFLAGS="''${CXXFLAGS:-} -I${final.cuda_cccl_with_prefix}/include -I${final.cudaPackages.cuda_crt}/include"
                  export CUDAFLAGS="''${CUDAFLAGS:-} -I${final.cuda_cccl_with_prefix}/include -I${final.cudaPackages.cuda_crt}/include"
                '';
              });
            })
          ];

          magma-cuda-static = prev.magma-cuda-static.overrideAttrs (old: {
            postPatch = (old.postPatch or "") + ''
              sed -i '/err = cudaGetDeviceProperties( &prop, dev );/a\        int clock_khz = 0; cudaDeviceGetAttribute(\&clock_khz, cudaDevAttrClockRate, dev);' interface_cuda/interface.cpp
              sed -i 's/prop\.clockRate/clock_khz/g' interface_cuda/interface.cpp
            '';
          });

          cuda_cccl_with_prefix = prev.runCommand "cuda13.0-cuda_cccl-with-cccl-prefix" { } ''
            mkdir -p $out/include
            ln -s ${final.cudaPackages.cuda_cccl}/include $out/include/cccl
          '';

          opencv = prev.opencv.override { enableCuda = false; };
          opencv4 = prev.opencv4.override { enableCuda = false; };
        })
    ];
  }
else
  null
