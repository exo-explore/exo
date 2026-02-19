{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    crane.url = "github:ipetkov/crane";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    dream2nix = {
      url = "github:nix-community/dream2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    # Python packaging with uv2nix
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Pinned nixpkgs for swift-format (swift is broken on x86_64-linux in newer nixpkgs)
    nixpkgs-swift.url = "github:NixOS/nixpkgs/08dacfca559e1d7da38f3cf05f1f45ee9bfd213c";
  };

  nixConfig = {
    extra-trusted-public-keys = "exo.cachix.org-1:okq7hl624TBeAR3kV+g39dUFSiaZgLRkLsFBCuJ2NZI=";
    extra-substituters = "https://exo.cachix.org";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule
        ./dashboard/parts.nix
        ./rust/parts.nix
        ./python/parts.nix
      ];

      perSystem =
        { config, self', inputs', pkgs, lib, system, ... }:
        let
          # Use pinned nixpkgs for swift-format (swift is broken on x86_64-linux in newer nixpkgs)
          pkgsSwift = import inputs.nixpkgs-swift { inherit system; };
        in
        {
          # Allow unfree for metal-toolchain (needed for Darwin Metal packages)
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfreePredicate = pkg: (pkg.pname or "") == "metal-toolchain";
            overlays = [
              (import ./nix/apple-sdk-overlay.nix)
            ];
          };
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixpkgs-fmt.enable = true;
              ruff-format = {
                enable = true;
                excludes = [ "rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi" ];
              };
              rustfmt = {
                enable = true;
                package = config.rust.toolchain;
              };
              prettier = {
                enable = true;
                package = self'.packages.prettier-svelte;
                includes = [ "*.ts" "*.svelte" ];
              };
              swift-format = {
                enable = true;
                package = pkgsSwift.swiftPackages.swift-format;
              };
              shfmt.enable = true;
            };
          };

          packages = lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin (
            let
              uvLock = builtins.fromTOML (builtins.readFile ./uv.lock);
              mlxPackage = builtins.head (builtins.filter (p: p.name == "mlx" && p.source ? git) uvLock.package);
              uvLockMlxVersion = mlxPackage.version;
            in
            {
              metal-toolchain = pkgs.callPackage ./nix/metal-toolchain.nix { };
              mlx = pkgs.callPackage ./nix/mlx.nix {
                inherit (self'.packages) metal-toolchain;
                inherit uvLockMlxVersion;
              };
              default = self'.packages.exo;
            }
          );

          devShells.default = with pkgs; pkgs.mkShell {
            inputsFrom = [ self'.checks.cargo-build ];

            packages =
              [
                # FORMATTING
                config.treefmt.build.wrapper

                # PYTHON
                python313
                uv
                ruff
                basedpyright

                # RUST
                config.rust.toolchain
                maturin

                # NIX
                nixpkgs-fmt

                # SVELTE
                nodejs

                # MISC
                just
                jq
              ]
              ++ lib.optionals stdenv.isLinux [
                unixtools.ifconfig
              ]
              ++ lib.optionals stdenv.isDarwin [
                macmon
              ];

            OPENSSL_NO_VENDOR = "1";

            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${python313}/lib"
              ${lib.optionalString stdenv.isLinux ''
                export LD_LIBRARY_PATH="${openssl.out}/lib:$LD_LIBRARY_PATH"
              ''}
            '';
          };
        };
    };
}
