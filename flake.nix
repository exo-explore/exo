{
  description = "The development environment for Exo";

  inputs = {
    # --- NIXPKGS VERSIONS ---

    # latest stable and unstable nixpkgs
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";

    # add specific versions of nixpkgs if needed (and initialize
    # in pkgs.nix module to make work with flake parts)
    #nixpkgs-23_11 = "github:NixOS/nixpkgs/nixos-23.11";

    # --- NIXPKGS VERSIONS END ---

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
  };

  nixConfig = {
    extra-trusted-public-keys = "exo.cachix.org-1:okq7hl624TBeAR3kV+g39dUFSiaZgLRkLsFBCuJ2NZI=";
    extra-substituters = "https://exo.cachix.org";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule

        # configures `pkgs*` used by flake-parts
        ./nix/modules/pkgs.nix

        ./dashboard/parts.nix
        ./rust/parts.nix
        ./python/parts.nix
      ];

      debug = true; # Enable options autocompletion

      perSystem = { config, self', pkgs, lib, system, ... }:
        {
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
              swift-format.enable = true;
              shfmt.enable = true;
              taplo.enable = true;
            };
          };

          packages = lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin (
            let
              uvLock = builtins.fromTOML (builtins.readFile ./uv.lock);
              mlxPackage = builtins.head (builtins.filter (p: p.name == "mlx" && p.source ? git) uvLock.package);
              uvLockMlxVersion = mlxPackage.version;
              uvLockMlxRev = builtins.elemAt (builtins.split "#" mlxPackage.source.git) 2;
            in
            {
              metal-toolchain = pkgs.callPackage ./nix/metal-toolchain.nix { };
              mlx = pkgs.callPackage ./nix/mlx.nix {
                inherit (self'.packages) metal-toolchain;
                inherit uvLockMlxVersion uvLockMlxRev;
              };
              default = self'.packages.exo;
            }
          );

          devShells.default = pkgs.mkShell {
            inputsFrom = [ self'.checks.cargo-build ];
            packages = with pkgs; [
              # FORMATTING
              config.treefmt.build.wrapper

              # PYTHON
              self'.packages.python
              unstable.uv
              ruff
              basedpyright

              # RUST
              config.rust.toolchain
              maturin

              # NIX
              nixd
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
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${self'.packages.python}/lib"
              ${lib.optionalString pkgs.stdenv.isLinux ''
                export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
              ''}
            '';
          };
        };
    };
}
