{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils = {
        url = "github:numtide/flake-utils";
        inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = (import nixpkgs) {
          inherit system;
        };

        # Go 1.23 compiler – align with go.mod
        go = pkgs.go_1_23;
        # Build the networking/forwarder Go utility.
        forwarder = pkgs.buildGoModule {
          pname = "exo-forwarder";
          version = "0.1.0";
          src = ./networking/forwarder;

          vendorHash = "sha256-BXIGg2QYqHDz2TNe8hLAGC6jVlffp9766H+WdkkuVgA=";

          # Only the main package at the repository root needs building.
          subPackages = [ "." ];
        };

        buildInputs = with pkgs; [
        ];
        nativeBuildInputs = with pkgs; [
        ];
      in
        {
          packages = {
            inherit forwarder;
            default = forwarder;
          };

          apps = {
            forwarder = {
              type = "app";
              program = "${forwarder}/bin/forwarder";
            };
            python-lsp = {
              type = "app";
              program = "${pkgs.basedpyright}/bin/basedpyright-langserver";
            };
            default = self.apps.${system}.forwarder;
          };

          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.python313
              pkgs.uv
              pkgs.just
              pkgs.protobuf
              pkgs.basedpyright
              pkgs.ruff
              go
            ];

            # TODO: change this into exported env via nix directly???
            shellHook = ''
              export GOPATH=$(mktemp -d)
            '';

            nativeBuildInputs = with pkgs; [
              nixpkgs-fmt
              cmake
            ] ++ buildInputs ++ nativeBuildInputs;

            # fixes libstdc++.so issues and libgl.so issues
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
          };
        }
    );
}