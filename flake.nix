{
  description = "Exo development flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

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
      in
      {
        packages = {
          inherit forwarder;
          default = forwarder;
        };

        apps.forwarder = {
          type = "app";
          program = "${forwarder}/bin/forwarder";
        };
        apps.python-lsp = {
          type = "app";
          program = "${pkgs.basedpyright}/bin/basedpyright-langserver";
        };
        apps.default = self.apps.${system}.forwarder;

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python313
            pkgs.uv
            pkgs.just
            pkgs.protobuf
            pkgs.rustc
            pkgs.cargo
            pkgs.basedpyright
            pkgs.ruff
            go
          ];

          shellHook = ''
            export GOPATH=$(mktemp -d)
          '';
        };
      }
    );
}