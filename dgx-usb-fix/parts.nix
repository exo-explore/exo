{
  perSystem =
    { pkgs, lib, ... }:
    let
      repoSource = ../.;

      rootWrapper =
        name: targetScript:
        pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [ pkgs.git ];
          text = ''
            set -euo pipefail

            command -v sudo >/dev/null 2>&1 || {
              printf 'ERROR: sudo is required for %s\n' "$0" >&2
              exit 1
            }

            repo_root="''${DGX_USB_FIX_REPO:-}"
            fallback_root="${repoSource}"
            if [[ -z "$repo_root" ]]; then
              repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
            fi
            if [[ -z "$repo_root" && -x "$PWD/${targetScript}" ]]; then
              repo_root="$PWD"
            fi
            if [[ -z "$repo_root" ]]; then
              repo_root="$fallback_root"
            fi

            target="$repo_root/${targetScript}"
            if [[ ! -x "$target" ]]; then
              printf 'ERROR: expected executable %s\n' "$target" >&2
              printf 'Run this from the repo checkout, or set DGX_USB_FIX_REPO=/path/to/exo.\n' >&2
              exit 1
            fi

            if [[ "''${1:-}" == "-h" || "''${1:-}" == "--help" ]]; then
              exec env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 "DGX_USB_FIX_REPO=$repo_root" "$target" "$@"
            fi

            sudo_env=(env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 "DGX_USB_FIX_REPO=$repo_root")
            for var in KERNEL_VERSION WORKDIR INSTALL_DIR MOK_KEY MOK_CERT MOK_SUBJECT; do
              if [[ -v "$var" ]]; then
                sudo_env+=("$var=''${!var}")
              fi
            done

            exec sudo "''${sudo_env[@]}" "$target" "$@"
          '';
        };

      linuxRootCommands = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
        (rootWrapper "dgx-usb-fix-create-mok-key" "dgx-usb-fix/create-mok-key.sh")
        (rootWrapper "dgx-usb-fix-diagnose" "dgx-usb-fix/diagnose.sh")
        (rootWrapper "dgx-usb-fix-install" "dgx-usb-fix/build-and-install.sh")
      ];
    in
    {
      devShells.dgx-usb-fix = pkgs.mkShell {
        packages = with pkgs; [
          bash
          bc
          binutils
          bison
          coreutils
          curl
          diffutils
          dpkg
          elfutils
          findutils
          flex
          gawk
          gcc
          git
          gnugrep
          gnumake
          gnused
          gnutar
          gzip
          kmod
          openssl
          patch
          perl
          pkg-config
          python3
          xz
          zstd
        ] ++ lib.optionals stdenv.hostPlatform.isLinux ([
          iproute2
          mokutil
          usbutils
        ] ++ lib.optional (pkgs ? pahole) pkgs.pahole) ++ linuxRootCommands;

        shellHook = ''
          export DGX_USB_FIX_NIX_ENV=1
          export MOK_KEY="''${MOK_KEY:-/root/MOK.priv}"
          export MOK_CERT="''${MOK_CERT:-/root/MOK.der}"
          fallback_root="${repoSource}"
          if [[ -z "''${DGX_USB_FIX_REPO:-}" ]]; then
            if command -v git >/dev/null 2>&1 && git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
              export DGX_USB_FIX_REPO="$git_root"
            elif [[ -x "$PWD/dgx-usb-fix/build-and-install.sh" ]]; then
              export DGX_USB_FIX_REPO="$PWD"
            else
              export DGX_USB_FIX_REPO="$fallback_root"
            fi
          fi
          cat <<'EOF'
DGX USB fix shell

Root commands are wrapped so sudo preserves this shell's PATH:
  dgx-usb-fix-diagnose
  dgx-usb-fix-create-mok-key
  dgx-usb-fix-install [--skip-load]
EOF
        '';
      };
    };
}
