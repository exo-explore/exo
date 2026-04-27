{
  perSystem =
    { pkgs, lib, ... }:
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
        ] ++ lib.optional (pkgs ? pahole) pkgs.pahole);

        shellHook = ''
          export DGX_USB_FIX_NIX_ENV=1
          cat <<'EOF'
DGX USB fix shell

Root commands must preserve this shell's PATH, for example:
  sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/diagnose.sh
  sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/build-and-install.sh
EOF
        '';
      };
    };
}
