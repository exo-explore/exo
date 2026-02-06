#!/usr/bin/env nix-shell
#!nix-shell -i bash -p coreutils jq

set -eu -o pipefail

pkgdir=$(dirname "$(dirname "$(realpath "$0")")")

echo '{}' > "$pkgdir/metadata/apple-oss-lockfile.json"

declare -a versions
readarray -t versions < <(jq -r '.[].version' "$pkgdir/metadata/versions.json")

declare -a packages=(
    CarbonHeaders
    CommonCrypto
    IOAudioFamily
    IOFireWireFamily
    IOFWDVComponents
    IOFireWireAVC
    IOFireWireSBP2
    IOFireWireSerialBusProtocolTransport
    IOGraphics
    IOHIDFamily
    IONetworkingFamily
    IOSerialFamily
    IOStorageFamily
    IOBDStorageFamily
    IOCDStorageFamily
    IODVDStorageFamily
    IOUSBFamily
    IOKitUser
    Libc
    Libinfo
    Libm
    Libnotify
    Librpcsvc
    Libsystem
    OpenDirectory
    Security
    architecture
    configd
    copyfile
    dtrace
    dyld
    eap8021x
    hfs
    launchd
    libclosure
    libdispatch
    libmalloc
    libplatform
    libpthread
    mDNSResponder
    objc4
    ppp
    removefile
    xnu
)

for version in "${versions[@]}"; do
    "$pkgdir/scripts/lock-sdk-deps.sh" "$version" "${packages[@]}"
done
