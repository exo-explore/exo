#!/usr/bin/env nix-shell
#!nix-shell -i bash -p coreutils curl git gnutar jq moreutils nix

set -eu -o pipefail

if [ ! -v 2 ]; then
    echo "usage: lock-sdk-deps.sh <SDK version> <Packages>" >&2
    echo " <SDK version>   Decimal-separated version number." >&2
    echo "                 Must correspond to a tag in https://github.com/apple-oss-distributions/distribution-macOS" >&2
    echo " <Packages>      List of packages from the distributions-macOS repository." >&2
    echo "                 Packages not in the repository at the tag for <SDK version> will be ignored."
    exit 1
fi

pkgdir=$(dirname "$(dirname "$(realpath "$0")")")

lockfile=$pkgdir/metadata/apple-oss-lockfile.json
if [ ! -e "$lockfile" ]; then
    touch "$lockfile"
fi

workdir=$(mktemp -d)
trap 'rm -rf -- "$workdir"' EXIT

sdkVersion=$1; shift
tag="macos-${sdkVersion//.}"

declare -a packages=("$@")

echo "Locking versions for macOS $sdkVersion using tag '$tag'..."

pushd "$workdir" > /dev/null

git clone --branch "$tag" https://github.com/apple-oss-distributions/distribution-macOS.git &> /dev/null
cd distribution-macOS

for package in "${packages[@]}"; do
    # If the tag exists in `release.json`, use that as an optimization to avoid downloading unnecessarily from Github.
    packageTag=$(jq -r --arg package "$package" '.projects[] | select(.project == $package) | .tag' release.json)
    packageCommit=$(git ls-tree -d HEAD "$package" | awk '{print $3}')

    if [ ! -d "$package" ]; then
        packageCommit=HEAD
    fi

    # However, sometimes it doesn’t exist. In that case, fall back to cloning the repo and check manually
    # which tag corresponds to the commit from the submodule.
    if [ -z "$packageTag" ]; then
        git clone --no-checkout "https://github.com/apple-oss-distributions/$package.git" ../source &> /dev/null
        pushd ../source > /dev/null
        packageTag=$(git tag --points-at "$packageCommit")
        popd > /dev/null
        rm -rf ../source
    fi

    packageVersion=${packageTag##"$package"-}

    curl -OL "https://github.com/apple-oss-distributions/$package/archive/$packageTag.tar.gz" &> /dev/null
    tar axf "$packageTag.tar.gz"

    packageHash=$(nix --extra-experimental-features nix-command hash path "$package-$packageTag")

    pkgsjson="{\"$sdkVersion\": {\"$package\": {\"version\": \"$packageVersion\", \"hash\": \"$packageHash\"}}}"

    echo "   - Locking $package to version $packageVersion with hash '$packageHash'"
    jq --argjson pkg "$pkgsjson" -S '. * $pkg' "$lockfile" | sponge "$lockfile"
done

popd > /dev/null
