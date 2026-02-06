#!/usr/bin/env nix-shell
#!nix-shell -i bash -p coreutils curl file gzip jq xcbuild yq

set -eu -o pipefail

catalog=${1-}

if [ -z "$catalog" ]; then
    echo "usage: get-sdks-from-catalog.sh <catalog>"
    echo " <catalog>   Apple software update catalog (may be gzipped)" >&2
    exit 1
fi

scratch=$(mktemp)
trap 'rm -f -- "$scratch"' EXIT

if [[ "$(file "$catalog")" =~ gzip ]]; then
    gzcat "$catalog" > "$scratch"
else
    cp --reflink=auto "$catalog" "$scratch"
fi

# Grab all SDK packages from the catalog
filter='.Products[].Packages[] | select(.URL | test(".*CLTools_macOSNMOS_SDK.pkg")) | "\(.URL)|\(.MetadataURL)"'

declare -A package_list
for package in $(plutil -convert json -o - "$scratch" | jq -r "$filter"); do
    package_list[${package%%|*}]=${package#*|}
done

truncate --size 0 "$scratch"
for pkg in "${!package_list[@]}"; do
    ver=$(curl --silent "${package_list[$pkg]}" | xq -r '."pkg-info"."@version"')
    echo "{\"url\": \"$pkg\", \"version\": \"$(cut -d. -f1-3 <<< "$ver")\", \"long_version\": \"$ver\"}" >> "$scratch"
done

jq -r --slurp '
    group_by(.version | split(".")[0])
    | map(max_by(.version))
    | sort_by(.version)[]
    | "Package URL: \(.url)\n  Xcode Ver: \(.version) (\(.long_version))\n"' "$scratch"
