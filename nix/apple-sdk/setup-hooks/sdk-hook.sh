local role_post
getHostRole

local sdkVersionVar=NIX_APPLE_SDK_VERSION${role_post}
local developerDirVar=DEVELOPER_DIR${role_post}

local sdkVersionArr=(@sdkVersion@)
local sdkVersion
sdkVersion=$(printf "%02d%02d%02d" "${sdkVersionArr[0]-0}" "${sdkVersionArr[1]-0}" "${sdkVersionArr[2]-0}")

if [ "$sdkVersion" -gt "${!sdkVersionVar-000000}" ]; then
    export "$developerDirVar"='@out@'
    export "$sdkVersionVar"="$sdkVersion"
    export "SDKROOT${role_post}"="${!developerDirVar}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
fi

unset -v role_post developerDirVar sdkVersion sdkVersionArr sdkVersionVar
