# Since the same derivation can be depended on in multiple ways, we need to
# accumulate *each* role (i.e. host and target platforms relative the depending
# derivation) in which the derivation is used.
#
# The role is intended to be used as part of other variables names like
#  - $NIX_SOMETHING${role_post}

function getRole() {
    case $1 in
        -1)
            role_post='_FOR_BUILD'
            ;;
        0)
            role_post=''
            ;;
        1)
            role_post='_FOR_TARGET'
            ;;
        *)
            echo "@name@: used as improper sort of dependency" >&2
            return 1
            ;;
    esac
}

# `hostOffset` describes how the host platform of the package is slid relative
# to the depending package. `targetOffset` likewise describes the target
# platform of the package. Both are brought into scope of the setup hook defined
# for dependency whose setup hook is being processed relative to the package
# being built.

function getHostRole()   {
    getRole "$hostOffset"
}
function getTargetRole() {
    getRole "$targetOffset"
}

# `depHostOffset` describes how the host platform of the dependencies are slid
# relative to the depending package. `depTargetOffset` likewise describes the
# target platform of dependenices. Both are brought into scope of the
# environment hook defined for the dependency being applied relative to the
# package being built.

function getHostRoleEnvHook()   {
    getRole "$depHostOffset"
}
function getTargetRoleEnvHook() {
    getRole "$depTargetOffset"
}

# This variant is intended specifically for code-producing tool wrapper scripts
# `NIX_@wrapperName@_TARGET_*_@suffixSalt@` tracks this (needs to be an exported
# env var so can't use fancier data structures).
function getTargetRoleWrapper() {
    case $targetOffset in
        -1)
            export NIX_@wrapperName@_TARGET_BUILD_@suffixSalt@=1
            ;;
        0)
            export NIX_@wrapperName@_TARGET_HOST_@suffixSalt@=1
            ;;
        1)
            export NIX_@wrapperName@_TARGET_TARGET_@suffixSalt@=1
            ;;
        *)
            echo "@name@: used as improper sort of dependency" >&2
            return 1
            ;;
    esac
}
