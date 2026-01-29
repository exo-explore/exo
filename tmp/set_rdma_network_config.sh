#!/usr/bin/env bash

set -euo pipefail

PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"

# Remove bridge0 interface
ifconfig bridge0 &>/dev/null && {
  ifconfig bridge0 | grep -q 'member' && {
    ifconfig bridge0 | awk '/member/ {print $2}' | xargs -n1 ifconfig bridge0 deletem 2>/dev/null || true
  }
  ifconfig bridge0 destroy 2>/dev/null || true
}

# Remove Thunderbolt Bridge from VirtualNetworkInterfaces in preferences.plist
/usr/libexec/PlistBuddy -c "Delete :VirtualNetworkInterfaces:Bridge:bridge0" "$PREFS" 2>/dev/null || true

networksetup -listlocations | grep -q exo || {
  networksetup -createlocation exo
}

networksetup -switchtolocation exo
networksetup -listallhardwareports |
  awk -F': ' '/Hardware Port: / {print $2}' |
  while IFS=":" read -r name; do
    case "$name" in
    "Ethernet Adapter"*) ;;
    "Thunderbolt Bridge") ;;
    "Thunderbolt "*)
      networksetup -listallnetworkservices |
        grep -q "EXO $name" ||
        networksetup -createnetworkservice "EXO $name" "$name" 2>/dev/null ||
        continue
      networksetup -setdhcp "EXO $name"
      ;;
    *)
      networksetup -listallnetworkservices |
        grep -q "$name" ||
        networksetup -createnetworkservice "$name" "$name" 2>/dev/null ||
        continue
      ;;
    esac
  done

networksetup -listnetworkservices | grep -q "Thunderbolt Bridge" && {
  networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off
} || true
