#!/usr/bin/env bash
set -euo pipefail

networksetup -listallnetworkservices | grep -q '^Thunderbolt Bridge$' \
  && echo "Disabling bridge in networksetup" \
  && networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off

networksetup -listallnetworkservices | grep -q '^\*Thunderbolt Bridge$' \
  && echo "Bridge disabled in networksetup"

ifconfig bridge0 &>/dev/null && {
  ifconfig bridge0 | grep -q 'member' && echo "Removing bridge members in ifconfig" && {
    ifconfig bridge0 | \
      awk '/member/ {print $2}' | \
      xargs -n1 sudo ifconfig bridge0 deletem
  }
  ifconfig bridge0 | grep -q 'status: active' && sudo ifconfig bridge0 down
  ifconfig bridge0 | grep -q 'status: inactive' && echo "Bridge disabled in ifconfig"
}

for iface in $(seq 2 7); do
  sudo ipconfig set "en$iface" dhcp && echo "enabled dhcp on en$iface" || echo "failed to enable dhcp on en$iface"
done

