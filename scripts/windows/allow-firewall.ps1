# Allow inbound TCP/UDP so other LAN nodes can join this exo process.
# PAIR's Windows installer does the equivalent for its proxy ports.
#
# TCP 52414 = zenoh, TCP 52415 = API/dashboard
# UDP 52413 = IPv4/IPv6 discovery (broadcast + multicast)
# UDP 5353  = mDNS (_exo._tcp.local.)

#Requires -RunAsAdministrator

$ErrorActionPreference = 'Stop'
$RuleName = 'exo local cluster'
$UdpRuleName = 'exo local cluster UDP'

foreach ($name in @($RuleName, $UdpRuleName)) {
    if (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $name
    }
}

New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 52413,52414,52415 | Out-Null
New-NetFirewallRule -DisplayName $UdpRuleName -Direction Inbound -Action Allow `
    -Protocol UDP -LocalPort 52413,5353 | Out-Null
Write-Host "Firewall rules '$RuleName' allow TCP 52413-52415 and UDP 52413,5353 inbound."
