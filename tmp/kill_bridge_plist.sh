#!/usr/bin/env bash
set -euo pipefail

PREFS="${PREFS:-/Library/Preferences/SystemConfiguration/preferences.plist}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
injson="$tmpdir/in.json"
outjson="$tmpdir/out.json"
plutil -convert json -o "$injson" "$PREFS"

perl -Mstrict -Mwarnings -MJSON::PP -e '
  my ($in, $out) = @ARGV;

  open my $fh, "<", $in or die "open $in: $!";
  local $/;
  my $txt = <$fh>;
  close $fh;

  my $json = JSON::PP->new->utf8->relaxed(1);
  my $d = $json->decode($txt);

  if (ref($d->{VirtualNetworkInterfaces}) eq "HASH"
      && ref($d->{VirtualNetworkInterfaces}{Bridge}) eq "HASH") {
    delete $d->{VirtualNetworkInterfaces}{Bridge}{bridge0};
  }

  my @bridge_svcs;
  if (ref($d->{NetworkServices}) eq "HASH") {
    for my $k (keys %{ $d->{NetworkServices} }) {
      my $svc = $d->{NetworkServices}{$k};
      next unless ref($svc) eq "HASH";
      my $iface = $svc->{Interface};
      next unless ref($iface) eq "HASH";
      my $dev = $iface->{DeviceName};
      if (defined $dev && $dev eq "bridge0") {
        push @bridge_svcs, $k;
      }
    }
    delete @{ $d->{NetworkServices} }{ @bridge_svcs } if @bridge_svcs;
  }

  my %is_bridge = map { $_ => 1 } @bridge_svcs;

  if (ref($d->{Sets}) eq "HASH") {
    for my $setk (keys %{ $d->{Sets} }) {
      my $set = $d->{Sets}{$setk};
      next unless ref($set) eq "HASH";
      my $net = $set->{Network};
      next unless ref($net) eq "HASH";

      if (ref($net->{Interface}) eq "HASH") {
        delete $net->{Interface}{bridge0};
      }

      if (ref($net->{Service}) eq "HASH" && @bridge_svcs) {
        for my $svc (@bridge_svcs) {
          delete $net->{Service}{$svc};
        }
      }

      my $g = $net->{Global};
      if (ref($g) eq "HASH"
          && ref($g->{IPv4}) eq "HASH"
          && ref($g->{IPv4}{ServiceOrder}) eq "ARRAY"
          && @bridge_svcs) {

        my @so = @{ $g->{IPv4}{ServiceOrder} };
        @so = grep { !defined($_) || !$is_bridge{$_} } @so;
        $g->{IPv4}{ServiceOrder} = \@so;
      }
    }
  }

  open my $oh, ">", $out or die "open $out: $!";
  print $oh JSON::PP->new->utf8->canonical(1)->pretty(1)->encode($d);
  close $oh;
' "$injson" "$outjson"

# Convert JSON -> plist (write back as binary1; change to xml1 if you prefer)
plutil -convert xml1 -o "$PREFS" "$outjson"

# Ask configd to reload SystemConfiguration state
killall -HUP configd 2>/dev/null || true
