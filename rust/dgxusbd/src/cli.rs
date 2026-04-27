use std::fmt;
use std::str::FromStr;

use crate::usb::UsbSelector;
use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "dgxusbd",
    about = "Userspace USB CDC-NCM probe/bridge for Spark-to-Mac direct USB-C networking"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List USB devices visible to the current user.
    List {
        #[command(flatten)]
        usb: UsbTargetArgs,

        /// Show every USB device instead of only the target Apple Mac device.
        #[arg(long)]
        all: bool,
    },

    /// Open and inspect a candidate CDC-NCM USB device.
    Probe {
        #[command(flatten)]
        usb: UsbTargetArgs,

        #[command(flatten)]
        claim: ClaimArgs,
    },

    /// Claim the selected pair, select data altsetting, open endpoints, and do one timed read.
    UsbSmoke {
        #[command(flatten)]
        open: OpenPairArgs,

        /// Skip CDC-NCM class setup requests and only claim/select/open.
        #[arg(long)]
        skip_ncm_init: bool,

        /// Timed bulk IN read timeout in milliseconds.
        #[arg(long, default_value_t = 250)]
        read_timeout_ms: u64,
    },

    /// Create a Linux TAP device and exit after optional hold time.
    TapSmoke {
        #[command(flatten)]
        tap: TapArgs,

        /// Keep the TAP open for this many seconds before exiting.
        #[arg(long, default_value_t = 0)]
        hold_seconds: u64,
    },

    /// Run a conservative one-pair TAP-to-CDC-NCM bridge.
    Bridge {
        #[command(flatten)]
        open: OpenPairArgs,

        #[command(flatten)]
        tap: TapArgs,

        /// Stop after this many seconds. If omitted, run until interrupted.
        #[arg(long)]
        duration_seconds: Option<u64>,

        /// Stop after this many TAP/USB events. Useful for bounded tests.
        #[arg(long)]
        max_events: Option<u64>,

        /// Per USB bulk transfer timeout in milliseconds.
        #[arg(long, default_value_t = 100)]
        usb_timeout_ms: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsbId(u16);

impl UsbId {
    #[must_use]
    #[inline]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    #[must_use]
    #[inline]
    pub const fn get(self) -> u16 {
        self.0
    }
}

impl fmt::Display for UsbId {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}", self.0)
    }
}

impl FromStr for UsbId {
    type Err = String;

    #[inline]
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        let without_prefix = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
            .unwrap_or(trimmed);

        u16::from_str_radix(without_prefix, 16)
            .or_else(|_| trimmed.parse::<u16>())
            .map(Self)
            .map_err(|err| err.to_string())
    }
}

#[derive(Clone, Copy, Debug, Args)]
pub struct UsbTargetArgs {
    /// USB vendor ID to match. USB IDs are parsed as hex first, so 05ac works.
    #[arg(long, default_value_t = UsbId::new(crate::usb::APPLE_VENDOR_ID))]
    pub vendor_id: UsbId,

    /// USB product ID to match. USB IDs are parsed as hex first, so 1905 works.
    #[arg(long, default_value_t = UsbId::new(crate::usb::APPLE_MAC_PRODUCT_ID))]
    pub product_id: UsbId,
}

impl UsbTargetArgs {
    #[must_use]
    #[inline]
    pub const fn selector(self) -> UsbSelector {
        UsbSelector {
            vendor_id: self.vendor_id.get(),
            product_id: self.product_id.get(),
        }
    }
}

#[derive(Clone, Copy, Debug, Args)]
pub struct ClaimArgs {
    /// Claim detected CDC-NCM control/data interfaces, then release on exit.
    #[arg(long)]
    pub claim: bool,

    /// Detach any bound kernel drivers before claiming interfaces.
    #[arg(long, requires = "claim")]
    pub detach_kernel_driver: bool,
}

#[derive(Clone, Copy, Debug, Args)]
pub struct OpenPairArgs {
    #[command(flatten)]
    pub usb: UsbTargetArgs,

    /// Detected NCM pair index to use.
    #[arg(long, default_value_t = 0)]
    pub pair_index: usize,

    /// Detach any bound kernel drivers before claiming interfaces.
    #[arg(long)]
    pub detach_kernel_driver: bool,
}

#[derive(Clone, Debug, Args)]
pub struct TapArgs {
    /// TAP interface name.
    #[arg(long, alias = "tap-name", default_value = "dgxusb0")]
    pub name: String,

    /// TAP MTU.
    #[arg(long, default_value_t = 1500)]
    pub mtu: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usb_id_parses_usb_hex_spelling() {
        assert_eq!("05ac".parse::<UsbId>().unwrap().get(), 0x05ac);
        assert_eq!("0x1905".parse::<UsbId>().unwrap().get(), 0x1905);
    }

    #[test]
    fn bridge_accepts_legacy_tap_name_alias() {
        let cli = Cli::try_parse_from(["dgxusbd", "bridge", "--tap-name", "labtap0"]).unwrap();

        match cli.command {
            Command::Bridge { tap, .. } => assert_eq!(tap.name, "labtap0"),
            _ => panic!("expected bridge command"),
        }
    }
}
