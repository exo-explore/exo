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

        /// Per USB bulk OUT write timeout in milliseconds.
        #[arg(long, alias = "usb-timeout-ms", default_value_t = 100)]
        usb_write_timeout_ms: u64,

        /// USB bulk IN wait timeout in milliseconds for the USB-to-TAP worker.
        #[arg(long, default_value_t = 100)]
        usb_read_timeout_ms: u64,

        /// Maximum TAP frames to collect for one readiness drain and NTB batch.
        #[arg(long, default_value_t = 32)]
        tap_budget_frames: usize,

        /// Maximum completed USB NTBs to drain before checking stop/error state.
        #[arg(long, default_value_t = 8)]
        usb_budget_ntbs: usize,

        /// Number of USB bulk IN transfers to keep queued.
        #[arg(long, default_value_t = crate::bridge::DEFAULT_BRIDGE_USB_READ_QUEUE_DEPTH)]
        usb_read_queue_depth: usize,

        /// Number of USB bulk OUT transfers to keep queued.
        #[arg(long, default_value_t = crate::bridge::DEFAULT_BRIDGE_USB_WRITE_QUEUE_DEPTH)]
        usb_write_queue_depth: usize,
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

    #[test]
    fn bridge_accepts_legacy_usb_timeout_alias_as_write_timeout() {
        let cli = Cli::try_parse_from(["dgxusbd", "bridge", "--usb-timeout-ms", "77"]).unwrap();

        match cli.command {
            Command::Bridge {
                usb_write_timeout_ms,
                ..
            } => assert_eq!(usb_write_timeout_ms, 77),
            _ => panic!("expected bridge command"),
        }
    }
}
