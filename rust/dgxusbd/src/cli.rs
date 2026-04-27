use clap::{Parser, Subcommand};
use color_eyre::eyre;

use crate::usb::UsbSelector;

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
        /// Show every USB device instead of only the target Apple Mac device.
        #[arg(long)]
        all: bool,

        /// USB vendor ID to match. USB IDs are parsed as hex first, so 05ac works.
        #[arg(long, value_parser = parse_u16, default_value = "05ac")]
        vendor_id: u16,

        /// USB product ID to match. USB IDs are parsed as hex first, so 1905 works.
        #[arg(long, value_parser = parse_u16, default_value = "1905")]
        product_id: u16,
    },

    /// Open and inspect a candidate CDC-NCM USB device.
    Probe {
        /// USB vendor ID to match. USB IDs are parsed as hex first, so 05ac works.
        #[arg(long, value_parser = parse_u16, default_value = "05ac")]
        vendor_id: u16,

        /// USB product ID to match. USB IDs are parsed as hex first, so 1905 works.
        #[arg(long, value_parser = parse_u16, default_value = "1905")]
        product_id: u16,

        /// Claim detected CDC-NCM control/data interfaces, then release on exit.
        #[arg(long)]
        claim: bool,

        /// Detach any bound kernel drivers before claiming interfaces.
        #[arg(long, requires = "claim")]
        detach_kernel_driver: bool,
    },

    /// Claim the selected pair, select data altsetting, open endpoints, and do one timed read.
    UsbSmoke {
        /// USB vendor ID to match. USB IDs are parsed as hex first, so 05ac works.
        #[arg(long, value_parser = parse_u16, default_value = "05ac")]
        vendor_id: u16,

        /// USB product ID to match. USB IDs are parsed as hex first, so 1905 works.
        #[arg(long, value_parser = parse_u16, default_value = "1905")]
        product_id: u16,

        /// Detected NCM pair index to use.
        #[arg(long, default_value_t = 0)]
        pair_index: usize,

        /// Detach any bound kernel drivers before claiming interfaces.
        #[arg(long)]
        detach_kernel_driver: bool,

        /// Skip CDC-NCM class setup requests and only claim/select/open.
        #[arg(long)]
        skip_ncm_init: bool,

        /// Timed bulk IN read timeout in milliseconds.
        #[arg(long, default_value_t = 250)]
        read_timeout_ms: u64,
    },

    /// Create a Linux TAP device and exit after optional hold time.
    TapSmoke {
        /// TAP interface name.
        #[arg(long, default_value = "dgxusb0")]
        name: String,

        /// TAP MTU.
        #[arg(long, default_value_t = 1500)]
        mtu: u16,

        /// Keep the TAP open for this many seconds before exiting.
        #[arg(long, default_value_t = 0)]
        hold_seconds: u64,
    },

    /// Run a conservative one-pair TAP-to-CDC-NCM bridge.
    Bridge {
        /// USB vendor ID to match. USB IDs are parsed as hex first, so 05ac works.
        #[arg(long, value_parser = parse_u16, default_value = "05ac")]
        vendor_id: u16,

        /// USB product ID to match. USB IDs are parsed as hex first, so 1905 works.
        #[arg(long, value_parser = parse_u16, default_value = "1905")]
        product_id: u16,

        /// Detected NCM pair index to use.
        #[arg(long, default_value_t = 0)]
        pair_index: usize,

        /// Detach any bound kernel drivers before claiming interfaces.
        #[arg(long)]
        detach_kernel_driver: bool,

        /// TAP interface name.
        #[arg(long, default_value = "dgxusb0")]
        tap_name: String,

        /// TAP MTU.
        #[arg(long, default_value_t = 1500)]
        mtu: u16,

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

impl Command {
    #[must_use]
    pub const fn selector(&self) -> UsbSelector {
        match self {
            Self::List {
                vendor_id,
                product_id,
                ..
            }
            | Self::Probe {
                vendor_id,
                product_id,
                ..
            }
            | Self::UsbSmoke {
                vendor_id,
                product_id,
                ..
            }
            | Self::Bridge {
                vendor_id,
                product_id,
                ..
            } => UsbSelector {
                vendor_id: *vendor_id,
                product_id: *product_id,
            },
            Self::TapSmoke { .. } => UsbSelector {
                vendor_id: crate::usb::APPLE_VENDOR_ID,
                product_id: crate::usb::APPLE_MAC_PRODUCT_ID,
            },
        }
    }
}

fn parse_u16(value: &str) -> eyre::Result<u16> {
    let trimmed = value.trim();
    let without_prefix = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);

    u16::from_str_radix(without_prefix, 16)
        .or_else(|_| trimmed.parse::<u16>())
        .map_err(Into::into)
}
