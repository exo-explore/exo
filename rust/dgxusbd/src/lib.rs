#[cfg(not(target_os = "linux"))]
compile_error!("dgxusbd is linux-only");

pub mod bridge;
pub mod cli;
pub mod dataplane;
pub mod ncm;
pub mod tap;
pub mod usb;

use std::thread;
use std::time::Duration;

use color_eyre::eyre;
use extend::ext;

use crate::bridge::{BridgeOptions, render_bridge_report, run_bridge};
use crate::cli::{Cli, Command};
use crate::tap::{TapOptions, create_tap, render_tap_smoke};
use crate::usb::{
    OpenPairOptions, ProbeOptions, UsbSmokeOptions, list_devices, probe, render_device_list,
    render_probe_report, render_usb_smoke_report, smoke_usb_data_path,
};

/// Run the selected `dgxusbd` command.
///
/// # Errors
///
/// Returns an error when USB discovery, device probing, or interface claiming fails.
pub fn run(cli: Cli) -> eyre::Result<()> {
    let command = cli.command;

    match command {
        Command::List { usb, all } => {
            let devices = list_devices(usb.selector(), all)?;
            print!("{}", render_device_list(&devices));
        }
        Command::Probe { usb, claim } => {
            let report = probe(ProbeOptions {
                selector: usb.selector(),
                claim: claim.claim,
                detach_kernel_driver: claim.detach_kernel_driver,
            })?;
            print!("{}", render_probe_report(&report));
        }
        Command::UsbSmoke {
            open,
            skip_ncm_init,
            read_timeout_ms,
        } => {
            let report = smoke_usb_data_path(UsbSmokeOptions {
                open: OpenPairOptions {
                    selector: open.usb.selector(),
                    pair_index: open.pair_index,
                    detach_kernel_driver: open.detach_kernel_driver,
                    initialize_ncm: !skip_ncm_init,
                    require_ncm_setup_success: false,
                    ntb_input_size: crate::ncm::DEFAULT_NTB_MAX_SIZE_U32,
                    max_datagram_size: crate::tap::DEFAULT_TAP_MTU.ethernet_datagram_size(),
                },
                read_timeout: read_timeout_ms.milliseconds(),
            })?;
            print!("{}", render_usb_smoke_report(&report));
        }
        Command::TapSmoke { tap, hold_seconds } => {
            let tap = create_tap(&TapOptions {
                name: tap.name,
                mtu: tap.mtu,
                nonblocking: false,
            })?;
            print!("{}", render_tap_smoke(&tap));
            if hold_seconds > 0 {
                thread::sleep(hold_seconds.seconds());
            }
        }
        Command::Bridge {
            open,
            tap,
            duration_seconds,
            max_events,
            usb_write_timeout_ms,
            usb_read_timeout_ms,
            usb_read_queue_depth,
            tap_budget_frames,
            usb_budget_ntbs,
        } => {
            let report = run_bridge(BridgeOptions {
                open: OpenPairOptions {
                    selector: open.usb.selector(),
                    pair_index: open.pair_index,
                    detach_kernel_driver: open.detach_kernel_driver,
                    initialize_ncm: true,
                    require_ncm_setup_success: true,
                    ntb_input_size: crate::ncm::DEFAULT_NTB_MAX_SIZE_U32,
                    max_datagram_size: tap.mtu.ethernet_datagram_size(),
                },
                tap: TapOptions {
                    name: tap.name,
                    mtu: tap.mtu,
                    nonblocking: true,
                },
                duration: duration_seconds.map(DurationArgExt::seconds),
                max_events,
                usb_read_timeout: usb_read_timeout_ms.milliseconds(),
                usb_write_timeout: usb_write_timeout_ms.milliseconds(),
                tap_budget_frames,
                usb_budget_ntbs,
                usb_read_queue_depth,
            })?;
            print!("{}", render_bridge_report(&report));
        }
    }

    Ok(())
}

#[ext(pub(crate), name = DurationArgExt)]
impl u64 {
    fn milliseconds(self) -> Duration {
        Duration::from_millis(self)
    }

    fn seconds(self) -> Duration {
        Duration::from_secs(self)
    }
}

#[ext(pub(crate), name = MtuExt)]
impl u16 {
    fn ethernet_datagram_size(self) -> u16 {
        self.saturating_add(crate::ncm::ETHERNET_HEADER_LEN_U16)
    }
}
