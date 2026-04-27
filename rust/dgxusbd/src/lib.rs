#[cfg(not(target_os = "linux"))]
compile_error!("dgxusbd is linux-only");

pub mod bridge;
pub mod cli;
pub mod ncm;
pub mod tap;
pub mod usb;

use std::thread;
use std::time::Duration;

use color_eyre::eyre;

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
    let selector = command.selector();

    match command {
        Command::List { all, .. } => {
            let devices = list_devices(selector, all)?;
            print!("{}", render_device_list(&devices));
        }
        Command::Probe {
            claim,
            detach_kernel_driver,
            ..
        } => {
            let report = probe(ProbeOptions {
                selector,
                claim,
                detach_kernel_driver,
            })?;
            print!("{}", render_probe_report(&report));
        }
        Command::UsbSmoke {
            pair_index,
            detach_kernel_driver,
            skip_ncm_init,
            read_timeout_ms,
            ..
        } => {
            let report = smoke_usb_data_path(UsbSmokeOptions {
                open: OpenPairOptions {
                    selector,
                    pair_index,
                    detach_kernel_driver,
                    initialize_ncm: !skip_ncm_init,
                    ntb_input_size: default_ntb_input_size(),
                    max_datagram_size: ethernet_max_datagram_size(crate::tap::DEFAULT_TAP_MTU),
                },
                read_timeout: Duration::from_millis(read_timeout_ms),
            })?;
            print!("{}", render_usb_smoke_report(&report));
        }
        Command::TapSmoke {
            name,
            mtu,
            hold_seconds,
        } => {
            let tap = create_tap(&TapOptions {
                name,
                mtu,
                nonblocking: false,
            })?;
            print!("{}", render_tap_smoke(&tap));
            if hold_seconds > 0 {
                thread::sleep(Duration::from_secs(hold_seconds));
            }
        }
        Command::Bridge {
            pair_index,
            detach_kernel_driver,
            tap_name,
            mtu,
            duration_seconds,
            max_events,
            usb_timeout_ms,
            ..
        } => {
            let report = run_bridge(BridgeOptions {
                open: OpenPairOptions {
                    selector,
                    pair_index,
                    detach_kernel_driver,
                    initialize_ncm: true,
                    ntb_input_size: default_ntb_input_size(),
                    max_datagram_size: ethernet_max_datagram_size(mtu),
                },
                tap: TapOptions {
                    name: tap_name,
                    mtu,
                    nonblocking: true,
                },
                duration: duration_seconds.map(Duration::from_secs),
                max_events,
                usb_timeout: Duration::from_millis(usb_timeout_ms),
            })?;
            print!("{}", render_bridge_report(&report));
        }
    }

    Ok(())
}

fn default_ntb_input_size() -> u32 {
    u32::try_from(crate::ncm::DEFAULT_NTB_MAX_SIZE).expect("default NTB size fits u32")
}

fn ethernet_max_datagram_size(mtu: u16) -> u16 {
    mtu.saturating_add(
        u16::try_from(crate::ncm::ETHERNET_HEADER_LEN).expect("Ethernet header fits u16"),
    )
}
