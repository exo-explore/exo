#[cfg(not(target_os = "linux"))]
compile_error!("dgxusbd is linux-only");

pub mod cli;
pub mod usb;

use color_eyre::eyre;

use crate::cli::{Cli, Command};
use crate::usb::{ProbeOptions, list_devices, probe, render_device_list, render_probe_report};

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
    }

    Ok(())
}
