//! Child-process lifecycle for the managed `babeld` instance.
//!
//! This module owns:
//!
//! - spawn-time configuration of the `babeld` subprocess
//! - the private control socket location used for the local Babel session
//! - shutdown/cleanup of the child and its socket
//!
//! It intentionally does **not** own the live socket session logic. Reading and writing the
//! local Babel protocol belongs in the session layer.

use ipnet::Ipv6Net;
use std::fs::Permissions;
use std::io;
use std::os::unix::fs::PermissionsExt;
use tokio::process::Command;
use tokio::time::Duration;

use crate::{BabbleError, Result};
use futures_lite::FutureExt;

#[cfg(target_os = "macos")]
const PRIVATE_SOCK_PATH: &str = "/var/run/babbler/private/babeld.sock";
#[cfg(target_os = "linux")]
const PRIVATE_SOCK_PATH: &str = "/run/babbler/private/babeld.sock";
#[cfg(target_os = "macos")]
const PRIVATE_DIR: &str = "/var/run/babbler/private";
#[cfg(target_os = "linux")]
const PRIVATE_DIR: &str = "/run/babbler/private";

#[must_use]
pub(crate) fn private_sock_path() -> &'static str {
    PRIVATE_SOCK_PATH
}

pub(crate) struct BabeldProcess {
    proc: tokio::process::Child,
}

impl Drop for BabeldProcess {
    #[inline]
    fn drop(&mut self) {
        // Emergency SIGKILL to avoid leaking an unmanaged babeld subprocess.
        match self.proc.try_wait() {
            Ok(None) => {}
            Ok(Some(sc)) => {
                if !sc.success() {
                    _ = self.proc.start_kill();
                }
            }
            _ => {
                _ = self.proc.start_kill();
            }
        }
    }
}

impl BabeldProcess {
    #[tracing::instrument]
    pub(crate) async fn spawn(advertised: Ipv6Net, iface: String) -> Result<Self> {
        tokio::fs::create_dir_all(PRIVATE_DIR).await?;
        // TODO: remove this magic constant (and magic constants in general)
        tokio::fs::set_permissions(PRIVATE_DIR, Permissions::from_mode(0o0700)).await?;
        tracing::info!("spawning babeld socket in {PRIVATE_SOCK_PATH}");
        let res = match Command::new("babeld")
            .arg("-G")
            .arg(PRIVATE_SOCK_PATH)
            .arg("-I")
            .arg(format!("{PRIVATE_DIR}/babeld.pid"))
            .arg("-C")
            .arg("kernel-install false")
            .arg("-C")
            .arg(format!("redistribute local ip {advertised}"))
            .arg("-C")
            .arg("redistribute local deny")
            .arg(iface)
            .spawn()
        {
            Ok(proc) => Ok(Self { proc }),
            Err(e) => {
                tracing::warn!(error=%e, "failed to spawn babeld");
                Err(e.into())
            }
        };

        // maybe spinning logic is fine with magic numbers...?
        tokio::time::sleep(Duration::from_millis(10)).await;
        while !matches!(tokio::fs::try_exists(PRIVATE_SOCK_PATH).await, Ok(true)) {
            tracing::info!("where is the sock");
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        res
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        let kill_res = if let Some(pid) = self.proc.id() {
            let pid: libc::pid_t = pid.try_into().expect("pid overflow");
            // SAFETY: pid >= 0, freshly checked.
            let rc = unsafe { libc::kill(pid, libc::SIGINT) };
            let rc_err = if rc != 0 && rc != libc::ESRCH {
                Err(io::Error::last_os_error().into())
            } else {
                Ok(())
            };
            let exit_code = async { Some(self.proc.wait().await) }
                .or(async {
                    // TODO: undocumented magic number
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    None
                })
                .await;
            match exit_code {
                Some(Ok(code)) => {
                    if code.success() {
                        rc_err
                    } else {
                        rc_err.and_then(|()| Err(BabbleError::BabeldCrashed(code.code())))
                    }
                }
                Some(Err(e)) => Err(e.into()),
                None => {
                    self.proc.kill().await?;
                    rc_err.and(Err(BabbleError::BabeldCrashed(None)))
                }
            }
        } else {
            Ok(())
        };
        let rem_res = match std::fs::remove_file(PRIVATE_SOCK_PATH) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        };
        kill_res.and(rem_res)
    }
}
