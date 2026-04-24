//! Persistent node identity for `babblerd`.
//!
//! The node ID occupies the full low 64 bits of the EXO ULA space.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;

use color_eyre::eyre::{self, WrapErr, eyre};
use ipnet::Ipv6Net;
use nix::unistd::geteuid;
use std::net::Ipv6Addr;

pub fn load_or_create_node_id(path: &Path) -> eyre::Result<u64> {
    match read_node_id(path) {
        Ok(node_id) => Ok(node_id),
        Err(err) if is_not_found(&err) => create_node_id(path),
        Err(err) => Err(err),
    }
}

pub fn node_addr(prefix: Ipv6Net, node_id: u64) -> eyre::Result<Ipv6Net> {
    if prefix.prefix_len() != 64 {
        return Err(eyre!(
            "expected EXO ULA prefix to be /64, got {prefix} with /{}",
            prefix.prefix_len()
        ));
    }
    Ok(Ipv6Net::new_assert(
        Ipv6Addr::from_bits(prefix.trunc().addr().to_bits() | u128::from(node_id)),
        128,
    ))
}

fn create_node_id(path: &Path) -> eyre::Result<u64> {
    let Some(parent) = path.parent() else {
        return Err(eyre!(
            "node id file has no parent directory: {}",
            path.display()
        ));
    };
    fs::create_dir_all(parent)
        .wrap_err_with(|| format!("creating node id directory {}", parent.display()))?;

    let node_id = generate_node_id();
    let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            return read_node_id(path);
        }
        Err(err) => {
            return Err(err).wrap_err_with(|| format!("creating node id file {}", path.display()));
        }
    };

    file.set_permissions(fs::Permissions::from_mode(0o600))
        .wrap_err_with(|| format!("setting permissions on {}", path.display()))?;

    writeln!(file, "{node_id:016x}")
        .wrap_err_with(|| format!("writing node id file {}", path.display()))?;
    file.sync_all()
        .wrap_err_with(|| format!("syncing node id file {}", path.display()))?;
    drop(file);

    read_node_id(path)
}

fn read_node_id(path: &Path) -> eyre::Result<u64> {
    let metadata =
        fs::metadata(path).wrap_err_with(|| format!("reading metadata for {}", path.display()))?;
    ensure_owner(path, &metadata)?;

    let raw = fs::read_to_string(path)
        .wrap_err_with(|| format!("reading node id file {}", path.display()))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(eyre!("node id file is empty: {}", path.display()));
    }

    let node_id = u64::from_str_radix(trimmed.trim_start_matches("0x"), 16)
        .wrap_err_with(|| format!("invalid node id in {}: {:?}", path.display(), trimmed))?;
    Ok(node_id)
}

fn ensure_owner(path: &Path, metadata: &fs::Metadata) -> eyre::Result<()> {
    let expected_uid = geteuid().as_raw();
    let actual_uid = metadata.uid();
    if actual_uid != expected_uid {
        return Err(eyre!(
            "node id file {} is owned by uid {}, expected {}",
            path.display(),
            actual_uid,
            expected_uid
        ));
    }
    Ok(())
}

fn generate_node_id() -> u64 {
    rand::random::<u64>()
}

fn is_not_found(err: &eyre::Report) -> bool {
    err.downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nonce = rand::random::<u64>();
        std::env::temp_dir().join(format!(
            "babblerd-identity-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn creates_and_reloads_same_node_id() {
        let dir = temp_path("create");
        let path = dir.join("node-id");

        let first = load_or_create_node_id(&path).expect("create node id");
        let second = load_or_create_node_id(&path).expect("reload node id");

        assert_eq!(first, second);

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn node_addr_uses_full_low_64_bits() {
        let addr = node_addr(
            Ipv6Net::new_assert(
                Ipv6Addr::new(0xfde0, 0x20c6, 0x1fa7, 0xffff, 0, 0, 0, 0),
                64,
            ),
            0x1234_5678_9abc_def0,
        )
        .expect("node address should be constructed");

        assert_eq!(addr.prefix_len(), 128);
        assert_eq!(
            addr.addr(),
            Ipv6Addr::new(
                0xfde0, 0x20c6, 0x1fa7, 0xffff, 0x1234, 0x5678, 0x9abc, 0xdef0,
            )
        );
    }
}
