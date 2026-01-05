//! Downloads module - BitTorrent downloads PyO3 bindings

use crate::ext::*;
use downloads::bencode::AnnounceParams;
use downloads::tracker::{PeerInfo, TopologyData, handle_announce as rust_handle_announce};
use downloads::{DownloadProgress, TorrentSession};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Handle a tracker announce request
///
/// Args:
///     params: Dictionary with announce parameters (info_hash, peer_id, port, etc.)
///     peers: List of peer dictionaries (node_id, ip, port, has_complete, priority)
///
/// Returns:
///     Bencoded announce response as bytes
#[pyfunction]
fn handle_tracker_announce(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    peers: &Bound<'_, pyo3::types::PyList>,
) -> PyResult<Py<PyBytes>> {
    // Parse announce params
    let info_hash = {
        let info_hash_item = params
            .get_item("info_hash")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing info_hash"))?;
        let info_hash_bytes: &[u8] = info_hash_item.extract()?;

        if info_hash_bytes.len() != 20 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "info_hash must be 20 bytes",
            ));
        }

        let mut info_hash = [0u8; 20];
        info_hash.copy_from_slice(info_hash_bytes);
        info_hash
    };

    let peer_id = {
        let peer_id_item = params
            .get_item("peer_id")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing peer_id"))?;
        let peer_id_bytes: &[u8] = peer_id_item.extract()?;

        if peer_id_bytes.len() != 20 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "peer_id must be 20 bytes",
            ));
        }

        let mut peer_id = [0u8; 20];
        peer_id.copy_from_slice(peer_id_bytes);
        peer_id
    };

    let port: u16 = params
        .get_item("port")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing port"))?
        .extract()?;

    let uploaded: u64 = params
        .get_item("uploaded")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing uploaded"))?
        .extract()?;

    let downloaded: u64 = params
        .get_item("downloaded")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing downloaded"))?
        .extract()?;

    let left: u64 = params
        .get_item("left")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing left"))?
        .extract()?;

    let compact: bool = params
        .get_item("compact")?
        .map(|v| v.extract().unwrap_or(true))
        .unwrap_or(true);

    let announce_params = AnnounceParams {
        info_hash,
        peer_id,
        port,
        uploaded,
        downloaded,
        left,
        compact,
        event: None, // TODO: parse event if needed
    };

    // Parse peer list
    let peer_infos: Result<Vec<PeerInfo>, PyErr> = peers
        .iter()
        .map(|peer_item| {
            let peer_dict: &Bound<'_, PyDict> = peer_item.downcast()?;
            let node_id: String = peer_dict
                .get_item("node_id")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing node_id"))?
                .extract()?;

            let ip_str: String = peer_dict
                .get_item("ip")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing ip"))?
                .extract()?;

            let ip: Ipv4Addr = ip_str
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid IP address"))?;

            let port: u16 = peer_dict
                .get_item("port")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing port"))?
                .extract()?;

            let has_complete: bool = peer_dict
                .get_item("has_complete")?
                .map(|v: Bound<'_, pyo3::PyAny>| v.extract().unwrap_or(false))
                .unwrap_or(false);

            let priority: i32 = peer_dict
                .get_item("priority")?
                .map(|v: Bound<'_, pyo3::PyAny>| v.extract().unwrap_or(0))
                .unwrap_or(0);

            Ok(PeerInfo {
                node_id,
                ip,
                port,
                has_complete,
                priority,
            })
        })
        .collect();

    let peer_infos = peer_infos?;

    let topology = TopologyData { peers: peer_infos };

    // Call Rust tracker handler
    let response_bytes = rust_handle_announce(&announce_params, &topology).pyerr()?;

    // Return as Python bytes
    Ok(PyBytes::new(py, &response_bytes).unbind())
}

/// Get an embedded torrent file
///
/// Args:
///     model_id: Model identifier (e.g., "mlx-community/Qwen3-30B-A3B-4bit")
///     revision: Git commit hash
///
/// Returns:
///     Torrent file contents as bytes, or None if not found
#[pyfunction]
fn get_embedded_torrent(
    py: Python<'_>,
    model_id: String,
    revision: String,
) -> PyResult<Option<Py<PyBytes>>> {
    match downloads::get_embedded_torrent(&model_id, &revision) {
        Some(data) => Ok(Some(PyBytes::new(py, &data).unbind())),
        None => Ok(None),
    }
}

/// Python wrapper for TorrentSession
#[pyclass]
struct TorrentSessionHandle {
    session: Arc<Mutex<TorrentSession>>,
}

#[pymethods]
impl TorrentSessionHandle {
    /// Create a new torrent session
    ///
    /// Args:
    ///     session_dir: Directory to store session state and downloads
    #[new]
    fn new(session_dir: String) -> PyResult<Self> {
        let session_path = PathBuf::from(session_dir);

        let session = tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async { TorrentSession::new(session_path).await })
            .pyerr()?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
        })
    }

    /// Add a torrent from bytes
    ///
    /// Args:
    ///     torrent_data: Raw .torrent file contents
    ///     save_path: Where to save downloaded files
    ///     file_indices: Optional list of file indices to download
    ///
    /// Returns:
    ///     Info hash as hex string
    fn add_torrent(
        &self,
        _py: Python<'_>,
        torrent_data: Vec<u8>,
        save_path: String,
        file_indices: Option<Vec<usize>>,
    ) -> PyResult<String> {
        let session = Arc::clone(&self.session);
        let save_path = PathBuf::from(save_path);

        tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async {
                session
                    .lock()
                    .await
                    .add_torrent(torrent_data, save_path, file_indices)
                    .await
            })
            .pyerr()
    }

    /// Get download progress for a torrent
    ///
    /// Args:
    ///     info_hash: Torrent info hash
    ///
    /// Returns:
    ///     Dictionary with progress information
    fn get_progress(&self, py: Python<'_>, info_hash: String) -> PyResult<Py<PyDict>> {
        let session = Arc::clone(&self.session);

        let progress: DownloadProgress = tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async { session.lock().await.get_progress(&info_hash).await })
            .pyerr()?;

        let dict = PyDict::new(py);
        dict.set_item("downloaded_bytes", progress.downloaded_bytes)?;
        dict.set_item("total_bytes", progress.total_bytes)?;
        dict.set_item("download_speed", progress.download_speed)?;
        dict.set_item("upload_speed", progress.upload_speed)?;
        dict.set_item("peers_connected", progress.peers_connected)?;
        dict.set_item("is_finished", progress.is_finished)?;

        Ok(dict.unbind())
    }

    /// Wait until torrent download is completed
    ///
    /// Args:
    ///     info_hash: Torrent info hash
    fn wait_until_completed(&self, _py: Python<'_>, info_hash: String) -> PyResult<()> {
        let session = Arc::clone(&self.session);

        tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async { session.lock().await.wait_until_completed(&info_hash).await })
            .pyerr()
    }

    /// Enable seeding for a torrent
    ///
    /// Args:
    ///     info_hash: Torrent info hash
    fn enable_seeding(&self, _py: Python<'_>, info_hash: String) -> PyResult<()> {
        let session = Arc::clone(&self.session);

        tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async { session.lock().await.enable_seeding(&info_hash).await })
            .pyerr()
    }

    /// Remove a torrent from the session
    ///
    /// Args:
    ///     info_hash: Torrent info hash
    fn remove_torrent(&self, _py: Python<'_>, info_hash: String) -> PyResult<()> {
        let session = Arc::clone(&self.session);

        tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async { session.lock().await.remove_torrent(&info_hash).await })
            .pyerr()
    }

    /// List all torrents in the session
    ///
    /// Returns:
    ///     List of info hashes
    fn list_torrents(&self, _py: Python<'_>) -> PyResult<Vec<String>> {
        let session = Arc::clone(&self.session);

        tokio::runtime::Runtime::new()
            .pyerr()?
            .block_on(async { Ok(session.lock().await.list_torrents().await) })
    }
}

/// Downloads submodule
pub(crate) fn downloads_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(handle_tracker_announce, m)?)?;
    m.add_function(wrap_pyfunction!(get_embedded_torrent, m)?)?;
    m.add_class::<TorrentSessionHandle>()?;
    Ok(())
}
