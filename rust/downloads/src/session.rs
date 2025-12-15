//! Torrent session management using rqbit
//!
//! Provides a wrapper around rqbit's Session for managing torrent downloads
//! with persistent seeding and selective file downloads.

use anyhow::{Context, Result};
use librqbit::{AddTorrent, AddTorrentOptions, Api, ManagedTorrentHandle, Session, SessionOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Download progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub download_speed: f64,
    pub upload_speed: f64,
    pub peers_connected: usize,
    pub is_finished: bool,
}

/// Torrent session handle for managing multiple torrents
pub struct TorrentSession {
    session: Arc<Session>,
    api: Arc<Api>,
    session_dir: PathBuf,
    torrents: Arc<RwLock<HashMap<String, ManagedTorrentHandle>>>,
}

impl TorrentSession {
    /// Create a new torrent session
    ///
    /// # Arguments
    /// * `session_dir` - Directory to store session state and downloaded files
    pub async fn new(session_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&session_dir).context("Failed to create session directory")?;

        let opts = SessionOptions {
            disable_dht: false,
            disable_dht_persistence: false,
            dht_config: None,
            persistence: true,
            fastresume: true,
            ..Default::default()
        };

        let session = Session::new_with_opts(session_dir.clone(), opts)
            .await
            .context("Failed to create rqbit session")?;

        let api = Api::new(Arc::clone(&session), None);

        Ok(Self {
            session: Arc::new(session),
            api: Arc::new(api),
            session_dir,
            torrents: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add a torrent from raw bytes
    ///
    /// # Arguments
    /// * `torrent_data` - Raw .torrent file contents
    /// * `save_path` - Where to save the downloaded files
    /// * `file_indices` - Optional list of file indices to download (None = all files)
    ///
    /// # Returns
    /// Info hash as hex string
    pub async fn add_torrent(
        &self,
        torrent_data: Vec<u8>,
        save_path: PathBuf,
        file_indices: Option<Vec<usize>>,
    ) -> Result<String> {
        let opts = AddTorrentOptions {
            overwrite: false,
            only_files_regex: None,
            only_files: file_indices
                .map(|indices| librqbit::AddTorrentOptions::only_files_from_vec(indices)),
            output_folder: Some(save_path.to_string_lossy().to_string()),
            ..Default::default()
        };

        let add_torrent = AddTorrent::from_bytes(torrent_data);

        let handle = self
            .session
            .add_torrent(add_torrent, Some(opts))
            .await
            .context("Failed to add torrent")?;

        let info_hash = handle.info_hash().as_string();

        self.torrents
            .write()
            .await
            .insert(info_hash.clone(), handle);

        Ok(info_hash)
    }

    /// Get download progress for a torrent
    pub async fn get_progress(&self, info_hash: &str) -> Result<DownloadProgress> {
        let torrents = self.torrents.read().await;
        let handle = torrents.get(info_hash).context("Torrent not found")?;

        let stats = handle.stats();
        let state = handle.state();

        Ok(DownloadProgress {
            downloaded_bytes: stats.downloaded_bytes,
            total_bytes: stats.total_bytes,
            download_speed: stats.download_speed,
            upload_speed: stats.upload_speed,
            peers_connected: stats.peers_connected,
            is_finished: state.is_finished(),
        })
    }

    /// Wait until torrent download is completed
    pub async fn wait_until_completed(&self, info_hash: &str) -> Result<()> {
        let torrents = self.torrents.read().await;
        let handle = torrents.get(info_hash).context("Torrent not found")?;

        handle
            .wait_until_completed()
            .await
            .context("Failed to wait for completion")?;

        Ok(())
    }

    /// Enable seeding for a completed torrent
    ///
    /// Note: rqbit seeds by default after completion, this is a no-op
    /// but kept for API compatibility
    pub async fn enable_seeding(&self, _info_hash: &str) -> Result<()> {
        // rqbit automatically seeds after download completion
        // This is kept for API compatibility
        Ok(())
    }

    /// Remove a torrent from the session
    pub async fn remove_torrent(&self, info_hash: &str) -> Result<()> {
        let mut torrents = self.torrents.write().await;

        if let Some(handle) = torrents.remove(info_hash) {
            drop(handle);
        }

        Ok(())
    }

    /// Get list of all torrent info hashes in the session
    pub async fn list_torrents(&self) -> Vec<String> {
        self.torrents.read().await.keys().cloned().collect()
    }
}
