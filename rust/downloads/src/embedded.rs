//! Embedded torrent file access
//!
//! Provides access to .torrent files embedded in the binary at compile time

use include_dir::{Dir, include_dir};

/// Embedded torrent files directory
static TORRENTS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/torrents");

/// Get an embedded torrent file by model_id and revision
///
/// # Arguments
/// * `model_id` - Model identifier (e.g., "mlx-community/Qwen3-30B-A3B-4bit")
/// * `revision` - Git commit hash
///
/// # Returns
/// The torrent file contents, or None if not found
#[inline]
pub fn get_embedded_torrent(model_id: &str, revision: &str) -> Option<Vec<u8>> {
    let path = format!("{}/{}.torrent", model_id, revision);
    TORRENTS
        .get_file(&path)
        .map(|file| file.contents().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_embedded_torrent() {
        // Test with the Qwen3 torrent we have
        let result = get_embedded_torrent(
            "mlx-community/Qwen3-30B-A3B-4bit",
            "d388dead1515f5e085ef7a0431dd8fadf0886c57",
        );

        assert!(result.is_some(), "Expected to find embedded torrent");
        let torrent_data = result.unwrap();
        assert!(!torrent_data.is_empty(), "Torrent data should not be empty");
    }

    #[test]
    fn test_missing_torrent() {
        let result = get_embedded_torrent("nonexistent/model", "abc123");
        assert!(result.is_none(), "Expected None for missing torrent");
    }
}
