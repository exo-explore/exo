//! Embedded torrent file access
//!
//! Provides access to .torrent files embedded in the binary at compile time.
//! Each model/revision can have multiple torrent variants (e.g., "small", "large").

use include_dir::{Dir, include_dir};

/// Embedded torrent files directory
static TORRENTS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/torrents");

/// Get all embedded torrent variants for a model_id and revision
///
/// # Arguments
/// * `model_id` - Model identifier (e.g., "mlx-community/Qwen3-30B-A3B-4bit")
/// * `revision` - Git commit hash
///
/// # Returns
/// Vec of (variant_name, torrent_data) tuples, e.g., [("small", data), ("large", data)]
/// Returns empty Vec if no torrents found for this model/revision.
#[inline]
pub fn get_embedded_torrents(model_id: &str, revision: &str) -> Vec<(String, Vec<u8>)> {
    let dir_path = format!("{model_id}");

    let Some(model_dir) = TORRENTS.get_dir(&dir_path) else {
        return Vec::new();
    };

    let mut results = Vec::new();
    let prefix = format!("{revision}.");
    let suffix = ".torrent";

    for file in model_dir.files() {
        let Some(name) = file.path().file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        // Match files like "{revision}.small.torrent" or "{revision}.large.torrent"
        if name.starts_with(&prefix) && name.ends_with(suffix) {
            // Extract variant: "{revision}.{variant}.torrent" -> "{variant}"
            let middle = &name[prefix.len()..name.len() - suffix.len()];

            // Skip plain "{revision}.torrent" files (wrong format)
            if middle.is_empty() {
                continue;
            }

            results.push((middle.to_string(), file.contents().to_vec()));
        }
    }

    // Sort by variant name for consistent ordering
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_embedded_torrents() {
        // Test with the Qwen3 torrent we have
        let result = get_embedded_torrents(
            "mlx-community/Qwen3-30B-A3B-4bit",
            "d388dead1515f5e085ef7a0431dd8fadf0886c57",
        );

        assert!(!result.is_empty(), "Expected to find embedded torrents");

        // Should have both small and large variants
        let variants: Vec<&str> = result.iter().map(|(v, _)| v.as_str()).collect();
        assert!(
            variants.contains(&"small"),
            "Expected 'small' variant, got: {variants:?}"
        );
        assert!(
            variants.contains(&"large"),
            "Expected 'large' variant, got: {variants:?}"
        );

        // Verify data is not empty
        for (variant, data) in &result {
            assert!(!data.is_empty(), "Torrent data for '{variant}' should not be empty");
        }
    }

    #[test]
    fn test_missing_torrent() {
        let result = get_embedded_torrents("nonexistent/model", "abc123");
        assert!(result.is_empty(), "Expected empty Vec for missing torrent");
    }

    #[test]
    fn test_variant_ordering() {
        let result = get_embedded_torrents(
            "mlx-community/Qwen3-30B-A3B-4bit",
            "d388dead1515f5e085ef7a0431dd8fadf0886c57",
        );

        if result.len() >= 2 {
            // Verify alphabetical ordering
            let variants: Vec<&str> = result.iter().map(|(v, _)| v.as_str()).collect();
            let mut sorted = variants.clone();
            sorted.sort();
            assert_eq!(variants, sorted, "Variants should be sorted alphabetically");
        }
    }
}
