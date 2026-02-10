#!/usr/bin/env python3
"""Unit tests for vision KV-cache media region validation.

Tests KVPrefixCache._validate_media_match — the logic that detects when a
cached KV prefix includes vision features from a different image than the
new query, and truncates the prefix hit accordingly.

Run:
    python -m pytest tests/test_vision_cache.py -v
"""

import pytest

from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.vision import MediaRegion

validate = KVPrefixCache._validate_media_match


class TestValidateMediaMatch:
    """Tests for _validate_media_match static method."""

    def test_text_only_no_truncation(self):
        """No media in cache or query — full prefix reuse."""
        assert validate(8000, [], []) == 8000

    def test_text_prefix_before_image(self):
        """Match stops before cached image region — no truncation needed."""
        cached = [MediaRegion("hashA", 5000, 8600)]
        assert validate(5000, cached, []) == 5000

    def test_same_image_same_position(self):
        """Identical image at same position — full match preserved."""
        cached = [MediaRegion("hashA", 5000, 8600)]
        query = [MediaRegion("hashA", 5000, 8600)]
        assert validate(9000, cached, query) == 9000

    def test_different_image_truncates(self):
        """Different image content at same position — truncate to region start."""
        cached = [MediaRegion("hashA", 5000, 8600)]
        query = [MediaRegion("hashB", 5000, 8600)]
        assert validate(9000, cached, query) == 5000

    def test_match_below_region_start(self):
        """Match doesn't reach the image region — no issue."""
        cached = [MediaRegion("hashA", 5000, 8600)]
        query = [MediaRegion("hashB", 5000, 8600)]
        assert validate(4000, cached, query) == 4000

    def test_text_followup_no_images_in_query(self):
        """Text-only follow-up with cached images — no truncation
        (image tokens are already baked into the KV cache)."""
        cached = [MediaRegion("hashA", 5000, 8600)]
        assert validate(9000, cached, []) == 9000

    def test_multiple_images_first_mismatch_truncates(self):
        """Two cached images, first matches, second doesn't — truncate at second."""
        cached = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashB", 6000, 8000),
        ]
        query = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashC", 6000, 8000),  # different from cached
        ]
        assert validate(9000, cached, query) == 6000

    def test_multiple_images_all_match(self):
        """Two cached images, both match query — full prefix preserved."""
        cached = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashB", 6000, 8000),
        ]
        query = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashB", 6000, 8000),
        ]
        assert validate(9000, cached, query) == 9000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
