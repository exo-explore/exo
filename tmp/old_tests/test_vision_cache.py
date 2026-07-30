from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.vision import MediaRegion

validate = KVPrefixCache._validate_media_match


class TestValidateMediaMatch:
    def test_text_only_no_truncation(self):
        assert validate(8000, [], []) == 8000

    def test_text_prefix_before_image(self):
        cached = [MediaRegion("hashA", 5000, 8600)]
        assert validate(5000, cached, []) == 5000

    def test_same_image_same_position(self):
        cached = [MediaRegion("hashA", 5000, 8600)]
        query = [MediaRegion("hashA", 5000, 8600)]
        assert validate(9000, cached, query) == 9000

    def test_different_image_truncates(self):
        cached = [MediaRegion("hashA", 5000, 8600)]
        query = [MediaRegion("hashB", 5000, 8600)]
        assert validate(9000, cached, query) == 5000

    def test_match_below_region_start(self):
        cached = [MediaRegion("hashA", 5000, 8600)]
        query = [MediaRegion("hashB", 5000, 8600)]
        assert validate(4000, cached, query) == 4000

    def test_text_followup_no_images_in_query(self):
        cached = [MediaRegion("hashA", 5000, 8600)]
        assert validate(9000, cached, []) == 9000

    def test_multiple_images_first_mismatch_truncates(self):
        cached = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashB", 6000, 8000),
        ]
        query = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashC", 6000, 8000),
        ]
        assert validate(9000, cached, query) == 6000

    def test_multiple_images_all_match(self):
        cached = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashB", 6000, 8000),
        ]
        query = [
            MediaRegion("hashA", 2000, 4000),
            MediaRegion("hashB", 6000, 8000),
        ]
        assert validate(9000, cached, query) == 9000

    def test_no_cached_regions(self):
        query = [MediaRegion("hashA", 100, 200)]
        assert validate(500, [], query) == 500

    def test_cached_region_beyond_match(self):
        cached = [MediaRegion("hashA", 10000, 12000)]
        query = [MediaRegion("hashB", 10000, 12000)]
        assert validate(5000, cached, query) == 5000
