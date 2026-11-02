"""Models on disk: completeness, read-only dirs, choosing a dir, deleting."""

import json
import shutil
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download import store
from exo.download.hub import RepoFile
from exo.download.store import Manifest, NotEnoughSpaceError
from exo.download.tests.harness import DiskUsage
from exo.shared.types.common import ModelId

MODEL_ID = ModelId("test-org/test-model")
NAME = MODEL_ID.normalize()


@pytest.fixture
def dirs(tmp_path: Path) -> Iterator[tuple[Path, Path, Path]]:
    """(default writable dir, second writable dir, read-only dir)"""
    default, second, read_only = (
        tmp_path / "default",
        tmp_path / "second",
        tmp_path / "ro",
    )
    for d in (default, second, read_only):
        d.mkdir()
    with (
        patch.object(store, "EXO_DEFAULT_MODELS_DIR", default),
        patch.object(store, "EXO_MODELS_DIRS", (default, second)),
        patch.object(store, "EXO_MODELS_READ_ONLY_DIRS", (read_only,)),
    ):
        yield default, second, read_only


def legacy_model(model_dir: Path, *, with_weights: bool = True) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    index = {"metadata": {"total_size": 7}, "weight_map": {"w": "model.safetensors"}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (model_dir / "config.json").write_text("{}")
    if with_weights:
        (model_dir / "model.safetensors").write_bytes(b"weights")


def downloaded_model(model_dir: Path, *, complete: bool = True) -> Manifest:
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    model_dir.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        (model_dir / path).write_bytes(content)
    manifest = Manifest(
        repo=str(MODEL_ID),
        commit="c" * 40,
        files=[RepoFile(path=p, size=len(c)) for p, c in files.items()],
        complete=complete,
    )
    store.write_manifest(model_dir, manifest)
    return manifest


class TestCompleteness:
    def test_downloaded_model_is_complete(self, dirs: tuple[Path, Path, Path]) -> None:
        downloaded_model(dirs[0] / NAME)
        assert store.find_complete(MODEL_ID) == dirs[0] / NAME

    def test_unfinished_download_is_not_complete(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        downloaded_model(dirs[0] / NAME, complete=False)
        assert store.find_complete(MODEL_ID) is None

    def test_a_removed_file_makes_it_incomplete(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        downloaded_model(dirs[0] / NAME)
        (dirs[0] / NAME / "config.json").unlink()
        assert store.find_complete(MODEL_ID) is None

    def test_a_truncated_file_makes_it_incomplete(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        downloaded_model(dirs[0] / NAME)
        (dirs[0] / NAME / "model.safetensors").write_bytes(b"w")
        assert store.find_complete(MODEL_ID) is None

    def test_legacy_model_with_its_weights_is_complete(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        legacy_model(dirs[1] / NAME)
        assert store.find_complete(MODEL_ID) == dirs[1] / NAME

    def test_legacy_model_missing_weights_is_not(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        legacy_model(dirs[0] / NAME, with_weights=False)
        assert store.find_complete(MODEL_ID) is None

    def test_legacy_model_with_partial_files_is_not(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        legacy_model(dirs[0] / NAME)
        (dirs[0] / NAME / "tokenizer.json.partial").write_bytes(b"tok")
        assert store.find_complete(MODEL_ID) is None

    def test_read_only_copy_comes_first(self, dirs: tuple[Path, Path, Path]) -> None:
        default, _, read_only = dirs
        legacy_model(default / NAME)
        legacy_model(read_only / NAME)
        assert store.find_complete(MODEL_ID) == read_only / NAME
        assert store.is_read_only(read_only / NAME)
        assert not store.is_read_only(default / NAME)


class TestChoosingADir:
    files = [RepoFile(path="model.safetensors", size=1000)]

    def free(self, free_by_dir: dict[Path, int]) -> AbstractContextManager[object]:
        real = shutil.disk_usage

        def disk_usage(path: str | Path) -> object:
            for directory, free in free_by_dir.items():
                if Path(path).is_relative_to(directory):
                    if free < 0:
                        raise OSError("not mounted")
                    usage = real(path)
                    return DiskUsage(usage.total, usage.total - free, free)
            return real(path)

        return patch("shutil.disk_usage", disk_usage)

    def test_first_dir_with_room(self, dirs: tuple[Path, Path, Path]) -> None:
        default, second, _ = dirs
        with self.free({default: 10, second: 10_000}):
            assert store.choose_root(MODEL_ID, self.files, {}) == (second, 1000)

    def test_unreadable_dir_is_skipped(self, dirs: tuple[Path, Path, Path]) -> None:
        default, second, _ = dirs
        with self.free({default: -1, second: 10_000}):
            assert store.choose_root(MODEL_ID, self.files, {})[0] == second

    def test_dir_holding_part_of_the_model_comes_first(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        default, second, _ = dirs
        (second / NAME).mkdir()
        (second / NAME / "model.safetensors.partial").write_bytes(b"x" * 400)
        with self.free({default: 10_000, second: 700}):
            assert store.choose_root(MODEL_ID, self.files, {}) == (second, 600)

    def test_other_downloads_reservations_count(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        default, second, _ = dirs
        with self.free({default: 1500, second: 1500}):
            assert store.choose_root(MODEL_ID, self.files, {default: 1000})[0] == second

    def test_says_how_much_is_needed(self, dirs: tuple[Path, Path, Path]) -> None:
        default, second, _ = dirs
        files = [RepoFile(path="model.safetensors", size=12_300_000_000)]
        with (
            self.free({default: 4_100_000_000, second: 100}),
            pytest.raises(NotEnoughSpaceError) as raised,
        ):
            store.choose_root(MODEL_ID, files, {})
        assert raised.value.message == (
            "Not enough disk space for test-org/test-model: needs 12.3 GB, 4.1 GB free."
        )


class TestWantedFiles:
    files = [
        RepoFile(path=p, size=1)
        for p in (
            "config.json",
            "model.safetensors",
            "original/consolidated.pth",
            "metal/model.bin",
            "transformer/diffusion.safetensors",
        )
    ]

    def test_skips_original_and_metal(self) -> None:
        assert [f.path for f in store.wanted_files(self.files)] == [
            "config.json",
            "model.safetensors",
            "transformer/diffusion.safetensors",
        ]

    def test_image_models_skip_root_weights(self) -> None:
        assert [f.path for f in store.wanted_files(self.files, image_model=True)] == [
            "config.json",
            "transformer/diffusion.safetensors",
        ]

    def test_only(self) -> None:
        wanted = store.wanted_files(self.files, only=("*.safetensors", "config.json"))
        assert [f.path for f in wanted] == [
            "config.json",
            "model.safetensors",
            "transformer/diffusion.safetensors",
        ]


class TestDelete:
    def test_removes_every_writable_copy_and_the_old_cache(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        default, second, read_only = dirs
        for d in dirs:
            legacy_model(d / NAME)
        (default / "caches" / NAME).mkdir(parents=True)
        assert store.delete(MODEL_ID)
        assert not (default / NAME).exists() and not (second / NAME).exists()
        assert not (default / "caches" / NAME).exists()
        assert (read_only / NAME).exists()

    def test_nothing_to_delete(self, dirs: tuple[Path, Path, Path]) -> None:
        del dirs
        assert not store.delete(MODEL_ID)
