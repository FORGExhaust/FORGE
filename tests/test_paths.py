"""Tests for ``forge._paths`` — data directory locator."""

import os

from forge._paths import data_dir_path


class TestDataDirPath:

    def test_returns_string(self):
        result = data_dir_path()
        assert isinstance(result, str)

    def test_path_exists(self):
        result = data_dir_path()
        assert os.path.isdir(result)

    def test_contains_expected_subdirs(self):
        base = data_dir_path()
        for subdir in ["geqdsk", "json"]:
            assert os.path.isdir(os.path.join(base, subdir))
